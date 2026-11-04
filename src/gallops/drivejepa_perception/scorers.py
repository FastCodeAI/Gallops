"""B1: four ways to give the scorer the side-car's agents, and the control that withholds them.

Modes (docs/drivejepa-perception-sidecar.md, section 5.1):
- `frozen` (B1.1): the release's scorer MLP unchanged; an input adapter adds to its input feature.
- `lora` (B1.2): as frozen, plus rank-r LoRA on the scorer MLP's three linear layers.
- `full` (B1.3): as frozen, with the scorer MLP itself trained.
- `replace` (B1.4): a new scorer built around agents; NAVSIM's sub-scores as named heads, the final
  score by NAVSIM's own formula.

**Arms.** C has the adapter's own-information half only: the candidate's shape, the ego status and
the release's logits, which is GC01's re-ranker inputs. F, O and S add the agent half: candidate ->
agent attention biased by the analytic clearances (geometry.py), and an MLP over the clearance
features. F reads the side-car's agents, O the ground truth, S another sample's. C against F is the
agents' information and nothing else.

**Exactly the release at step 0.** Modes frozen, lora and full output
    cached release logits + (scorer'(f + delta f) - scorer(f)),
both scorers on the same fp16 cached feature f. Every new path starts at zero (the adapter's output
layers, LoRA's B), and `full` starts as a copy, so the difference is exactly 0 at step 0. The pick
is then the release's own, bit for bit, though f was stored in fp16. The preflight checks it.

The scorer MLP is `drivejepa_b2d.head.Head`, Drive-JEPA's MLP(256, 1024, 6) with the same keys,
loaded with the release's tensors (`head_tensors_from_checkpoint`).

Its six logits are the last six PDM columns: DDC, EP, TTC, LK, HC, final. The release trained only
the last (`sub_score_weight = 0`).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from gallops.drivejepa_perception import geometry

#: The release head's columns among the nine PDM columns (pdm_labels.COLUMNS).
HEAD_COLUMNS = (3, 4, 5, 6, 7, 8)
#: The replacement scorer's heads: all eight sub-scores.
SUB_COLUMNS = (0, 1, 2, 3, 4, 5, 6, 7)
AGENT_FEATURES = 18
SELF_FEATURES = 256 + 24 + 11 + 6
MODES = ("frozen", "lora", "full", "replace")


def final_from_subscores(sub):
    """NAVSIM's aggregation (train_pdm_scorer._aggregate_pdm_scores), on (..., 8) sub-scores."""
    multiplicative = sub[..., 0] * sub[..., 1] * sub[..., 2] * sub[..., 3]
    weighted = (5 * sub[..., 4] + 5 * sub[..., 5] + 2 * sub[..., 6] + 2 * sub[..., 7]) / 14.0
    return multiplicative * weighted


def mlp(i, h, o, zero=False):
    net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
    if zero:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


def describe_agents(agents: dict):
    """(B, A, AGENT_FEATURES) from decoded agents: class, confidence, box, velocity, path, scaled."""
    cls = F.one_hot(agents["cls"].long().clamp(0, 2), 3).float()
    yaw = agents["yaw"][..., None]
    return torch.cat(
        [
            cls,
            agents["conf"][..., None],
            agents["xyz"][..., :2] / 20.0,
            agents["xyz"][..., 2:3] / 2.0,
            agents["lwh"].clamp(min=0.05).log(),
            torch.sin(yaw),
            torch.cos(yaw),
            agents["vel"] / 10.0,
            agents["path"].flatten(-2) / 5.0,
        ],
        dim=-1,
    )


class AgentRead(nn.Module):
    """Candidate -> agent attention with the clearance as a bias, and a null slot for empty scenes.

    Zero weights do not cancel the NaN of a softmax over nothing, so a learned null key is always
    there to attend to.
    """

    def __init__(self, d_query: int, d: int = 128, heads: int = 4):
        super().__init__()
        self.heads, self.d = heads, d
        self.encode = mlp(AGENT_FEATURES, d, d)
        self.q, self.k, self.v = nn.Linear(d_query, d), nn.Linear(d, d), nn.Linear(d, d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.bias_scale = nn.Parameter(torch.ones(heads))

    def forward(self, query, agents_x, pairwise, mask):
        b, k, _ = query.shape
        tokens = torch.cat([self.encode(agents_x), self.null.expand(b, 1, -1)], dim=1)  # B, A+1, d
        keep = torch.cat([mask, torch.ones(b, 1, dtype=torch.bool, device=mask.device)], dim=1)
        bias = torch.cat([-pairwise / 5.0, torch.zeros(b, k, 1, device=pairwise.device)], dim=-1)  # B, K, A+1
        h, dh = self.heads, self.d // self.heads
        q = self.q(query).view(b, k, h, dh).transpose(1, 2)
        kk = self.k(tokens).view(b, -1, h, dh).transpose(1, 2)
        v = self.v(tokens).view(b, -1, h, dh).transpose(1, 2)
        logits = q @ kk.transpose(-1, -2) / dh**0.5 + bias[:, None] * self.bias_scale[None, :, None, None]
        logits = logits.masked_fill(~keep[:, None, None, :], float("-inf"))
        weights = logits.softmax(dim=-1)  # B, h, K, A+1
        out = (weights @ v).transpose(1, 2).reshape(b, k, self.d)
        return out, weights.mean(dim=1)[..., :-1]  # per-agent weight, null dropped


class Adapter(nn.Module):
    """Adds to the scorer's input feature: own information always, agents in F, O and S."""

    def __init__(self, agents: bool):
        super().__init__()
        self.own = mlp(SELF_FEATURES, 256, 256, zero=True)
        self.agents = agents
        if agents:
            self.read = AgentRead(256)
            self.out = nn.Linear(128 + geometry.N_FEATURES, 256)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, f, own_x, agent_x=None, features=None, pairwise=None, mask=None):
        delta = self.own(torch.cat([f, own_x], dim=-1))
        attention = None
        if self.agents:
            read, attention = self.read(f, agent_x, pairwise, mask)
            delta = delta + self.out(torch.cat([read, features], dim=-1))
        return delta, attention


class LoRALinear(nn.Module):
    """A frozen Linear plus B A, rank r, B at zero: the layer is unchanged until it trains."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, base.in_features) * (1.0 / rank))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.a.t() @ self.b.t()) * self.scale


def with_lora(head: nn.Module, rank: int) -> nn.Module:
    """A copy of the head whose three Linear layers carry LoRA; everything else frozen."""
    head = copy.deepcopy(head)
    for p in head.parameters():
        p.requires_grad_(False)
    for i in (0, 3, 6):
        head.mlp[i] = LoRALinear(head.mlp[i], rank)
    return head


class Replacement(nn.Module):
    """B1.4: a scorer built around agents; eight named sub-scores, the final by NAVSIM's formula."""

    def __init__(self, agents: bool, d: int = 128):
        super().__init__()
        self.token = mlp(SELF_FEATURES, d, d)
        self.agents = agents
        if agents:
            self.read = AgentRead(d, d)
            self.mix = nn.Linear(d + geometry.N_FEATURES, d)
        self.across = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True)
        self.heads = nn.Linear(d, len(SUB_COLUMNS))

    def forward(self, own_in, agent_x=None, features=None, pairwise=None, mask=None):
        x = self.token(own_in)
        attention = None
        if self.agents:
            read, attention = self.read(x, agent_x, pairwise, mask)
            x = x + self.mix(torch.cat([read, features], dim=-1))
        x = self.across(x)
        sub = torch.sigmoid(self.heads(x))
        return sub, attention


class B1Model(nn.Module):
    def __init__(self, mode: str, agents: bool, head: nn.Module, rank: int = 16):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode, self.use_agents = mode, agents
        self.reference = copy.deepcopy(head)  # the release's scorer, never trained
        for p in self.reference.parameters():
            p.requires_grad_(False)
        if mode == "replace":
            self.replacement = Replacement(agents)
            return
        self.adapter = Adapter(agents)
        if mode == "frozen":
            self.head = self.reference
        elif mode == "lora":
            self.head = with_lora(head, rank)
        else:
            self.head = copy.deepcopy(head)
            for p in self.head.parameters():
                p.requires_grad_(True)

    def forward(self, batch: dict) -> dict:
        f = batch["pooled"].float()
        own_x = torch.cat(
            [batch["shape"], batch["ego"][:, None, :].expand(-1, f.shape[1], -1), batch["logits"]], -1
        )
        agent_x = features = pairwise = mask = nearest = None
        if self.use_agents:
            agents = batch["agents"]
            features, pairwise, nearest = geometry.terms(
                batch["candidates"],
                agents["xyz"][..., :2],
                agents["lwh"][..., :2],
                agents["vel"],
                agents["path"],
                agents["mask"],
            )
            agent_x = describe_agents(agents)
            mask = agents["mask"]
        if self.mode == "replace":
            sub, attention = self.replacement(torch.cat([f, own_x], -1), agent_x, features, pairwise, mask)
            final = final_from_subscores(sub)
            return {"sub": sub, "final": final, "score": final, "attention": attention, "nearest": nearest}
        delta, attention = self.adapter(f, own_x, agent_x, features, pairwise, mask)
        # The difference first: (a + b) - b is not a in floating point; a + (b - b) is.
        logits = batch["logits"] + (self.head(f + delta) - self.reference(f))
        return {"logits": logits, "score": logits[..., -1], "attention": attention, "nearest": nearest}


def loss(out: dict, batch: dict, w_sub: float = 0.5, w_imitation: float = 0.5, tau: float = 1.0):
    """BCE on the PDM labels plus listwise CE toward the human, the same in every mode and arm."""
    labels = batch["pdm"]  # B, K, 9
    target = F.softmax(-batch["ade"] / tau, dim=-1)
    if "logits" in out:
        logits = out["logits"]
        heads = labels[..., list(HEAD_COLUMNS)]
        bce_final = F.binary_cross_entropy_with_logits(logits[..., -1], heads[..., -1])
        bce_sub = F.binary_cross_entropy_with_logits(logits[..., :-1], heads[..., :-1])
        ranking = logits[..., -1]
    else:
        sub = out["sub"].clamp(1e-6, 1 - 1e-6)
        bce_sub = F.binary_cross_entropy(sub, labels[..., list(SUB_COLUMNS)])
        bce_final = F.binary_cross_entropy(out["final"].clamp(1e-6, 1 - 1e-6), labels[..., -1])
        ranking = torch.logit(out["final"].clamp(1e-6, 1 - 1e-6))
    imitation = -(target * F.log_softmax(ranking, dim=-1)).sum(-1).mean()
    total = bce_final + w_sub * bce_sub + w_imitation * imitation
    return total, {"bce_final": float(bce_final), "bce_sub": float(bce_sub), "imitation": float(imitation)}
