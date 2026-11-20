"""Paddock adapter run3b_v2 vs the trainer's eval path (train3.model_fn + eval_v2.scores_from_pred, 'head' pick) on the
first 64 of eval_v2's 1,008 NAVSIM held-out samples, same net object, same inputs (the data object's images).
Checks: picks equal, proposals equal; prints the pick's EPDMS (eval_v2.epdms) for n 64."""
import os, sys, json
import numpy as np, torch
V2 = "{runs}/run3b/v2"
for p in (f"{V2}/model", f"{V2}/data", f"{V2}/eval", "{runs}/run3b/r3b", "{runs}/run3b/gpulab",
          "{runs}/run3a-adapted", "{data.drivejepa_tree}/navsim_v2",
          "{runs}/worktrees/gate-oct27/src"):
    sys.path.insert(0, p)
os.environ.update(RUN3B_CHECKPOINT="{data.paddock_data}/run3b/runs_v2/run3b_v2b_19nov/trainable_step023859_final.pt",
                  RUN3B_CODE=V2, RUN3B_R3B="{runs}/run3b/r3b", RUN3B_SCORE_DOMAIN="navsim")
from paddock.adapters.models.run3b_v2.policy import Run3bV2Policy
def main():
    import run3b_data2 as D2, eval_v2 as EV
    import time; T0 = time.time()
    pol = Run3bV2Policy(device="cuda"); pol.load(); print("loaded", round(time.time() - T0), "s", flush=True)
    print(json.dumps({k: v for k, v in pol.ckpt_info.items() if k != "encoder_load"}), flush=True)
    net, ag = pol.net, pol.agent
    data = D2.Run3BData2("heldout", seed=12345); data.deterministic = True; print("data", round(time.time() - T0), "s", flush=True)
    nav_i = np.linspace(0, len(data.nav) - 1, 1008).round().astype(int)[:64]
    same, pdiff, toks, trajs, sdiff, gaps = 0, 0.0, [], [], 0.0, []
    for s in range(0, 64, 32):
        idx = [int(g) for g in nav_i[s:s + 32]]
        b = D2.collate2([data[g] for g in idx]); B = len(idx)
        with torch.no_grad():
            clips = D2.to_clips(b["images"].cuda())
            feats = {"clips": clips.view(B, 4, *clips.shape[1:]), "lidar2img": b["lidar2img"].cuda().float(),
                     "ego_status": b["ego_status"].cuda().float()}
            out = ag.forward(feats)
            # reference: train3.model_fn's decoded-agents pass + eval_v2's head pick
            l2i = b["lidar2img"].cuda().float()
            tokens = net.encoder(clips, group=16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, agents = net.decode(tokens, l2i)
                pl = net.plan(tokens, l2i, b["ego_status"].cuda(), agents)
                paths = torch.cat([b["paths"].cuda(), pl["proposals"].float()], 1)
                logits, numbers = net.read(paths, tokens, l2i, agents)
            em = pl["expert_metres"].float().cpu()
            sc = EV.scores_from_pred(logits[:, -32:].cpu(), numbers[:, -32:].cpu(), em, em[:, -1] < 0.5, torch.ones(B, dtype=torch.bool), 0.108)
        pk_ref = sc.argmax(1).numpy(); pk = out["pdm_score"].argmax(1).cpu().numpy()
        sdiff = max(sdiff, float((out["pdm_score"].float().cpu() - sc).abs().max()))
        for i in np.where(pk != pk_ref)[0]:
            srt = np.sort(sc[i].numpy())[::-1]; gaps.append((int(idx[i]), float(srt[0] - srt[1]), float(sc[i, pk[i]]), float(sc[i, pk_ref[i]])))
        same += int((pk == pk_ref).sum()); pdiff = max(pdiff, float((out["proposals"].cpu() - pl["proposals"].float().cpu()).abs().max()))
        for i, g in enumerate(idx):
            m, lm = b["meta"][i], data.label_meta(g)
            toks.append(lm.get("mc") or ((m["log"], m["token"]) if m.get("log") else m["token"]))
            trajs.append(out["trajectory"][i].float().cpu().numpy()[None])
    print("forward done", round(time.time() - T0), "s", flush=True)
    print("picks equal", same, "/ 64; proposals max abs diff", pdiff)
    print("score max abs diff", sdiff, "; differing samples top-2 gap (ref)", gaps)
    e = EV.epdms(toks, np.stack(trajs).astype(np.float32), workers=8)
    print("pick EPDMS n", len(e), "mean", float(np.mean(e)))


if __name__ == "__main__":
    main()
