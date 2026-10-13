"""S10 task 7: load one generator checkpoint into the model for the dumps (s10_dump_gen.py, s10_dump_pullout.py). No side effects at import."""


def load_ckpt(m, path, torch):
    """Loads generator + scorer tensors of `path` into m (the shipped encoder is already in m). `path` is a pass file ('trainable') or a
    merged file ('state_dict', upstream layout) whose encoder tensors must equal the loaded shipped encoder bit for bit. -> description."""
    sd = torch.load(path, map_location="cpu")
    if "trainable" in sd:
        w = {k[len("_pad_model."):]: v for k, v in sd["trainable"].items()}; kind = f"pass file, passno {sd.get('passno')}"
    else:
        full = {k[len("agent._pad_model."):]: v for k, v in sd["state_dict"].items()}; own = m.state_dict(); enc = [k for k in full if k.startswith("_backbone")]
        assert enc and all(torch.equal(full[k], own[k]) for k in enc), "merged file: encoder tensors differ from the shipped encoder"
        w = {k: v for k, v in full.items() if not k.startswith("_backbone")}; kind = f"merged file, {len(enc)} encoder tensors equal to the shipped encoder, info {sd.get('s10')}"
    missing, unexpected = m.load_state_dict(w, strict=False); assert not unexpected and all(k.startswith("_backbone") for k in missing), (missing[:3], unexpected[:3])
    return f"{kind}, {len(w)} tensors loaded"
