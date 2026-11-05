#!/usr/bin/env python
"""S4: assemble the candidate weights file from one saved pass, in upstream's checkpoint layout (state_dict keys 'agent._pad_model.*',
what DriveJEPAAgent.initialize loads): encoder tensors bit-for-bit from the shipped checkpoint, generator + scorer from the pass.
Also fills the 0.1.6 patch placeholders and writes adapter_0.1.6.diff. Usage (conda drive-jepa python, CPU): build_candidate.py <pass> [tag]"""
import sys, json, hashlib, pathlib, subprocess, torch
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "05_small_overfit_check/train_patch")); import s35_common as C
p = int(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "candidate"; ck = torch.load(C.NAS / f"06_first_training_run/run/ckpt/pass_{p:02d}.pt", map_location="cpu"); assert ck["passno"] == p
base = torch.load(C.CKPT, map_location="cpu")["state_dict"]; new = {}; n_enc = n_new = n_changed = 0
for k, v in base.items():
    kk = k.replace("agent._pad_model", "_pad_model")
    if "_backbone" in kk: new[k] = v; n_enc += 1; assert kk not in ck["trainable"]
    else:
        w = ck["trainable"][kk]; assert w.shape == v.shape and w.dtype == v.dtype, k; new[k] = w.clone(); n_new += 1; n_changed += int(not torch.equal(w, v))
assert set(ck["trainable"]) == {k.replace("agent._pad_model", "_pad_model") for k in base if "_backbone" not in k}
out = C.NAS / f"06_first_training_run/{tag}"; out.mkdir(parents=True, exist_ok=True); f = out / f"drive_jepa_s4_pass{p:02d}.ckpt"; torch.save({"state_dict": new, "s4": dict(passno=p, base_sha256=C.CKPT_SHA256)}, f)
h = hashlib.sha256()
with open(f, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
info = dict(path=str(f), sha256=h.hexdigest(), bytes=f.stat().st_size, passno=p, tensors_from_shipped_encoder=n_enc, tensors_from_s4=n_new, s4_tensors_that_differ_from_shipped=n_changed, base_checkpoint=str(C.CKPT), base_sha256=C.CKPT_SHA256)
json.dump(info, open(HERE / f"{tag}.json", "w"), indent=1); print(json.dumps(info, indent=1))
if tag == "candidate":
    b = HERE / "adapter_016/b/adapters/drivejepa016/constants.py"; s = b.read_text()
    if "@S4_CANDIDATE_PATH@" in s: b.write_text(s.replace("@S4_CANDIDATE_PATH@", str(f)).replace("@S4_CANDIDATE_SHA256@", info["sha256"]).replace("0  # @S4_CANDIDATE_BYTES@", str(info["bytes"])))
    d = subprocess.run(["diff", "-u", "-r", "a/adapters/drivejepa014b", "b/adapters/drivejepa016"], cwd=HERE / "adapter_016", capture_output=True, text=True).stdout
    (HERE / "adapter_0.1.6.diff").write_text(d); print("adapter_0.1.6.diff", len(d.splitlines()), "lines (NOT applied)")
