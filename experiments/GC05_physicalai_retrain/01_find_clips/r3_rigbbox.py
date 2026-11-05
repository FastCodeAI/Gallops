"""1b: per-scene ego box from the usdz: rig_trajectories.json -> rig_trajectories[0].rig_bbox {centroid, dim, rot} (rig frame).
Only the first 4 KB of the STORED member is range-read (rig_bbox sits at byte ~500). nurec_train + nurec_val scenes only."""
import os, re, json, struct, hashlib, concurrent.futures as cf, common, hfrange
V = common.load_split_views(); _, exam, _ = common.load_views(); todo = sorted(V["nurec_train"] | V["nurec_val"]); assert not (set(todo) & exam)
def one(u):
    out = common.NAS / "pool" / u / "rig_bbox.json"
    if out.exists(): return u, 0
    for a in range(4):
        try:
            f, zf = hfrange.open_zip(f"https://huggingface.co/datasets/{common.REPO}/resolve/main/{common.RELEASE}/{u}/{u}.usdz", os.environ["HF_TOKEN"])
            zi = zf.getinfo("rig_trajectories.json"); raw = f._fetch(zi.header_offset, zi.header_offset + 4095)
            nl, el = struct.unpack("<HH", raw[26:30]); head = raw[30 + nl + el:].decode("utf-8", "replace")
            m = re.search(r'"rig_bbox"\s*:\s*(\{[^{}]*\})', head); bb = json.loads(m.group(1)); break
        except Exception as e:
            if a == 3: print("FAILED", u, repr(e), flush=True); return u, -1
    s = json.dumps(bb); out.write_text(s); common.manifest_add(u, "rig_trajectories.json[0:4096]::rig_bbox", len(raw), str(out), hashlib.sha256(s.encode()).hexdigest())
    return u, f.bytes_fetched
with cf.ThreadPoolExecutor(6) as ex: r = list(ex.map(one, todo))
print("scenes", len(r), "failed", sum(1 for x in r if x[1] < 0), "bytes", sum(max(x[1], 0) for x in r))
