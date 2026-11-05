"""STEP C: range-read pilot on 3 seeded scenes from the candidate-A training pool (outside every exam candidate)."""
import os, sys, json, random, time, hashlib, io, pathlib, common, hfrange
import pandas as pd, pyarrow.parquet as pq
excl = common.load_exclusions()
strict = {l.strip() for l in open(common.HERE/"exam_exclusions_strict_our30.txt") if l.strip() and not l.startswith("#")}
assert strict <= excl
pool = [l.strip() for l in open(common.HERE/"training_pool_candA.txt") if l.strip()]
assert not (set(pool) & excl), "pool overlaps exclusions"
SEED = 20261001
rng = random.Random(SEED); pilot = rng.sample(pool, 3)
print("seed", SEED, "pilot scenes", pilot)
open(common.HERE/"c_pilot_uuids.txt","w").write("\n".join(pilot)+"\n")
WANT = ["clipgt/egomotion_estimate.parquet","clipgt/obstacle.parquet","clipgt/lane.parquet","clipgt/drivable_space.parquet",
        "sequence_tracks.json","rig_trajectories.json","map.xodr","metadata.yaml","data_info.json"]
tok = os.environ["HF_TOKEN"]; out = {}
for i, u in enumerate(pilot):
    assert u not in excl
    url = f"https://huggingface.co/datasets/{common.REPO}/resolve/main/{common.RELEASE}/{u}/{u}.usdz"
    t0 = time.time(); f, zf = hfrange.open_zip(url, tok, readahead=64*1024)
    members = zf.infolist()
    rec = dict(uuid=u, usdz_size=f.size, accept_ranges=f.accept_ranges, final_host=f.final_url.split("/")[2],
               n_members=len(members), cd_bytes=f.bytes_fetched, cd_requests=f.requests, members=[
               dict(name=m.filename, size=m.file_size, comp=m.compress_type, csize=m.compress_size, offset=m.header_offset) for m in members])
    if i == 0:
        print(f"\n[{u}] usdz {f.size/1e6:.1f} MB host={rec['final_host']} Accept-Ranges={f.accept_ranges} members={len(members)} (central dir via {f.requests} req, {f.bytes_fetched} B)")
        for m in members: print(f"   off={m.header_offset:>11d} size={m.file_size:>11d} comp={m.compress_type} {m.filename}")
    names = {m.filename for m in members}
    # match wanted names by suffix (layout inside usdz may be nested)
    got = {}
    d = common.NAS / "pilot" / u; d.mkdir(parents=True, exist_ok=True)
    for w in WANT:
        cands = [n for n in names if n == w or n.endswith("/"+w) or n.split("/")[-1] == w.split("/")[-1]]
        if not cands: got[w] = None; continue
        n = cands[0]; b0 = f.bytes_fetched; t = time.time()
        data, zi = hfrange.fetch_member(zf, f, n)
        p = d / n.replace("/", "__"); p.write_bytes(data)
        common.manifest_add(u, n, len(data), str(p), hashlib.sha256(data).hexdigest())
        got[w] = dict(member=n, bytes=len(data), fetched=f.bytes_fetched-b0, secs=round(time.time()-t,2))
    rec["fetched"] = got; rec["total_bytes_fetched"] = f.bytes_fetched; rec["total_requests"] = f.requests
    rec["secs"] = round(time.time()-t0, 1); rec["frac_of_usdz"] = rec["total_bytes_fetched"]/f.size
    out[u] = rec
    print(f"[{u}] fetched {rec['total_bytes_fetched']/1e6:.2f} MB in {rec['total_requests']} req, {rec['secs']} s = {100*rec['frac_of_usdz']:.2f}% of usdz")
    for w, g in got.items():
        line = "MISSING" if g is None else f"{g['bytes']:>10d} B  {g['secs']:>5.2f} s  ({g['member']})"
        print(f"   {w:40s} {line}")
json.dump(out, open(common.HERE/"c_pilot_results.json","w"), indent=1)
# --- C3 schemas on first scene ---
u = pilot[0]; d = common.NAS/"pilot"/u
print("\n=== C3 schemas ===")
for p in sorted(d.iterdir()):
    print("\n##", p.name, p.stat().st_size, "B")
    if p.suffix == ".parquet":
        t = pq.read_table(p); print(t.schema)
        df = t.to_pandas(); print(df.head(3).to_string()[:1500]); print("rows", len(df))
        md = t.schema.metadata or {}
        for k, v in md.items():
            if k != b"pandas": print("meta", k, v[:300])
    elif p.suffix in (".json",):
        j = json.loads(p.read_bytes()); 
        def summ(o, depth=0):
            if depth > 2: return type(o).__name__
            if isinstance(o, dict): return {k: summ(v, depth+1) for k, v in list(o.items())[:12]}
            if isinstance(o, list): return [f"list[{len(o)}]", summ(o[0], depth+1) if o else None]
            return o if isinstance(o, (int, float, str, bool)) and len(str(o)) < 80 else type(o).__name__
        print(json.dumps(summ(j), indent=1)[:2500])
    else:
        print(p.read_text(errors="replace")[:1500])
# --- C4 projection ---
n_pool = len(pool); tot = sum(r["total_bytes_fetched"] for r in out.values()); secs = sum(r["secs"] for r in out.values())
print(f"\n=== C4 === 3 scenes: {tot/1e6:.1f} MB in {secs:.0f} s -> per scene {tot/3/1e6:.2f} MB, {secs/3:.1f} s")
print(f"projection for {n_pool} pool scenes (serial): {n_pool*tot/3/1e9:.2f} GB, {n_pool*secs/3/3600:.2f} h; with 8 workers ~{n_pool*secs/3/3600/8:.2f} h")
