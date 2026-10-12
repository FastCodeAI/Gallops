"""Run 3 raw-dataset IO helpers: whole-file download with sha256 + manifest, and exam-safe member reads."""
import os, time, hashlib, requests, common
def url(path): return f"https://huggingface.co/datasets/{common.RAW_REPO}/resolve/main/{path}"
def download(path, tag):
    """Stream one repo file to RAW_NAS/<path>; returns (bytes, secs). Skips if present."""
    dst = common.RAW_NAS / path
    if dst.exists(): return dst.stat().st_size, 0.0
    dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_suffix(dst.suffix + ".part"); h = hashlib.sha256(); n = 0; t = time.time()
    for attempt in range(4):
        try:
            with requests.get(url(path), headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}, stream=True, timeout=120) as r:
                r.raise_for_status(); h = hashlib.sha256(); n = 0
                with open(tmp, "wb") as f:
                    for b in r.iter_content(1 << 20): f.write(b); h.update(b); n += len(b)
            break
        except Exception as e:
            if attempt == 3: raise
            time.sleep(5 * (attempt + 1))
    tmp.rename(dst); common.manifest_add(tag, path, n, str(dst), h.hexdigest())
    return n, time.time() - t
