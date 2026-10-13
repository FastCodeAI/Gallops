"""Minimal HTTP range-read file object over an HF resolve URL, usable by zipfile. Every byte fetched is counted."""
import os, requests, zipfile, time
class HttpRangeFile:
    def __init__(self, url, token, readahead=64*1024):
        self.url, self.readahead = url, readahead
        self.s = requests.Session(); self.s.headers["Authorization"] = f"Bearer {token}"
        h = self.s.head(url, allow_redirects=True, timeout=60); h.raise_for_status()
        self.final_url = h.url; self.size = int(h.headers["Content-Length"])
        self.accept_ranges = h.headers.get("Accept-Ranges"); self.pos = 0
        self.bytes_fetched = 0; self.requests = 0; self.cache = {}  # (start,end)->bytes
        self.log = []
    def _fetch(self, start, end):  # inclusive end
        t = time.time()
        r = self.s.get(self.final_url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
        if r.status_code != 206: raise RuntimeError(f"range not honoured: {r.status_code} {r.headers.get('Content-Range')}")
        b = r.content; self.bytes_fetched += len(b); self.requests += 1
        self.log.append((start, end, len(b), round(time.time()-t, 3)))
        return b
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        start, end = self.pos, min(self.pos + n, self.size) - 1
        if end < start: return b""
        for (cs, ce), cb in self.cache.items():
            if cs <= start and end <= ce:
                self.pos = end + 1; return cb[start-cs:end-cs+1]
        fe = min(max(end, start + self.readahead - 1), self.size - 1)
        b = self._fetch(start, fe); self.cache[(start, fe)] = b
        self.pos = end + 1; return b[:end-start+1]
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True
    def close(self): pass

def open_zip(url, token, readahead=64*1024):
    f = HttpRangeFile(url, token, readahead)
    return f, zipfile.ZipFile(f)

def fetch_member(zf, f, name):
    """Read one member with a single tight range (local header + data). Returns bytes."""
    zi = zf.getinfo(name)
    f.cache.clear(); f.readahead = 1024   # local header is 30 B + name + extra; read tight
    with zf.open(zi) as m: data = m.read()
    return data, zi
