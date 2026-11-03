# C.2 projection, written 2026-10-09T00:16:47+05:30 before the full pull-out run is launched

Timing run: 10 pull-out clips (seeded draw), 320 examples, all 32 pass-9 plans, 6 threads + 2 grading workers + 2 loaders, CPU only.
- wall 244 s = 0.76 s per example (encoder 0.59, trunk 0.02, map-free checks 0.09 for both settings, loading 0.01)
- CPU 1,302 s = 4.07 CPU-s per example (includes start-up)
- own memory peak 7.5 GB; split forward equals the model's own forward exactly (0.0 difference, same pick)

Projection for the full set (283 clips in the manifest, 9,053 examples; the brief's 285 clips would be about 9,120):
- wall 9,053 x 0.76 s = about 1.9 hours, longer if the gate pauses it
- CPU about 10 CPU hours (9,053 x 4.07 s = 10.2)
- NAS: about 0.6 MB per example before merging -> about 0.3 GB for the shards and the same again for the merged file
