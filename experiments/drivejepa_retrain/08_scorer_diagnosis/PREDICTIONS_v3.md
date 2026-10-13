# S6 scorer diagnosis — predictions, version 3 (adds P-a4 to version 2)

Written 2026-10-07 on Sid's instruction, before any of a, b, c was run (the start gate has not opened; no model has been run).
PREDICTIONS_v2.md (sha256 c3f4057afbb1c3a4e82a2758be402f9acf44e704fcfb4f2a2342a0732262ab60, hashed 2026-10-07T22:58:57+05:30) stays in
force word for word and is not repeated here; PREDICTIONS.md (version 1, sha256
8791c38f36df15ed2852b9efbcac174b826023b071d6bc3d3d24d646c9d51c1b) is kept unchanged. This file adds one prediction and is hashed and not
edited afterwards.

## Added prediction

- **P-a4**. If suspect 8 is most of the story, the share of UNSEEN picks graded ≥ 0.9 on pass 9 exceeds the 1.5 m human-match rate by
  more than 30 points; otherwise the gap is less than 10 points.
  Reasoning: S4's 15 % asks the pick to end within 1.5 m of the human path at 4 s; a pick can end elsewhere and still earn a high grade.
  How it is measured: both rates on the same picks (pass 9 on its own menu, raw score), over all UNSEEN graded examples, and per slice.
  A gap between 10 and 30 points is reported as "between the two readings".

## What I had already seen when this was written (disclosure)

From S4's stored pass-9 readout on UNSEEN: the share of picks graded ≥ 0.9 is 48.8 % over all graded examples (seen); the 1.5 m
human-match rate is 14.8 % on close-pass examples (seen, 1,570 examples). I have **not** computed the 1.5 m rate over all UNSEEN
examples or on highway / ordinary, nor the ≥ 0.9 share per slice. So P-a4 is partly not blind: one side of the all-UNSEEN gap is known,
and the close-pass 1.5 m rate is known.
