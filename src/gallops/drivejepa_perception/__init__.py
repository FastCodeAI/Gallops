"""H15: a perception side-car on frozen Drive-JEPA, and the questions it is tested on.

The plan, with every rule and the reason for it: docs/drivejepa-perception-sidecar/first-pass-plan.md.
The record: experiments/H15_perception_sidecar/.

Everything that touches Drive-JEPA runs in its interpreter (Python 3.9, torch 2.1), entered through
`env.enter`, against `{code.drivejepa}` -- upstream e21f474 with paddock's patches 0001-0003 as
commits. Nothing is imported at package level, so the numpy-only parts stay importable from tests.
"""
