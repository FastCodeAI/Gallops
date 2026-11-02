"""A memory for Drive-JEPA's picker: a small state-space layer that carries a summary from one moment
to the next, trained on navtrain. Its plan is the memory-picker plan; this package holds its steps.

    ticks   numpy only -- which navtrain frames are usable moments, how they chain, and which are
            graded; plus the driving statistics the missing-image check compares
    inputs  the model's inputs for one moment, built the way paddock's node builds them on NAVSIM
    cache   torch -- the released model run once over every moment, its outputs on disk

Everything here runs inside Drive-JEPA's own interpreter (Python 3.9) as well as gallops' 3.11, so
it stays valid Python 3.9, and nothing is imported at package level.
"""
