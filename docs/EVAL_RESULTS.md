# Evaluation Results

> **STALE -- BLOCKED ON DEV 3.** Per `Objectives v3`, these numbers predate
> every fix in that document (train.py crash fix, max_len 32->48, pre-flight
> check) and must not be treated as current model quality. No
> `checkpoints/final/best.pt` exists in this environment yet (`checkpoints/`
> is gitignored and train.py currently cannot complete a run -- see
> Dev 3 objective 1, the `model()` call with undefined `src_seq`/`tgt_in`).
>
> Dev 2's regeneration objective is unblocked only once Dev 3 lands:
> 1. the crash fix, 2. `max_len` raised to >=48, 3. a completed training run.
>
> Once that checkpoint lands, regenerate this file with:
> `python eval/run_eval.py`
> Do not hand-edit the table below in the meantime.

**Checkpoint Evaluated:** `checkpoints\final\best.pt`

| Operation | Total Problems | Exact Match (Accuracy) | Verification Rate |
|---|---|---|---|
| diff | 80 | 19/80 (23.8%) | 19/80 (23.8%) |
| gradient | 50 | 0/50 (0.0%) | 0/50 (0.0%) |
| integrate | 60 | 37/60 (61.7%) | 37/60 (61.7%) |
| partial | 60 | 9/60 (15.0%) | 9/60 (15.0%) |
| tangent_line | 50 | 0/50 (0.0%) | 0/50 (0.0%) |
| **Overall** | **300** | **65/300 (21.7%)** | **65/300 (21.7%)** |
