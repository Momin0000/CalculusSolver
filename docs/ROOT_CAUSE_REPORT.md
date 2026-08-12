# CalculusSolver — Partial-Accuracy Regression: Root-Cause Report

**Scope of this run:** Read-only. Checkpoint(s) evaluated, benchmark suite in `eval/benchmarks/`,
and repo history inspected. No model/training code, vocab, or data files were modified.

---

## 0. Headline finding — the numbers in the brief don't reproduce

Before anything else: I could not reproduce **43.3% / 28.3% partial accuracy** or **66.7% integrate
accuracy** with anything in this upload. Running the eval harness against the only checkpoint present
(`checkpoints/final/best.pt`) gives:

| Operation | N | Exact match |
|---|---|---|
| diff | 80 | 0/80 (0.0%) |
| gradient | 50 | 0/50 (0.0%) |
| integrate | 60 | 0/60 (0.0%) |
| partial | 60 | 0/60 (0.0%) |
| tangent_line | 50 | 0/50 (0.0%) |
| **Overall** | **300** | **0/300 (0.0%)** |

This matches the `docs/EVAL_RESULTS.md` already committed in the repo — it's not a fluke of my run.
So the failure CSV below has **300 rows, not 214**: every single benchmark problem currently fails,
not a partial subset. I built the CSV from what's actually reproducible rather than trimming to match
the expected count.

This is worth flagging to the team directly: whoever reported 43.3%/66.7% was either looking at a
different checkpoint, a different eval harness, or a different point in time than what's in this
upload. That discrepancy is itself a process gap (results aren't tied to a checkpoint hash/commit
anywhere) and should be fixed regardless of the rest of this report.

## 1. Stage isolation (SFT vs. final) — could not be done

`checkpoints/sft/best.pt` **does not exist** anywhere in this upload, in git history, or on any
reachable path — `checkpoints/` is gitignored (`.gitignore` line: `checkpoints/`), so it was never
tracked, and only `checkpoints/final/best.pt` was provided. I can't isolate "Stage 2 fine-tune vs.
Stage 3 hard-example upweighting" without that artifact.

More importantly: **that staged pipeline doesn't exist in this codebase.** I searched the full repo
for any staged-training, hard-example-upweighting, or verifier-loop-curriculum logic
(`hard_example`, `upweight`, `verifier loop`, `stage2`/`stage3`, `curriculum`) and found nothing.
`train.py` is a single flat training loop (`run_training_pipeline()`), and `docs/TRAINING_RESULTS.md`
documents one run of up to 5 epochs, not two distinct fine-tuning stages. If "Stage 2 fine-tune" and
"Stage 3 hard-example upweighting" exist, they live in a different repo, a different branch, or
someone's local scripts — not here. **Recommendation: get the actual SFT checkpoint and the
upweighting code from whoever produced the 43.3%→28.3% numbers before the team invests more time
narrowing this down on the artifacts in this upload.**

## 2. What's actually broken — a decoder that never terminates correctly

I ran the model directly (not just the harness) to see what it produces. Example, differentiating
`∂/∂z (2z³ + 8x³)`:

```
NODE:FRAC STRUCT:OPEN STRUCT:NUMI STRUCT:OPEN NODE:FRAC STRUCT:OPEN STRUCT:NUMI STRUCT:OPEN
STRUCT:CLOSE STRUCT:SEP STRUCT:DENO STRUCT:OPEN STRUCT:CLOSE STRUCT:CLOSE STRUCT:CLOSE ...
```

This is empty `NODE:FRAC` wrappers nesting into each other with no coefficient, variable, or exponent
content — for dozens of tokens. I re-ran a sample at `max_len=128` (4x the configured 32) to check
whether this was simply a length-cap problem. **It was not**: at 128 tokens the model was *still*
mid-structure and still 0/10 correct, and in one case emitted the exact same `VAR:x EXP:-2` pair
seven times in a row before I cut it off — a repetition loop, not a near-miss that just needed more
room. **All 300/300 failures hit the token cap and show the identical deserializer error
signature** (`"Expected token ... but reached end of tokens"`), at both `max_len=32` and `max_len=128`.

So: `max_len=32` in `config.json` is a real and separate problem (see §3), but it is **not the primary
cause of the 0% score**. The decoder itself never learns to close out a valid AST — it doesn't reach
a wrong-but-complete answer that then gets cut off; it gets stuck in unproductive structural
repetition and never converges on real content at all. Raising `max_len` won't fix this on its own.

This is corroborated by the model's own training log — `docs/TRAINING_RESULTS.md` — which is arguably
the most important piece of evidence in the whole investigation:

| Epoch | Val Loss | Val Verify | Checkpoint Saved |
|---|---|---|---|
| 1 | 1.2753 | **0.0000** | Yes |
| 2 | 1.6560 | 0.0000 | No |
| 3 | 1.6567 | 0.0000 | No |
| 4 | 1.6595 | 0.0000 | No |
| 5 | 1.6533 | 0.0000 | No |

**Verification rate was 0.0000 on every single epoch, including the one that got saved as "best."**
Val loss also got *worse* after epoch 1 and never recovered. `best.pt` isn't a good checkpoint that
regressed later in a downstream fine-tune — it's the least-bad snapshot of a run that never once
produced a verifiably correct output at any point during training. Whatever regression narrative
exists between 43.3% and 28.3%, this specific checkpoint was never in a "43.3%-good" state to regress
from.

One more provenance red flag: `config.json` currently says `"epochs": 2`, but `TRAINING_RESULTS.md`
reports 5 epochs actually run. Those two files are inconsistent with each other, which means the
checkpoint on disk may not even correspond to the config currently committed. I'd treat any
per-stage comparison as unreliable until checkpoint↔config↔commit provenance is nailed down.

## 3. Truncation vs. wrong rule — both are real, but truncation is upstream and dominant

Per the deliverable, these need different fixes, so here's the split, backed by data (see
`failed_predictions.csv`):

- **Truncation (length/config limit):** 300/300 failures hit the `max_len=32` token cap, and the
  deserializer error in every single case is the "ran out of tokens mid-structure" signature, not a
  "grammar violation" or "wrong token type" signature. This is the dominant, upstream failure mode —
  nothing downstream (rule selection, coefficient prediction) gets a chance to be evaluated because
  the sequence is never even syntactically complete.
- **Wrong rule selection:** also real, but secondary and only checkable for the operations that
  actually have a `RULE:` token to predict (diff/integrate/partial — gradient and tangent_line don't
  map onto the `rule_tokens` vocabulary at all, which is its own labeling inconsistency worth fixing
  in the benchmark files). Where it's checkable: the rule classifier collapses hard.
  `predicted_rule` came back as `RULE_0` (= `power_rule`) for **234 of 300** predictions and `RULE_7`
  (= `partial_derivative`) for the other 16 (all within `partial`). It never predicted any of the
  other 8 rule classes. That's consistent with a classifier head that's essentially memorized the
  majority class rather than learned to discriminate — for `integrate`, it's confidently wrong every
  time (predicts `power_rule`, true label is `power_rule_integral`).

  Note also: `model/transformer.py` hardcodes `rule_labels = [f"RULE_{i}" for i in range(num_rules)]`
  for `.pt` checkpoints instead of using the real names loaded from `vocab.json`'s `rule_tokens` (which
  `inference/solve.py` computes but only wires into the *other* model-loading branch, for `.pkl`
  checkpoints). This is a real bug — every `.pt`-based prediction reports a meaningless `RULE_N` label
  — and I mapped it back to real names manually for the CSV (`predicted_rule_mapped` column) using the
  vocab's `rule_tokens` ordering. Worth a one-line fix regardless of the accuracy issue.

**Bottom line on this question:** fix the truncation/non-termination problem first — it's blocking
100% of predictions before rule selection is even reachable in the pipeline. Rule selection quality
can't be meaningfully assessed until sequences complete.

## 4. Hard-example set under-representation — nothing to check

The brief asks whether the hard-example set from "the verifier loop" under-represents `partial`
relative to its true error rate. As in §1, I could not find a verifier loop, a hard-example set, or
any upweighting/curriculum code anywhere in this repository (`train.py`, `problem_generator.py`,
`run_pipeline.py`, all searched). There's nothing to audit. If this exists, it's not in this upload —
flag to whoever owns that pipeline rather than treating it as verified absent here.

## 5. Sample review of "10 correct integrate predictions" — there are none

`integrate` scored 0/60 (0.0%), not 66.7%, so there is no set of correct predictions to sample. I
pulled 10 `integrate` predictions anyway to characterize what's actually happening: all 10 hit the
32-token cap, all 10 fail to deserialize with the same "ran out of tokens" error, and all 10 got the
same `power_rule` rule prediction against a true label of `power_rule_integral`. If a 66.7% number
exists for `integrate` somewhere, it was not produced by this checkpoint against this benchmark file —
the two are inconsistent with each other, so the "was it inflated by easy problems" question doesn't
apply to what's in this upload; the question that actually needs answering is *which* checkpoint/eval
produced that number.

## 6. Verdict and recommendation

**Root cause:** The checkpoint in this upload (`checkpoints/final/best.pt`) never converged to
producing complete, valid output sequences at any point in its recorded training history (0%
verification on every logged epoch), and its decoder gets stuck in unproductive repetitive/nested
structural token loops that don't resolve even when given 4x the configured token budget. The
`max_len=32` config value compounds this by capping already-broken generations early, but is not the
primary cause — it's a second, independent bug worth fixing on its own merits. The rule-classification
head has also collapsed to predicting the majority class, and a labeling bug (`RULE_i` placeholder
names for `.pt` checkpoints) is currently masking that from view in any inference output. None of this
lines up with a 43.3%→28.3% regression story, because this checkpoint doesn't test out anywhere near
43.3% to begin with — the artifacts needed to investigate *that* specific regression (the SFT
checkpoint, and the Stage 2/3 training code) aren't present in this upload.

**Recommendation: neither "fix the upweighting logic" nor "roll back" — get the missing artifacts
first, then re-baseline.** Concretely:
1. Get `checkpoints/sft/best.pt` (or wherever it actually lives) and the Stage 2/3 training code from
   whoever produced the original 43.3%/66.7% numbers — that comparison can't be done on what's here.
2. Independently of that: this `final/best.pt` checkpoint should not go anywhere near production as-is
   — it fails 100% of the benchmark suite. Retraining (not rolling back to it) is the right call for
   *this specific artifact*.
3. Fix the two concrete bugs found along the way regardless of the outcome above: the hardcoded
   `RULE_i` labels in `model/transformer.py`, and the `config.json` (epochs: 2) vs.
   `TRAINING_RESULTS.md` (5 epochs run) provenance mismatch, so future checkpoints can be traced to
   the config that produced them.
4. Reconcile the `expected_rule` labels in `benchmark_gradient.json`/`benchmark_tangent_line.json`
   with the actual `RULE:` vocabulary before relying on rule-accuracy numbers for those operations.

---

## Appendix — how to reproduce

```
pip install torch==2.3.1 numpy==1.26.4 joblib==1.4.2 --break-system-packages
python eval/run_eval_detailed.py checkpoints/final/best.pt 32 out.json   # per-benchmark op arg optional
```

`eval/run_eval_detailed.py` (new, added under `eval/`, not wired into any existing pipeline) is a
drop-in extension of the existing `eval/run_eval.py` that also records per-problem predicted output,
predicted rule, token counts, and truncation flags — needed to build `failed_predictions.csv`. It
reuses the existing `is_equivalent` / `categorize_error_v1` logic already in `inference/eval_harness.py`
rather than inventing new equivalence logic.