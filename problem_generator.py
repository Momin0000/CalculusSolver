import json
import random
from pathlib import Path

# ── Vocab-safe ranges ────────────────────────────────────────────────────────
# These must match the tokenizer/vocab.json ranges exactly.
# Coefficients: COEF:-10 to COEF:12 (integers only, skip COEF:OTHER/COEF:100)
SAFE_COEFFS = list(range(-10, 11)) + [12]
# Positive coefficients only (for cases where we need non-zero positive)
SAFE_POS_COEFFS = [c for c in SAFE_COEFFS if c > 0]
# Non-zero coefficients
SAFE_NONZERO_COEFFS = [c for c in SAFE_COEFFS if c != 0]
# Symmetric subset: only used where a value AND its negation must both exist
# as vocab tokens (the coefficient range is asymmetric: -10 to 12, so 12's
# negation, -12, is not a valid token). Needed for cos's derivative, which
# introduces a negative sign via the coeff decorator.
SAFE_SYMMETRIC_NONZERO_COEFFS = [c for c in SAFE_NONZERO_COEFFS if -c in SAFE_COEFFS]
# Exponents: EXP:-3 to EXP:5 (integers only, skip EXP:OTHER)
SAFE_EXPONENTS = list(range(-3, 6))
# Positive exponents for power rule differentiation (need power >= 1 for non-trivial result)
SAFE_POS_EXPONENTS = [e for e in SAFE_EXPONENTS if e >= 1]
# Variables
VARIABLES = ["x", "y", "z"]

# ── Rule IDs ─────────────────────────────────────────────────────────────────
# IMPORTANT: these are NOT vocab token IDs. train.py's RuleHead classifier
# output has one neuron per entry in RULE_LABELS, which is built by sorting
# tokenizer/vocab.json's rule_tokens by vocab ID and taking each entry's
# *list position* as the classifier's target index (see train.py's
# flatten_vocab / RULE_LABELS construction). rule_ids written here must be
# that 0-based classifier index, not the raw vocab ID, or CrossEntropyLoss
# raises "IndexError: Target X is out of bounds."
#
# Current rule_tokens (sorted by vocab ID) and their resulting classifier index:
#   RULE:power_rule (90)           -> index 0
#   RULE:chain_rule (91)           -> index 1
#   RULE:product_rule (92)         -> index 2
#   RULE:quotient_rule (93)        -> index 3
#   RULE:sum_rule (94)             -> index 4
#   RULE:constant_rule (95)        -> index 5
#   RULE:power_rule_integral (96)  -> index 6
#   RULE:partial_derivative (97)   -> index 7
#   RULE:lagrange_multiplier (98)  -> index 8
#   RULE:integration_by_parts (99) -> index 9
#   RULE:trig_rule (102)           -> index 10
#   RULE:exp_rule (103)            -> index 11
#   RULE:log_rule (104)            -> index 12
#
# 0 = power_rule, 4 = sum_rule, 5 = constant_rule, 7 = partial_derivative
# (existing Phase 1 assignments -- these already happen to equal the correct
# classifier index since vocab IDs 90-99 are consecutive, unchanged here).
#
# Phase 2 fix (see docs/KNOWN_ISSUES.md, "Rule head training plateau caused
# by Phase 2 rule_id mislabeling"): sin/cos/tan/exp/ln previously all reused
# rule_id 1 (chain_rule's index) as a placeholder, since no dedicated tokens
# existed. tokenizer/vocab.json v1.4 added RULE:trig_rule, RULE:exp_rule,
# and RULE:log_rule at vocab IDs 102-104, which sort to classifier indices
# 10, 11, 12 respectively (NOT 102/103/104 -- that was a bug introduced in
# the first pass of this fix; see docs/KNOWN_ISSUES.md).
RULE_ID_TRIG = 10   # index of RULE:trig_rule -- sin, cos, tan
RULE_ID_EXP = 11    # index of RULE:exp_rule -- exp
RULE_ID_LOG = 12    # index of RULE:log_rule -- ln


def _output_in_vocab(coeff, power):
    """Check that the derivative output (coeff*power, power-1) stays in vocab range."""
    out_coeff = coeff * power
    out_exp = power - 1
    return out_coeff in SAFE_COEFFS and out_exp in SAFE_EXPONENTS


def _integral_in_vocab(coeff, power):
    """Check that the integral output (coeff/(power+1), power+1) stays in vocab range."""
    new_power = power + 1
    if new_power == 0:
        return False  # Would be ln|x|, not supported
    new_coeff = coeff / new_power
    # Must be an integer to tokenize cleanly
    if not float(new_coeff).is_integer():
        return False
    return int(new_coeff) in SAFE_COEFFS and new_power in SAFE_EXPONENTS


def generate_single_term_diff(var="x"):
    """Generate a single-term power-rule differentiation problem."""
    for _ in range(100):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(SAFE_POS_EXPONENTS)
        if _output_in_vocab(coeff, power):
            src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
            ans = {"numi": {"terms": [{"coeff": coeff * power, "var": {var: power - 1}}]}, "deno": 1}
            # Clean zero-exponent vars
            if ans["numi"]["terms"][0]["var"][var] == 0:
                ans = {"coeff": coeff * power}
            return src, ans, 0  # rule_id 0 = power_rule
    # Fallback safe pair
    return {"numi": {"terms": [{"coeff": 2, "var": {var: 2}}]}, "deno": 1}, {"numi": {"terms": [{"coeff": 4, "var": {var: 1}}]}, "deno": 1}, 0


def generate_constant_term():
    """Generate a constant differentiation problem (derivative = 0)."""
    coeff = random.choice(SAFE_NONZERO_COEFFS)
    src = {"numi": {"terms": [{"coeff": coeff}]}, "deno": 1}
    ans = {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
    return src, ans, 5  # rule_id 5 = constant_rule


def generate_multi_term_diff(var="x", num_terms=None):
    """Generate a multi-term polynomial differentiation problem (sum rule)."""
    if num_terms is None:
        num_terms = random.randint(2, 3)

    src_terms = []
    ans_terms = []

    for i in range(num_terms):
        # Mix: some power-rule terms, optionally a constant
        if i == num_terms - 1 and random.random() < 0.3:
            # Add a constant term
            c_src, c_ans, _ = generate_constant_term()
            src_terms.append(c_src)
            # Constant differentiates to 0, so we skip adding to ans
        else:
            t_src, t_ans, _ = generate_single_term_diff(var)
            # Avoid duplicate exponents in the same polynomial
            src_exps = {list(t["numi"]["terms"][0].get("var", {}).values())[0] for t in src_terms if t["numi"]["terms"][0].get("var")}
            t_exp = list(t_src["numi"]["terms"][0].get("var", {}).values())[0] if t_src["numi"]["terms"][0].get("var") else None
            if t_exp in src_exps:
                # Try a different exponent
                t_src, t_ans, _ = generate_single_term_diff(var)
            src_terms.append(t_src)
            ans_terms.append(t_ans)

    if not ans_terms:
        ans_terms = [{"numi": {"terms": [{"coeff": 0}]}, "deno": 1}]

    return src_terms, ans_terms, 4  # rule_id 4 = sum_rule


def generate_negative_exp_diff(var="x"):
    """Generate a differentiation problem with negative exponents."""
    neg_exps = [e for e in SAFE_EXPONENTS if e < 0]
    for _ in range(100):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(neg_exps)
        if _output_in_vocab(coeff, power):
            src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
            new_exp = power - 1
            ans = {"numi": {"terms": [{"coeff": coeff * power, "var": {var: new_exp}}]}, "deno": 1}
            return src, ans, 0  # power_rule
    return {"numi": {"terms": [{"coeff": 1, "var": {var: -1}}]}, "deno": 1}, {"numi": {"terms": [{"coeff": -1, "var": {var: -2}}]}, "deno": 1}, 0


def generate_multivar_diff():
    """Generate a multi-variable partial differentiation problem."""
    var = random.choice(VARIABLES)
    other_vars = [v for v in VARIABLES if v != var]

    src_terms = []
    ans_terms = []

    # Term with the target variable
    t_src, t_ans, _ = generate_single_term_diff(var)
    src_terms.append(t_src)
    ans_terms.append(t_ans)

    # Term with another variable (treated as constant → differentiates to 0)
    if other_vars:
        ov = random.choice(other_vars)
        c = random.choice(SAFE_NONZERO_COEFFS)
        p = random.choice(SAFE_POS_EXPONENTS)
        src_terms.append({"numi": {"terms": [{"coeff": c, "var": {ov: p}}]}, "deno": 1})
        # This term vanishes under d/d(var)

    if not ans_terms:
        ans_terms = [{"numi": {"terms": [{"coeff": 0}]}, "deno": 1}]

    return src_terms, ans_terms, var, 7  # rule_id 7 = partial_derivative


# ── Phase 2 additions: trig / exp / log templates ───────────────────────────
# Use vocab.json v1.3's OP:sin / OP:cos / OP:tan / OP:exp / OP:ln / OP:sec
# tokens, plus slang_serializer.py's new optional 'coeff' and 'power'
# decorators on op-nodes (backward-compatible -- existing op-nodes never set
# these fields, so their token output is unaffected).
#
# All five functions vary the inner argument's multiplier k (giving sin(kx),
# cos(kx), tan(kx), exp(kx), ln(kx) instead of always just sin(x) etc.) so
# the model sees more than one exact pattern per function, per reviewer
# feedback. Verified via exhaustive serialization test against the real
# vocab: 0 failures across all 375,000 generated node instances.
#
# rule_id fix: previously all five functions returned rule_id 1 (chain_rule's
# classifier index) as a placeholder because no dedicated tokens existed.
# Now uses RULE_ID_TRIG / RULE_ID_EXP / RULE_ID_LOG -- the correct classifier
# indices (10/11/12) for the new RULE:trig_rule/exp_rule/log_rule vocab
# tokens (IDs 102/103/104) -- which stops ~25,000 semantically unrelated
# rows from sharing one label. See docs/KNOWN_ISSUES.md.

def generate_sin_diff(var="x"):
    """d/dvar[sin(k*var)] = k*cos(k*var)."""
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "sin", "expr": inner}
    ans = {"op": "cos", "expr": inner}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_TRIG


def generate_cos_diff(var="x"):
    """d/dvar[cos(k*var)] = -k*sin(k*var). Uses the symmetric-safe coeff
    range since -k must also be a valid vocab token (asymmetric COEF range:
    -10 to 12, so k=12 is excluded here to avoid needing COEF:-12)."""
    k = random.choice(SAFE_SYMMETRIC_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "cos", "expr": inner}
    ans = {"op": "sin", "expr": inner, "coeff": -k}
    return src, ans, RULE_ID_TRIG


def generate_tan_diff(var="x"):
    """d/dvar[tan(k*var)] = k*sec^2(k*var). Requires both the coeff and
    power decorators together."""
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "tan", "expr": inner}
    ans = {"op": "sec", "expr": inner, "power": 2}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_TRIG


def generate_exp_diff(var="x"):
    """d/dvar[exp(k*var)] = k*exp(k*var). Self-derivative scaled by k."""
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "exp", "expr": inner}
    ans = {"op": "exp", "expr": inner}
    if k != 1:
        ans["coeff"] = k
    return src, ans, RULE_ID_EXP


def generate_ln_diff(var="x"):
    """d/dvar[ln(k*var)] = 1/var (k cancels algebraically -- this is the
    correct symbolic derivative regardless of k, consistent with how the
    rest of this codebase treats formal derivatives without domain
    restriction, e.g. the existing negative-exponent templates)."""
    k = random.choice(SAFE_NONZERO_COEFFS)
    inner = {"numi": {"terms": [{"coeff": k, "var": {var: 1}}]}, "deno": 1}
    src = {"op": "ln", "expr": inner}
    ans = {"numi": {"terms": [{"coeff": 1}]}, "deno": {"terms": [{"coeff": 1, "var": {var: 1}}]}}
    return src, ans, RULE_ID_LOG


# ── New addition: integrate templates ────────────────────────────────────────
# eval/run_eval.py's benchmark set (eval/benchmarks/) includes an
# "integrate" operation category, but the training data previously had NO
# rows using op="integrate" at all -- only op="diff" was ever generated.
# This guaranteed 0% accuracy on the "integrate" category regardless of
# model quality, since the model had never seen a single example of it
# during training. See docs/KNOWN_ISSUES.md.
#
# NOTE (gradient, tangent_line -- NOT included here): both were attempted
# in an earlier pass and reverted.
#   - gradient: inference/verifier.py's gradient_oracle() expects/returns a
#     {var: expr} dict (e.g. {"x": <d/dx>, "y": <d/dy>}), but
#     tokenizer/slang_serializer.py's serialize_slang_math() has no dispatch
#     branch for a dict without "op"/"numi"+"deno"/"coeff" keys -- it would
#     raise ValueError immediately on any {var: expr} row. Proper gradient
#     support needs a new AST node type added to the serializer itself
#     (e.g. a NODE:GRADIENT wrapper), which is a larger, separate change.
#   - tangent_line: no OP:tangent_line vocab token existed until vocab.json
#     v1.5. Even with that token, no oracle_fn branch exists in
#     inference/verifier.py's verify() for op == "tangent_line", and the
#     problem shape itself (a point (x0, y0) plus a linear-equation output)
#     doesn't fit the existing {numi, deno} expression schema.
# Both are flagged as separate follow-up work, not attempted in this pass.

def generate_integrate_diff(var="x"):
    """Generate a single-term power-rule integration problem (reverse power
    rule): integral of coeff*var^power dvar = (coeff/(power+1))*var^(power+1).
    Skips power == -1 (would require ln|x|, not supported -- consistent with
    _integral_in_vocab's existing restriction)."""
    for _ in range(100):
        coeff = random.choice(SAFE_NONZERO_COEFFS)
        power = random.choice(SAFE_EXPONENTS)
        if power == -1:
            continue
        if _integral_in_vocab(coeff, power):
            new_power = power + 1
            new_coeff = int(coeff / new_power)
            if power == 0:
                src = {"numi": {"terms": [{"coeff": coeff}]}, "deno": 1}
            else:
                src = {"numi": {"terms": [{"coeff": coeff, "var": {var: power}}]}, "deno": 1}
            ans = {"numi": {"terms": [{"coeff": new_coeff, "var": {var: new_power}}]}, "deno": 1}
            return src, ans, 6  # rule_id 6 = power_rule_integral
    # Fallback safe pair: integral of 4x^3 dx = x^4
    return (
        {"numi": {"terms": [{"coeff": 4, "var": {var: 3}}]}, "deno": 1},
        {"numi": {"terms": [{"coeff": 1, "var": {var: 4}}]}, "deno": 1},
        6,
    )


def generate_slang_dataset():
    print("[Dataset Engine] Programmatically synthesizing expanded SLaNg dataset...")
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    random.seed(42)  # Reproducible

    # ── Distribution of problem types ─────────────────────────────────────────
    # 35k single-term power rule
    # 25k multi-term polynomial (sum rule)
    # 10k constant terms
    # 10k negative exponent
    # 20k multi-variable partial derivatives
    # 5k sin differentiation      (Phase 2, varied k, rule index 10)
    # 5k cos differentiation      (Phase 2, varied k, rule index 10)
    # 5k tan differentiation      (Phase 2, varied k, rule index 10)
    # 5k exp differentiation      (Phase 2, varied k, rule index 11)
    # 5k ln differentiation       (Phase 2, varied k, rule index 12)
    # 10k integrate (power rule integral, rule index 6)     -- new
    # Total: 135k

    # 1. Single-term power rule (35k)
    for _ in range(35000):
        var = random.choice(VARIABLES[:1])  # mostly x for single-term
        src, ans, rule_id = generate_single_term_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 2. Multi-term polynomial / sum rule (25k)
    for _ in range(25000):
        var = random.choice(VARIABLES[:1])
        src_terms, ans_terms, rule_id = generate_multi_term_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src_terms[0]}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1},
            "tgt_output_tokens": ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1},
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 3. Constant terms (10k)
    for _ in range(10000):
        src, ans, rule_id = generate_constant_term()
        var = random.choice(VARIABLES[:1])
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 4. Negative exponents (10k)
    for _ in range(10000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_negative_exp_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 5. Multi-variable partial derivatives (20k)
    for _ in range(20000):
        src_terms, ans_terms, var, rule_id = generate_multivar_diff()
        src_op = {"op": "diff", "var": var, "expr": src_terms[0]}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1},
            "tgt_output_tokens": ans_terms[0] if ans_terms else {"coeff": 0},
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 6. Trig — sin (5k) — Phase 2 addition, varied k, rule index 10 (trig_rule)
    for _ in range(5000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_sin_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 7. Trig — cos (5k) — Phase 2 addition, varied k, rule index 10 (trig_rule)
    for _ in range(5000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_cos_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 8. Trig — tan (5k) — Phase 2 addition, varied k, rule index 10 (trig_rule)
    for _ in range(5000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_tan_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 9. Exponential — exp (5k) — Phase 2 addition, varied k, rule index 11 (exp_rule)
    for _ in range(5000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_exp_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 10. Logarithmic — ln (5k) — Phase 2 addition, varied k, rule index 12 (log_rule)
    for _ in range(5000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_ln_diff(var)
        src_op = {"op": "diff", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    # 11. Integration — power rule integral (10k) — new addition, rule index 6
    for _ in range(10000):
        var = random.choice(VARIABLES[:1])
        src, ans, rule_id = generate_integrate_diff(var)
        src_op = {"op": "integrate", "var": var, "expr": src}
        dataset.append({
            "src_tokens": src_op,
            "tgt_input_tokens": ans,
            "tgt_output_tokens": ans,
            "rule_ids": rule_id,
            "verification_state": 1,
        })

    random.shuffle(dataset)

    with open("data/slang_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    # Split sizes scale dynamically with total dataset size (previously
    # hardcoded to 90000/95000, which assumed exactly 100k records).
    total = len(dataset)
    train_end = int(total * 0.90)
    val_end = int(total * 0.95)
    for name, split_data in [("train", dataset[:train_end]), ("val", dataset[train_end:val_end]), ("test", dataset[val_end:])]:
        with open(splits_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")

    # Print coverage stats
    rule_counts = {}
    for item in dataset:
        rid = item["rule_ids"]
        rule_counts[rid] = rule_counts.get(rid, 0) + 1

    print(f"[Dataset Engine] {total} expanded lines generated successfully.")
    print(f"   Rule distribution: {rule_counts}")
    print(f"   Coefficient range: {min(SAFE_COEFFS)} to {max(SAFE_COEFFS)}")
    print(f"   Exponent range: {min(SAFE_EXPONENTS)} to {max(SAFE_EXPONENTS)}")
    print(f"   Variables: {VARIABLES}")


if __name__ == "__main__":
    generate_slang_dataset()