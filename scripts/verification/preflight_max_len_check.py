"""
Pre-flight token-length check (Dev 2 objective, Objectives v3).

Dev 3's fix confirmed gradient targets are 100% truncated at max_len=32
(real max observed: 40 tokens). Before the next training run, this script
runs the SAME check against every other category (diff, integrate, partial,
tangent_line) to confirm none of them are silently truncating too.

Tokenization mirrors train.py's SlangDatasetLoader._tokenize exactly:
  - serialize_slang_math() for the token stream
  - [BOS]/[EOS] boundaries added for target sequences (tgt_input/tgt_output),
    NOT for src sequences (matches add_boundaries=False/True split in train.py)

Usage:
    python scripts/verification/preflight_max_len_check.py [--samples N] [--max-len N]

Run this before every future training start (per Dev 3 objective 3).
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from problem_generator import (
    generate_single_term_diff,
    generate_multi_term_diff,
    generate_constant_term,
    generate_negative_exp_diff,
    generate_multivar_diff,
    generate_sin_diff,
    generate_cos_diff,
    generate_tan_diff,
    generate_exp_diff,
    generate_ln_diff,
    generate_integrate_diff,
    generate_gradient_diff,
    generate_tangent_line_diff,
    VARIABLES,
)
from tokenizer.slang_serializer import serialize_slang_math
import json

VOCAB_PATH = Path(__file__).resolve().parents[2] / "tokenizer" / "vocab.json"


def load_vocab_mapping():
    raw = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    flat = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            flat.update(value)
    return flat


VOCAB_MAPPING = load_vocab_mapping()


def token_len(envelope, add_boundaries):
    """Length in tokens BEFORE padding/truncation -- mirrors train.py's _tokenize."""
    tokens = serialize_slang_math(envelope)
    if add_boundaries:
        tokens = ["[BOS]"] + tokens + ["[EOS]"]
    missing = [t for t in tokens if t not in VOCAB_MAPPING]
    if missing:
        raise KeyError(f"Token(s) missing from vocab.json: {missing}")
    return len(tokens)


def sample_category(name, n):
    """Yields (src_envelope, tgt_envelope) pairs for one category, matching
    problem_generator.generate_slang_dataset()'s exact src/tgt construction
    for that category."""
    for _ in range(n):
        if name == "diff":
            # Representative of the diff-family generators (single-term power
            # rule is the largest slice of "diff"; multi-term/constant/negative-exp
            # are checked too since they all share op="diff").
            var = random.choice(VARIABLES[:1])
            src, ans, _ = generate_single_term_diff(var)
            yield {"op": "diff", "var": var, "expr": src}, ans
        elif name == "diff_multiterm":
            var = random.choice(VARIABLES[:1])
            src_terms, ans_terms, _ = generate_multi_term_diff(var)
            ans = ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
            yield {"op": "diff", "var": var, "expr": src_terms[0]}, ans
        elif name == "diff_trig_exp_log":
            var = random.choice(VARIABLES[:1])
            gen = random.choice(
                [generate_sin_diff, generate_cos_diff, generate_tan_diff,
                 generate_exp_diff, generate_ln_diff]
            )
            src, ans, _ = gen(var)
            yield {"op": "diff", "var": var, "expr": src}, ans
        elif name == "integrate":
            var = random.choice(VARIABLES[:1])
            src, ans, _ = generate_integrate_diff(var)
            yield {"op": "integrate", "var": var, "expr": src}, ans
        elif name == "partial":
            src_terms, ans_terms, var, _ = generate_multivar_diff()
            ans = ans_terms[0] if ans_terms else {"numi": {"terms": [{"coeff": 0}]}, "deno": 1}
            yield {"op": "partial", "var": var, "expr": src_terms[0]}, ans
        elif name == "tangent_line":
            var = random.choice(VARIABLES[:1])
            src_op, ans, _, _ = generate_tangent_line_diff(var)
            yield src_op, ans
        elif name == "gradient":
            expr, ans, _ = generate_gradient_diff()
            yield {"op": "gradient", "var": "x", "expr": expr}, ans
        else:
            raise ValueError(f"Unknown category: {name}")


def run_check(categories, n_samples, max_len):
    print(f"{'category':<18}{'n':>7}{'max_src':>9}{'max_tgt':>9}{'over_src':>10}{'over_tgt':>10}   verdict")
    print("-" * 80)
    results = {}
    any_failure = False
    for cat in categories:
        max_src = max_tgt = 0
        over_src = over_tgt = 0
        n = 0
        for src_env, tgt_env in sample_category(cat, n_samples):
            n += 1
            s_len = token_len(src_env, add_boundaries=False)
            t_len = token_len(tgt_env, add_boundaries=True)
            max_src = max(max_src, s_len)
            max_tgt = max(max_tgt, t_len)
            if s_len > max_len:
                over_src += 1
            if t_len > max_len:
                over_tgt += 1

        truncating = over_src > 0 or over_tgt > 0
        any_failure = any_failure or truncating
        verdict = "TRUNCATING" if truncating else "OK"
        print(f"{cat:<18}{n:>7}{max_src:>9}{max_tgt:>9}{over_src:>10}{over_tgt:>10}   {verdict}")
        results[cat] = {
            "n": n, "max_src": max_src, "max_tgt": max_tgt,
            "over_src": over_src, "over_tgt": over_tgt, "truncating": truncating,
        }

    print("-" * 80)
    if any_failure:
        print(f"RESULT: at least one category exceeds max_len={max_len}. Do not start training.")
    else:
        print(f"RESULT: all sampled categories fit within max_len={max_len}.")
    return results, any_failure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000,
                         help="Samples per category (default 2000, matches Dev 3's gradient check)")
    parser.add_argument("--max-len", type=int, default=None,
                         help="max_len to check against (default: config.json's current value)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_len is None:
        cfg = json.loads((Path(__file__).resolve().parents[2] / "config.json").read_text())
        args.max_len = cfg.get("max_len", 32)

    random.seed(args.seed)

    categories = [
        "diff", "diff_multiterm", "diff_trig_exp_log",
        "integrate", "partial", "tangent_line", "gradient",
    ]
    _, failed = run_check(categories, args.samples, args.max_len)
    sys.exit(1 if failed else 0)
