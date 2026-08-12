"""
Verifies the ONNX export against the original PyTorch checkpoint:
  1. Runs every problem in eval/benchmarks/*.json through BOTH the PyTorch
     path (inference/solve.py) and the ONNX path (deployment/onnx_solve.py).
  2. Checks token-exact match between the two -- conversion must not
     silently change numerical/generation behaviour.
  3. Measures the real deployment bundle size against Vercel's ~250MB
     serverless function limit (not just the raw .onnx file -- the bundle
     includes onnxruntime + numpy + application code).
  4. Writes docs/EXPORT_DECISION.md from these actual measured numbers.

Per the task's re-validation rule: this must be re-run against every new
signed-off checkpoint. A different checkpoint can convert/behave
differently even if a previous one converted cleanly -- do not assume
results from a prior run still hold.
"""

import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VERCEL_LIMIT_MB = 250


def _load_benchmarks(pattern: str = os.path.join("eval", "benchmarks", "*.json")):
    problems = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("problems", [])
        for item in items:
            problems.append(item)
    return problems


def _measure_bundle_size_mb(onnx_path: str) -> float:
    """Real bundle size: the .onnx weights file plus the installed
    onnxruntime + numpy packages that ship alongside it in production,
    per requirements-onnx.txt -- not just the raw model file."""
    total_bytes = os.path.getsize(onnx_path)

    try:
        import onnxruntime, numpy
        for mod in (onnxruntime, numpy):
            mod_dir = os.path.dirname(mod.__file__)
            for root, _, files in os.walk(mod_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if os.path.exists(fpath):
                        total_bytes += os.path.getsize(fpath)
    except ImportError:
        pass

    return total_bytes / (1024 * 1024)


def run_verification(
    checkpoint_path: str = os.path.join("checkpoints", "final", "best.pt"),
    onnx_path: str = os.path.join("deployment", "artifacts", "best.onnx"),
) -> dict:
    from inference.solve import CalculusSolverInference
    from deployment.onnx_solve import ONNXCalculusSolverInference

    problems = _load_benchmarks()
    if not problems:
        raise RuntimeError("No benchmark problems found under eval/benchmarks/*.json")

    pt_solver = CalculusSolverInference(model_path=checkpoint_path)
    onnx_solver = ONNXCalculusSolverInference(model_path=onnx_path)

    total = 0
    exact_matches = 0
    mismatches = []

    try:
        for problem in problems:
            input_env = problem.get("input", problem)
            total += 1

            pt_result = pt_solver.solve(input_env)
            onnx_result = onnx_solver.solve(input_env)

            match = pt_result["output_tokens"] == onnx_result["output_tokens"]
            if match:
                exact_matches += 1
            else:
                mismatches.append({
                    "input": input_env,
                    "pytorch_output": pt_result["output_tokens"],
                    "onnx_output": onnx_result["output_tokens"],
                })
    finally:
        pt_solver.close()
        onnx_solver.close()

    match_rate = exact_matches / total if total else 0.0
    bundle_size_mb = _measure_bundle_size_mb(onnx_path)
    fits_limit = bundle_size_mb < VERCEL_LIMIT_MB

    return {
        "checkpoint": checkpoint_path,
        "onnx_path": onnx_path,
        "total_problems": total,
        "exact_matches": exact_matches,
        "match_rate": match_rate,
        "bundle_size_mb": bundle_size_mb,
        "vercel_limit_mb": VERCEL_LIMIT_MB,
        "fits_limit": fits_limit,
        "mismatches": mismatches[:20],  # cap for report readability
        "mismatch_count": len(mismatches),
    }


def write_decision_doc(report: dict, out_path: str = os.path.join("docs", "EXPORT_DECISION.md")) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = "PASS" if report["fits_limit"] and report["match_rate"] == 1.0 else "NEEDS ATTENTION"

    lines = [
        "# Export Decision — Option A (ONNX)",
        "",
        f"**Generated:** {timestamp}",
        f"**Checkpoint verified:** `{report['checkpoint']}`",
        f"**ONNX artifact:** `{report['onnx_path']}`",
        "",
        f"## Verdict: {verdict}",
        "",
        "## Numerical correctness (PyTorch vs ONNX)",
        f"- Problems tested: {report['total_problems']}",
        f"- Exact token match: {report['exact_matches']}/{report['total_problems']} "
        f"({report['match_rate']:.1%})",
        f"- Mismatches: {report['mismatch_count']}",
        "",
        "## Deployment size vs Vercel limit",
        f"- Measured bundle size (.onnx + onnxruntime + numpy): {report['bundle_size_mb']:.1f} MB",
        f"- Vercel serverless limit: {report['vercel_limit_mb']} MB",
        f"- Fits limit: {'Yes' if report['fits_limit'] else 'No'}",
        "",
    ]

    if report["mismatch_count"] > 0:
        lines.append("## Sample mismatches (first 20)")
        lines.append("")
        for i, m in enumerate(report["mismatches"], 1):
            lines.append(f"### Mismatch {i}")
            lines.append(f"- Input: `{json.dumps(m['input'])}`")
            lines.append(f"- PyTorch output: `{m['pytorch_output']}`")
            lines.append(f"- ONNX output: `{m['onnx_output']}`")
            lines.append("")

    lines.append(
        "**Re-validation rule:** this file must be regenerated against every new "
        "signed-off checkpoint. Do not treat this verdict as valid for a checkpoint "
        "other than the one named above."
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[verify_export] Wrote {out_path} -- verdict: {verdict}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join("checkpoints", "final", "best.pt")
    onnx = sys.argv[2] if len(sys.argv) > 2 else os.path.join("deployment", "artifacts", "best.onnx")

    report = run_verification(ckpt, onnx)
    write_decision_doc(report)

    print(json.dumps(
        {k: v for k, v in report.items() if k != "mismatches"},
        indent=2,
    ))