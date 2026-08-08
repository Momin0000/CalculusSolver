"""
Unit tests for Quotient Rule, Trig/Exp/Log functions, Logarithmic Integrals, and Hessian Matrix in FallbackSolver.
"""

import pytest
from inference.fallback_solver import FallbackSolver, _diff_fraction, _fraction_to_latex, _integrate_fraction


def test_quotient_rule_differentiation():
    """Test differentiation of (3x^2 + 1) / (x + 2) using quotient rule."""
    solver = FallbackSolver()
    payload = {
        "op": "diff",
        "var": "x",
        "expr": {
            "numi": {
                "terms": [
                    {"coeff": 3, "var": {"x": 2}},
                    {"coeff": 1}
                ]
            },
            "deno": {
                "terms": [
                    {"coeff": 1, "var": {"x": 1}},
                    {"coeff": 2}
                ]
            }
        }
    }
    res = solver.solve(payload)
    assert res["status"] == "solved"
    assert res["rule"] == "quotient_rule"
    assert "deno" in res["expr"]
    assert res["latex"] != ""


def test_trig_exp_log_differentiation():
    """Test differentiation of trig, exponential, and log terms."""
    solver = FallbackSolver()
    # d/dx (sin(x) + e^x)
    payload = {
        "op": "diff",
        "var": "x",
        "expr": {
            "numi": {
                "terms": [
                    {"coeff": 1, "func": "sin", "arg": "x"},
                    {"coeff": 1, "func": "exp", "arg": "x"}
                ]
            },
            "deno": 1
        }
    }
    res = solver.solve(payload)
    assert res["status"] == "solved"
    # d/dx sin(x) -> cos(x), d/dx e^x -> e^x
    latex = res["latex"]
    assert "cos(x)" in latex or "e^{x}" in latex


def test_logarithmic_integral():
    """Test integral of 1/x -> ln(x)."""
    solver = FallbackSolver()
    payload = {
        "op": "integrate",
        "var": "x",
        "expr": {
            "numi": {
                "terms": [
                    {"coeff": 1, "var": {"x": -1}}
                ]
            },
            "deno": 1
        }
    }
    res = solver.solve(payload)
    assert res["status"] == "solved"
    assert "ln(x)" in res["latex"]


def test_hessian_matrix():
    """Test Hessian matrix of f(x, y) = x^2 + 3xy + y^2."""
    solver = FallbackSolver()
    payload = {
        "op": "hessian",
        "var": "x",
        "expr": {
            "numi": {
                "terms": [
                    {"coeff": 1, "var": {"x": 2}},
                    {"coeff": 3, "var": {"x": 1, "y": 1}},
                    {"coeff": 1, "var": {"y": 2}}
                ]
            },
            "deno": 1
        }
    }
    res = solver.solve(payload)
    assert res["status"] == "solved"
    assert "hessian" in res["expr"]
    assert res["rule"] == "hessian"
    assert "pmatrix" in res["latex"]
