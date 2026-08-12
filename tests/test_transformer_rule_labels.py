"""
Unit tests for model/transformer.py's rule_labels handling.

Regression test for the bug found in the Dev 1 audit: CalculusSolverModel
used to hardcode rule_labels = [f"RULE_{i}" ...] internally, ignoring the
real rule names inference/solve.py and train.py already computed from
vocab.json's rule_tokens. That meant every .pt-loaded model reported
placeholder rule names (RULE_0, RULE_7, ...) instead of real ones
(add_rule, chain_rule, ...), no matter what vocab it was paired with.
"""

import pytest
from model.transformer import CalculusSolverModel


def test_real_rule_labels_are_used_when_provided():
    real_labels = ["power_rule", "chain_rule", "product_rule"]
    model = CalculusSolverModel(
        vocab_size=50, num_rules=3, hidden_dim=16, rule_labels=real_labels
    )
    assert model.rule_head.labels() == real_labels


def test_falls_back_to_placeholder_labels_when_none_provided():
    model = CalculusSolverModel(vocab_size=50, num_rules=3, hidden_dim=16)
    assert model.rule_head.labels() == ["RULE_0", "RULE_1", "RULE_2"]


def test_mismatched_label_count_raises_instead_of_silently_mislabeling():
    with pytest.raises(ValueError):
        CalculusSolverModel(
            vocab_size=50, num_rules=3, hidden_dim=16, rule_labels=["only_one"]
        )
