"""Unit tests for System One label translation helpers."""

from app.core.model import ModelService


def test_system_one_choice_labels_keep_option_names() -> None:
    labels, metadata = ModelService._system_one_labels(
        {
            "type": "choice",
            "criteria": {"payments": "Checkout issues", "account": None},
        }
    )

    assert labels == {"payments": "Checkout issues", "account": "account"}
    assert metadata == {"type": "choice", "labels": ["payments", "account"]}


def test_system_one_noul_uses_true_and_false_criteria() -> None:
    labels, metadata = ModelService._system_one_labels(
        {
            "type": "noul",
            "criteria": {"true": "Immediate risk", "false": "No immediate risk"},
        }
    )

    assert labels == {"true": "Immediate risk", "false": "No immediate risk"}
    assert metadata["type"] == "noul"


def test_system_one_score_creates_ordered_legend() -> None:
    labels, metadata = ModelService._system_one_labels(
        {"type": "score", "criteria": ["Low", "Medium", "High"]}
    )

    assert labels == {"level_0": "Low", "level_1": "Medium", "level_2": "High"}
    assert metadata["legend"] == {"0": "Low", "1": "Medium", "2": "High"}


def test_fallback_distribution_is_normalized() -> None:
    distribution = ModelService._fallback_distribution(
        ["a", "b", "c"],
        "b",
        0.7,
    )

    assert distribution["b"] == 0.7
    assert abs(sum(distribution.values()) - 1.0) < 1e-9
