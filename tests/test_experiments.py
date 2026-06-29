import pytest
import pandas as pd

from src.experiments import summarize_paired_difference


def test_summarize_paired_difference_calculates_user_level_difference():
    data = pd.DataFrame(
        {
            "control": [0.0, 0.5, 1.0],
            "treatment": [0.5, 0.5, 0.0],
        }
    )

    summary = summarize_paired_difference(
        data=data,
        control_column="control",
        treatment_column="treatment",
    )

    assert summary["n_users"] == 3
    assert summary["control_mean"] == 0.5
    assert summary["treatment_mean"] == pytest.approx(1 / 3)
    assert summary["mean_difference"] == pytest.approx(-1 / 6)
    assert summary["treatment_win_rate"] == pytest.approx(1 / 3)
    assert summary["control_win_rate"] == pytest.approx(1 / 3)
    assert summary["tie_rate"] == pytest.approx(1 / 3)


def test_summarize_paired_difference_drops_missing_pairs():
    data = pd.DataFrame(
        {
            "control": [0.5, None],
            "treatment": [1.0, 0.0],
        }
    )

    summary = summarize_paired_difference(
        data=data,
        control_column="control",
        treatment_column="treatment",
    )

    assert summary["n_users"] == 1
    assert summary["mean_difference"] == 0.5


def test_summarize_paired_difference_requires_paired_data():
    data = pd.DataFrame(
        {
            "control": [None],
            "treatment": [1.0],
        }
    )

    with pytest.raises(ValueError):
        summarize_paired_difference(
            data=data,
            control_column="control",
            treatment_column="treatment",
        )
