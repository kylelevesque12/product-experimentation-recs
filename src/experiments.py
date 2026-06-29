import math

import pandas as pd


def summarize_paired_difference(
    data: pd.DataFrame,
    control_column: str,
    treatment_column: str,
) -> dict:
    """Summarize paired treatment-control differences for one metric."""
    paired_data = data[[control_column, treatment_column]].dropna().copy()

    if paired_data.empty:
        raise ValueError("No paired non-missing observations are available.")

    differences = paired_data[treatment_column] - paired_data[control_column]

    mean_difference = differences.mean()
    standard_error = differences.std(ddof=1) / math.sqrt(len(differences))
    confidence_margin = 1.96 * standard_error

    return {
        "n_users": len(differences),
        "control_mean": paired_data[control_column].mean(),
        "treatment_mean": paired_data[treatment_column].mean(),
        "mean_difference": mean_difference,
        "ci_lower": mean_difference - confidence_margin,
        "ci_upper": mean_difference + confidence_margin,
        "treatment_win_rate": (differences > 0).mean(),
        "control_win_rate": (differences < 0).mean(),
        "tie_rate": (differences == 0).mean(),
    }
