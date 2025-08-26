import pandas as pd
from typing import List


# -----------------------------------------------------------------------------
def filter_high_values(
    df: pd.DataFrame,
    prob_col: str,
    threshold: float = 0.8,
    group_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Return rows with values in 'prob_col' above a threshold

    Args:
    df : pd.DataFrame
        Input DataFrame
    prob_col : str
        Column containing the probability or score to filter on.
    threshold : float, optional
        Minimum value to keep. Default is 0.8.
    group_cols : list of str, optional
        Subset of columns to return alongside 'prob_col'.
        If None, returns all columns.

    Returns:
    pandas.DataFrame:
        Filtered and sorted dataframe.

    """
    if group_cols is None:
        group_cols = [c for c in df.columns if c != prob_col]

    out = (
        df.loc[df[prob_col] > threshold, group_cols + [prob_col]]
        .sort_values(
            [prob_col] + group_cols, ascending=[False] + [True] * len(group_cols)
        )
        .reset_index(drop=True)
    )
    return out 