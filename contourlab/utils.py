import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from typing import List, Union, Optional, Dict


# -----------------------------------------------------------------------------
def interpolate_grid(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    resolution: int = 100,
    method: str = "cubic",
):
    """Interpolate data onto a finer grid for smooth contours."""
    # --- mask Invalid points --------------------------------------------------
    mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    if mask.sum() < 3:
        return X, Y, np.ma.masked_invalid(Z)

    x_flat, y_flat, z_flat = X.flatten(), Y.flatten(), Z.flatten()
    mask_flat = mask.flatten()
    # --- Build finer grid -----------------------------------------------------
    X_fine = np.linspace(np.nanmin(X), np.nanmax(X), resolution)
    Y_fine = np.linspace(np.nanmin(Y), np.nanmax(Y), resolution)
    X_fine, Y_fine = np.meshgrid(X_fine, Y_fine)
    # --- Interpolate only from valid samples ----------------------------------
    Z_fine = griddata(
        (X.flatten()[mask_flat], Y.flatten()[mask_flat]),
        Z.flatten()[mask_flat],
        (X_fine, Y_fine),
        method=method,
    )
    # Fallback : if result is all NaN (common for cubic on sparse grids),retry
    # with 'nearest'
    if Z_fine is None or np.all(~np.isfinite(Z_fine)):
        Z_fine = griddata(
            (X.flatten()[mask_flat], Y.flatten()[mask_flat]),
            Z.flatten()[mask_flat],
            (X_fine, Y_fine),
            method="nearest",
        )
    return X_fine, Y_fine, Z_fine


# -----------------------------------------------------------------------------
def highlight_region(
    ax, X, Y, Z, percent: float, levels: int = 10, cmap: str = "Blues"
):
    """Highlight top values in Z"""
    Zmin, Zmax = np.nanmin(Z), np.nanmax(Z)
    threshold = np.nanpercentile(Z, percent)

    if np.isscalar(levels):
        n = max(int(levels), 2)
        fill_levels = np.linspace(max(threshold, Zmin), Zmax, n)
    else:
        levels = np.asarray(levels)
        fill_levels = levels[levels >= threshold]
        if fill_levels.size < 2:
            # Ensure have at least two levels for contourf
            fill_levels = np.array([max(threshold, Zmin), Zmax])
    # Mask values below threshold
    Z_highlight = np.where(Z >= threshold, Z, Zmin - 1.0)
    return ax.contourf(X, Y, Z_highlight, levels=fill_levels, cmap=cmap)


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


# -----------------------------------------------------------------------------
def label_dataframe(
    dataframe: pd.DataFrame, col: str, thr: Union[int, float, str] = 1000
) -> pd.DataFrame:
    """
    Binary labeling of a datframe based on a threshold applied to a column assigns
    1 if dataframe[col] == thr, else 0.
    """
    dataframe["label"] = np.where(dataframe[col] == thr, 1, 0)
    return dataframe


# -----------------------------------------------------------------------------
def make_grouped_mean(
    datadir: Union[str, List[str]],
    filters: Optional[Dict[str, Union[int, float, str]]] = None,
    target_col: str = "label",
    drop_cols: Optional[List[str]] = None,
    first_group_keys: Optional[List[str]] = None,
    second_group_keys: Optional[List[str]] = None,
    do_label: bool = False,
    label_col: Optional[str] = None,
    label_thr: Union[int, float, str] = 1000,
) -> pd.DataFrame:
    """
    2-stage grouping with optional filtering

    Args:
    datadir: str or List
        Path(s) to datafile(s)
    filters: dict, optional
        Dictionary of column: value filters to apply before grouping.
        Example:{"period": 110, "temperature": 37}
    target_col : str, default = "label"
        Column to average
    drop_cols: list of str, optional
        Drop unwanted columns.
    first_group_keys : list of str, optional
        Grouping keys for the first aggregation step.
    second_group_keys: list of str, optional
        Keys for the second aggregation step
    do_label : bool
        create binary labels  (Y/N: 1/0)
    label_col: str, optional
    label_thr: int, float, str
        reference for binary labeling


    returns:
    pd.DataFrame
        Aggregated DataFrame with column "mean_<target_col>"
    """
    # --- Load Data ---------------------------------------------------------------
    if isinstance(datadir, str):
        df = [pd.read_csv(datadir, sep=r"\s+")]
    else:
        df = [pd.read_csv(f, sep=r"\s+") for f in datadir]
    df = pd.concat(df, ignore_index=True)
    # --- Drop unwanted cols ------------------------------------------------------
    if drop_cols:
        df.drop(columns=drop_cols, errors="ignore", inplace=True)
    # --- Apply labeling ----------------------------------------------------------
    if do_label:
        if label_col is None:
            raise ValueError("You must specify 'label_col' when do_label=True")
        df = label_dataframe(df, col=label_col, thr=label_thr)
    # --- Apply filters -----------------------------------------------------------
    if filters:
        for col, val in filters.items():
            df = df[df[col] == val]
    # --- Fine grouping -----------------------------------------------------------
    if first_group_keys:
        df = df.groupby(first_group_keys, as_index=False).agg({target_col: "mean"})
    # --- Coarser grouping --------------------------------------------------------
    if second_group_keys:
        df = df.groupby(second_group_keys, as_index=False).agg({target_col: "mean"})
    return df 
