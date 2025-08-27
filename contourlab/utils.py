import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from typing import List


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
