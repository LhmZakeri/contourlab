import pandas as pd
from typing import Optional, Union, Sequence, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from .utils import interpolate_grid, highlight_region


# -----------------------------------------------------------------------------
def plot_contour(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    ax: Optional[plt.Axes] = None,
    levels: Union[int, Sequence[float]] = 10,
    interp: bool = True,
    highlight: bool = True,
    annotate: bool = True,
    storytelling: bool = False,
    story_labels: Optional[Dict[float, str]] = None,
    cmap: str = "Blues",
    add_colorbar: bool = False,
    percentile: float = 80.0,
    norm: Optional[mpl.colors.Normalize] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Dict[str, object]:
    """
    Plot a contour map from a DataFrame with optional interpolation.

    Args:
    df : pandas.DataFrame
        Input dataframe containing the x, y, and z columns.
    x_col, y_col, z_col : str
        Column names for x-axis, y-axis, and z-axis.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    levels : int or sequence, default=10
        Number of contour levels or explicit sequence of levels.
    interp : bool, default=True
        Interpolate data onto a finer grid for smoother contours.
    highlight : bool, default=True
        If True, highlight top values (above 'percentile').
    annotate : bool, default=True
        Add inline labels to contour lines.
    storytelling : bool, default=False
        Use custom labels from 'story_labels'.
    cmap : str, default="Blues"
        Colormap for filled contours.
    add_colorbar : bool, default=False
        Add colorbar to the plot.
    percentile : float, default=80.0
        Percentile cutoff for highlighting.
    norm : matplotlib.colors.Normalize, optional
        Custom normalization object (e.g., Normalize, LogNorm).
        If None, constructed from vmin/vmax.
    vmin, vmax : float, optional
        Data range for normalization. Ignored if `norm` is provided.

    Returns:
    dict
        Dictionary containing references to artists:
        {
            "contour": contour_lines,
            "filled": contour_filled,
            "colorbar": colorbar
        }
    """
    # --- Pivot to grid -------------------------------------------------------
    pivot_df = df.pivot_table(index=y_col, columns=x_col, values=z_col)
    X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
    Z = pivot_df.values

    # --- Interpolate if requested --------------------------------------------
    if interp:
        X, Y, Z = interpolate_grid(X, Y, Z)

    # --- Determine data min/max ----------------------------------------------
    data_min, data_max = float(np.nanmin(Z)), float(np.nanmax(Z))

    # --- Levels --------------------------------------------------------------
    if isinstance(levels, int):
        levels = np.linspace(data_min, data_max, levels)
    else:
        levels = np.asarray(levels, dtype=float)
        if levels.min() > data_min:
            levels = np.insert(levels, 0, data_min)
        if levels.max() < data_max:
            levels = np.append(levels, data_max)

    # --- Axes ---------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # --- Contour lines -------------------------------------------------------
    contour_lines = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=1.0)

    if annotate:
        if storytelling and story_labels:
            ax.clabel(
                contour_lines,
                inline=True,
                fontsize=8,
                fmt=lambda v: story_labels.get(v, f"{v:.3f}"),
            )
        else:
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

    # --- Filled contours -----------------------------------------------------
    contour_filled = None
    colorbar = None

    if highlight:
        contour_filled = highlight_region(
            ax, X, Y, Z, percent=percentile, levels=levels, cmap=cmap
        )
    else:
        if norm is None:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        contour_filled = ax.contourf(
            X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend="both"
        )

    # --- Colorbar ------------------------------------------------------------
    if add_colorbar and contour_filled is not None:
        colorbar = plt.colorbar(contour_filled, ax=ax)
        if storytelling and story_labels:
            colorbar.set_ticks(list(story_labels.keys()))
            colorbar.set_ticklabels(list(story_labels.values()))
        colorbar.ax.tick_params(labelsize=12)

    return {
        "contour": contour_lines,
        "filled": contour_filled,
        "colorbar": colorbar,
    }


# -----------------------------------------------------------------------------
def plot_multiple_contours(
    dfs: List[pd.DataFrame],
    x_col: str,
    y_col: str,
    z_col: str,
    ncols: int = 2,
    share_norm: bool = True,
    norm: Optional[mpl.colors.Normalize] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple = (10, 6),
    **kwargs,
) -> Dict[str, object]:
    """
    Plot multiple contour maps in agrid layout.

    Args:
    dfs : list of pandas.DataFrame
        List of dataframes containing x, y, z columns.
    x_cols, y_cols, z_cols : str
        column names of x-axis, y-axis, and z-axis.
    ncols : int, default=2
        Number of columns in subplot grid.
    share_norm : bool, default=True
        If True, all subplots share the same normalization(vmin/vmax or norm)
    norm : matplotlib.colors.Normalize, optional
        Custom normalization for all subplots (override vmin/vmax)
    vmin, vmax : float, optional
        Gloal normalization bounds (only used if norm=None)
    figsize: tuple, default=(10, 6)
        Size of the entire figure.
    **kwargs :
        Additional arguments passed to 'plot_contour'.

    Returns:
    dict
        Dictionary containing subplot axes and contour handles.
    """
    nplots = len(dfs)
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.ravel()

    # --- Shared normalization ------------------------------------------------
    if share_norm:
        if norm is None:
            # compute global min/max across all dfs
            all_vals = []
            for df in dfs:
                pivot_df = df.pivot_table(index=y_col, columns=x_col, values=z_col)
                all_vals.append(pivot_df.values)
            data_min = min(float(arr.min()) for arr in all_vals)
            data_max = max(float(arr.max()) for arr in all_vals)
            norm = mpl.colors.Normalize(
                vmin=data_min if vmin is None else vmin,
                vmax=data_max if vmax is None else vmax,
            )
    else:
        norm = None

    results = []
    for i, df in enumerate(dfs):
        res = plot_contour(
            df,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            ax=axes[i],
            norm=norm if share_norm else None,
            vmin=vmin if not share_norm else None,
            vmax=vmax if not share_norm else None,
            **kwargs,
        )
        results.append(res)

    filled_example = next(
        (r["filled"] for r in results if r["filled"] is not None), None
    )
    if share_norm and filled_example is not None:
        cbar = fig.colorbar(filled_example, ax=axes, orientation="vertical", shrink=0.8)
    else:
        cbar = None

    # Turn of unused axes
    for j in range(nplots, len(axes)):
        axes[j].axis("off")

    return {"fig": fig, "axes": axes, "results": results, "colorbar": cbar}
