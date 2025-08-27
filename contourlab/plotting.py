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
    contour_filled, colorbar = None, None

    if highlight:
        contour_filled = highlight_region(
            ax, X, Y, Z, percent=percentile, levels=levels, cmap=cmap
        )

    if norm is None:
        if vmin is None or vmax is None:
            vmin, vmax = data_min, data_max
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
    levels: Optional[int] = 10,
    figsize: tuple = (10, 6),
    **kwargs,
) -> Dict[str, object]:
    """
    Plot multiple contour maps in a grid layout.

    Args:
        dfs : list of pandas.DataFrame
            List of dataframes containing x, y, z columns.
        x_col, y_col, z_col : str
            Column names of x-axis, y-axis, and z-axis.
        ncols : int, default=2
            Number of columns in subplot grid.
        share_norm : bool, default=True
            If True, all subplots share the same normalization.
        norm : matplotlib.colors.Normalize, optional
            Custom normalization (overrides vmin/vmax if provided).
        vmin, vmax : float, optional
            Global normalization bounds (only used if norm=None).
        levels : int or sequence, default=10
            Number of contour levels or explicit list of levels.
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
        all_data = np.concatenate([df[z_col].to_numpy().ravel() for df in dfs])
        global_vmin, global_vmax = np.nanmin(all_data), np.nanmax(all_data)
        global_norm = mpl.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
    else:
        global_norm = None

    results = []
    for i, df in enumerate(dfs):
        if share_norm:
            # shared normalization → use *only* the same norm
            res = plot_contour(
                df,
                x_col=x_col,
                y_col=y_col,
                z_col=z_col,
                ax=axes[i],
                norm=global_norm,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
        else:
            # independent normalization
            res = plot_contour(
                df,
                x_col=x_col,
                y_col=y_col,
                z_col=z_col,
                ax=axes[i],
                levels=levels,
                **kwargs,
            )
        results.append(res)

    # --- Colorbar (only for shared normalization) ----------------------------
    filled_example = next(
        (r["filled"] for r in results if r["filled"] is not None), None
    )
    if share_norm and filled_example is not None:
        cbar = fig.colorbar(filled_example, ax=axes, orientation="vertical", shrink=0.8)
    else:
        cbar = None

    # Turn off unused axes
    for j in range(nplots, len(axes)):
        axes[j].axis("off")

    return {"fig": fig, "axes": axes, "results": results, "colorbar": cbar}
