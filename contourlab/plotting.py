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
    show_stats: bool = True,
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
    axes = np.atleast_1d(axes).ravel()

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


# -------------------------------------------------------------------------
def stack_contours_in_z(
    contours_list: List[Dict[str, object]],
    z_gap: Optional[float] = None,
    z_offsets: Optional[List[float]] = None,
    figsize: tuple = (10, 8),
    elev: int = 22,
    azim: int = -60,
    mode: str = "line",  # "line" or "filled"
    alpha: float = 0.6,  # transparency for filled
    show_lines: bool = True,  # show contour lines in filled mode
    cmap: Optional[str] = None,  # colormap for filled mode
) -> Dict[str, object]:
    """
    Stack 2D contour plots in 3D along Z.

    Args:
        contours_list : list of QuadContourSet
        z_gap, z_offsets : Z placement control
        mode : {"line", "filled"}
        alpha : float transparency (only filled)
        show_lines : bool, draw contour outlines in filled mode
        cmap : str or None, override face colors with a colormap
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # --- global XY extents ------------------------------------------------
    gx_min, gx_max = float("inf"), -float("inf")
    gy_min, gy_max = float("inf"), -float("inf")
    for cs in contours_list:
        if cs is None or not cs.collections:
            continue
        for coll in cs.collections:
            for path in coll.get_paths():
                v = path.vertices
                if v.size == 0 or v.shape[0] < 2:
                    continue
                gx_min = min(gx_min, v[:, 0].min())
                gx_max = max(gx_max, v[:, 0].max())
                gy_min = min(gy_min, v[:, 1].min())
                gy_max = max(gy_max, v[:, 1].max())

    if not np.isfinite([gx_min, gx_max, gy_min, gy_max]).all():
        return {"fig": fig, "ax": ax, "z_offsets": []}

    gx_rng = gx_max - gx_min
    gy_rng = gy_max - gy_min
    n = len(contours_list)

    # --- Resolve z offsets ------------------------------------------------
    if z_offsets is not None:
        if len(z_offsets) != n:
            raise ValueError("len(z_offsets) must match the number of contours")
        offsets = list(z_offsets)
    else:
        if z_gap is None:
            z_gap = 0.15 * max(gx_rng, gy_rng)
        offsets = [i * z_gap for i in range(n)]

    ax.set_proj_type("ortho")
    ax.set_box_aspect((gx_rng, gy_rng, max(offsets[-1] if offsets else 1.0, 1.0)))
    ax.view_init(elev=elev, azim=azim)

    # --- Draw each contour set --------------------------------------------
    for z0, cs in zip(offsets, contours_list):
        if cs is None or not cs.collections:
            continue

        if mode == "line":
            # Pure line contours
            for coll in cs.collections:
                ec = coll.get_edgecolor()
                color = ec[0] if len(ec) else "k"
                for path in coll.get_paths():
                    v = path.vertices
                    if v.shape[0] < 2:
                        continue
                    x, y = v[:, 0], v[:, 1]
                    z = np.full_like(x, z0, dtype=float)
                    ax.plot(x, y, z, linewidth=1.0, color=color)

        elif mode == "filled":
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            import matplotlib.cm as cm

            # prepare colormap if requested
            if cmap is not None:
                cmap_obj = cm.get_cmap(cmap)
                levels = cs.levels
                norm = plt.Normalize(vmin=min(levels), vmax=max(levels))
            else:
                cmap_obj = None
                norm = None

            for level_idx, coll in enumerate(cs.collections):
                # smooth Z increment per contour level
                dz = 0.8 * z_gap / max(len(cs.collections), 1)
                z_level = z0 + level_idx * dz

                if cmap_obj is not None:
                    level_val = cs.levels[level_idx]
                    facecolor = cmap_obj(norm(level_val))
                else:
                    fc = coll.get_facecolor()
                    facecolor = fc[0] if len(fc) else (0.5, 0.5, 0.5, 1.0)

                for path in coll.get_paths():
                    v = path.vertices
                    if v.shape[0] < 3:
                        continue
                    verts3d = [(vx, vy, z_level) for vx, vy in v]
                    poly = Poly3DCollection([verts3d])
                    poly.set_facecolor(facecolor)
                    poly.set_alpha(alpha)
                    poly.set_edgecolor("none")
                    ax.add_collection3d(poly)

                    if show_lines:
                        x, y = v[:, 0], v[:, 1]
                        z = np.full_like(x, z_level, dtype=float)  # match band z
                        ax.plot(x, y, z, color="k", linewidth=0.5)

        # slice box
        ax.plot(
            [gx_min, gx_max, gx_max, gx_min, gx_min],
            [gy_min, gy_min, gy_max, gy_max, gy_min],
            [z0] * 5,
            color="k",
            linewidth=1,
        )

    # --- Limits & Labels --------------------------------------------------
    margin_x = 0.02 * gx_rng
    margin_y = 0.02 * gy_rng
    ax.set_xlim(gx_min - margin_x, gx_max + margin_x)
    ax.set_ylim(gy_min - margin_y, gy_max + margin_y)

    ax.set_xlabel("period")
    ax.set_ylabel("wavelength")
    ax.set_zlabel("slice (Z)")
    ax.set_title(f"Stacked 2D contours in 3D ({mode})")
    plt.tight_layout()

    return {"fig": fig, "ax": ax, "z_offsets": offsets}
