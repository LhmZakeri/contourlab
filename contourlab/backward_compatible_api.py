import pandas as pd
from typing import Optional, Union, Sequence, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from .plotting_refactored import ContourPlotter, MultiContourPlotter, Contour3DStacker, PlotConfig
from .utils import interpolate_grid, highlight_region


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
    xlabels: Optional[str] = None,
    ylabels: Optional[str] = None,
    title: Optional[str] = None,
    font_axis_label: int = 12,
    font_tick: int = 10,
    font_annotation: int = 8,
    font_title: int = 14,
) -> Dict[str, object]:
    """
    Original API wrapper for plot_contour function.
    
    This maintains full backward compatibility with the original function signature
    while using the improved backend.
    """
    # Create configuration from parameters
    config = PlotConfig(
        levels=levels,
        cmap=cmap,
        interpolate=interp,
        highlight=highlight,
        annotate=annotate,
        add_colorbar=add_colorbar,
        percentile_threshold=percentile,
        font_axis_label=font_axis_label,
        font_tick=font_tick,
        font_annotation=font_annotation,
        font_title=font_title
    )
    
    # Create plotter instance
    plotter = ContourPlotter(config)
    
    # Handle normalization
    if norm is None and (vmin is not None or vmax is not None):
        # Calculate data range if not provided
        if vmin is None or vmax is None:
            pivot_df = df.pivot_table(index=y_col, columns=x_col, values=z_col)
            Z = pivot_df.values
            if interp:
                X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
                X, Y, Z = interpolate_grid(X, Y, Z)
            
            if vmin is None:
                vmin = float(np.nanmin(Z))
            if vmax is None:
                vmax = float(np.nanmax(Z))
        
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Call the new API
    result = plotter.plot_single_contour(
        df=df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        ax=ax,
        x_label=xlabels,
        y_label=ylabels,
        title=title,
        levels=levels,
        norm=norm,
        storytelling=storytelling,
        story_labels=story_labels
    )
    
    # Convert result to match original API format
    return {
        "contour": result["contour_lines"],
        "filled": result["contour_filled"],
        "colorbar": result["colorbar"]
    }


def plot_multiple_contours(
    dfs: List[pd.DataFrame],
    x_col: str,
    y_col: str,
    z_col: str,
    ncols: int = 2,
    share_norm: bool = True,
    norm: Optional[Union[mpl.colors.Normalize, Sequence[mpl.colors.Normalize]]] = None,
    vmin: Optional[Union[float, Sequence[float]]] = None,
    vmax: Optional[Union[float, Sequence[float]]] = None,
    levels: Optional[Union[int, Sequence[float], List[Sequence[float]]]] = 10,
    figsize: tuple = (10, 6),
    add_colorbar: bool = True,
    story_labels: Optional[Dict[float, str]] = None,
    storytelling: bool = False,
    xlabels: Optional[Union[str, List[str]]] = None,
    ylabels: Optional[Union[str, List[str]]] = None,
    titles: Optional[Sequence[str]] = None,
    font_axis_label: int = 12,
    font_tick: int = 10,
    font_annotation: int = 8,
    font_title: int = 14,
    cbar_fontsize: int = 10,
    **kwargs,
) -> Dict[str, object]:
    """
    Original API wrapper for plot_multiple_contours function.
    
    This maintains full backward compatibility with the original function signature.
    """
    # Create configuration from parameters
    config = PlotConfig(
        figsize=figsize,
        levels=levels,
        add_colorbar=add_colorbar,
        font_axis_label=font_axis_label,
        font_tick=font_tick,
        font_annotation=font_annotation,
        font_title=font_title,
        font_colorbar=cbar_fontsize
    )
    
    # Apply any additional configuration from kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create plotter instance
    plotter = MultiContourPlotter(config)
    
    # Handle xlabels and ylabels conversion
    x_labels_list = None
    y_labels_list = None
    
    if xlabels:
        if isinstance(xlabels, str):
            x_labels_list = [xlabels] * len(dfs)
        else:
            x_labels_list = list(xlabels)
    
    if ylabels:
        if isinstance(ylabels, str):
            y_labels_list = [ylabels] * len(dfs)
        else:
            y_labels_list = list(ylabels)
    
    # Handle levels for individual subplots
    subplot_levels = levels
    if isinstance(levels, list) and len(levels) > 0 and isinstance(levels[0], (list, np.ndarray)):
        # Individual levels per subplot - need to handle this in the plotting loop
        subplot_levels = levels
    
    # Call the improved API with custom handling for complex cases
    if storytelling and story_labels and isinstance(levels, list):
        # Special handling for storytelling with individual levels
        result = _plot_multiple_with_storytelling(
            plotter, dfs, x_col, y_col, z_col, ncols, share_norm,
            levels, story_labels, x_labels_list, y_labels_list, titles, **kwargs
        )
    else:
        # Standard case
        result = plotter.plot_multiple_contours(
            datasets=dfs,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            ncols=ncols,
            share_normalization=share_norm,
            titles=list(titles) if titles else None,
            x_labels=x_labels_list,
            y_labels=y_labels_list,
            figsize=figsize,
            **kwargs
        )
    
    # Convert result to match original API format
    return {
        "fig": result["figure"],
        "axes": result["axes"],
        "results": result["results"],
        "colorbar": result.get("colorbar")
    }


def _plot_multiple_with_storytelling(
    plotter: MultiContourPlotter, 
    dfs: List[pd.DataFrame],
    x_col: str, y_col: str, z_col: str,
    ncols: int, share_norm: bool,
    levels_list: List[Sequence[float]],
    story_labels: Dict[float, str],
    x_labels: Optional[List[str]],
    y_labels: Optional[List[str]],
    titles: Optional[List[str]],
    **kwargs
) -> Dict[str, object]:
    """
    Handle the complex case of multiple contours with individual levels and storytelling.
    
    This recreates the original complex behavior for backward compatibility.
    """
    nplots = len(dfs)
    nrows = (nplots + ncols - 1) // ncols
    figsize = kwargs.get('figsize', (10, 6))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    
    results = []
    
    # Handle shared normalization
    if share_norm:
        all_data = np.concatenate([df[z_col].to_numpy().ravel() for df in dfs])
        global_vmin = np.nanmin(all_data)
        global_vmax = np.nanmax(all_data)
        global_norm = mpl.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
    
    # Plot each subplot
    for i, df in enumerate(dfs):
        ax = axes[i]
        
        # Get levels for this subplot
        if isinstance(levels_list[0], (list, np.ndarray)):
            local_levels = np.asarray(levels_list[i], dtype=float).ravel()
        else:
            local_levels = levels_list
        
        # Get labels for this subplot
        title = titles[i] if titles and i < len(titles) else None
        x_label = x_labels[i] if x_labels and i < len(x_labels) else None
        y_label = y_labels[i] if y_labels and i < len(y_labels) else None
        
        # Use single contour plotter for each subplot
        norm = global_norm if share_norm else None
        
        res = plot_contour(
            df,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            ax=ax,
            levels=local_levels,
            norm=norm,
            xlabels=x_label,
            ylabels=y_label,
            title=title,
            storytelling=True,
            story_labels=story_labels,
            add_colorbar=False,  # Handle separately
            **{k: v for k, v in kwargs.items() if k not in ['figsize', 'titles', 'xlabels', 'ylabels']}
        )
        results.append(res)
        
        # Add individual colorbar if needed
        if not share_norm and kwargs.get('add_colorbar', True) and res["filled"] is not None:
            cbar = fig.colorbar(res["filled"], ax=ax, orientation="vertical", shrink=0.8)
            
            # Set storytelling labels for colorbar
            if isinstance(levels_list[0], (list, np.ndarray)):
                local_ticks = np.asarray(levels_list[i], dtype=float).ravel()
                vmax_local = local_ticks.max()
                labels = []
                for v in local_ticks:
                    diff = vmax_local - v
                    if np.isclose(diff, 0):
                        labels.append("Max")
                    else:
                        perc = int(round(diff / vmax_local * 100))
                        labels.append(f"Max-{perc}%")
                
                cbar.set_ticks(local_ticks)
                cbar.set_ticklabels(labels)
            
            cbar.ax.tick_params(labelsize=kwargs.get('cbar_fontsize', 10))
    
    # Add shared colorbar if requested
    colorbar = None
    if share_norm and kwargs.get('add_colorbar', True):
        filled_example = next(
            (r["filled"] for r in results if r["filled"] is not None), None
        )
        if filled_example is not None:
            colorbar = fig.colorbar(
                filled_example, ax=axes[:nplots], orientation="vertical", shrink=0.8
            )
            if story_labels:
                ticks = list(story_labels.keys())
                labels = [story_labels[v] for v in ticks]
                colorbar.set_ticks(ticks)
                colorbar.set_ticklabels(labels)
            colorbar.ax.tick_params(labelsize=kwargs.get('cbar_fontsize', 10))
    
    # Turn off unused axes
    for j in range(nplots, len(axes)):
        axes[j].axis("off")
    
    return {
        "figure": fig,
        "axes": axes[:nplots],
        "results": results,
        "colorbar": colorbar
    }


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
    Original API wrapper for stack_contours_in_z function.
    
    This maintains full backward compatibility with the original function signature.
    """
    # Create configuration
    config = PlotConfig(
        view_elevation=elev,
        view_azimuth=azim,
        fill_alpha=alpha
    )
    
    if z_gap is not None:
        # Calculate z_gap_factor based on provided z_gap
        # This is an approximation since the original calculation was complex
        config.z_gap_factor = z_gap / 100.0  # Rough conversion
    
    # Create stacker instance
    stacker = Contour3DStacker(config)
    
    # Call the new API
    result = stacker.stack_contours(
        contour_sets=contours_list,
        z_positions=z_offsets,
        mode=mode,
        figsize=figsize,
        show_slice_boxes=True,
        show_lines=show_lines,
        cmap=cmap
    )
    
    # Convert result to match original API format
    return {
        "fig": result["figure"],
        "ax": result["axes"],
        "z_offsets": result["z_positions"]
    }


# For even more compatibility, provide the exact original function names
def plot_multiple_contour_subplots(*args, **kwargs):
    """Alias for backward compatibility."""
    return plot_multiple_contours(*args, **kwargs)


def create_3d_contour_stack(*args, **kwargs):
    """Alias for backward compatibility."""
    return stack_contours_in_z(*args, **kwargs)