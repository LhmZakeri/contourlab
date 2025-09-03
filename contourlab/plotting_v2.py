from typing import Union, Sequence, Optional, Tuple, Dict, Any, List
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass, field

from .utils import interpolate_grid, highlight_region


# -----------------------------------------------------------------------
@dataclass
class PlotConfig:
    """Configuration class for contour plot styling and behavior."""

    # --- Figure setting ---
    figsize: Tuple[float, float] = (6, 5)
    dpi: int = 100

    # --- Font settings ---
    font_axis_label: int = 12
    font_tick: int = 10
    font_annotation: int = 8
    font_title: int = 14
    font_colorbar: int = 10

    # --- Contour settings ---
    levels: Union[int, Sequence[float]] = 10
    cmap: str = "Blues"
    line_colors: str = "k"
    line_width: float = 1.0

    # --- Behavior flags ---
    interpolate: bool = True
    highlight: bool = False
    annotate: bool = True
    add_colorbar: bool = False

    # --- Highlighting ---
    percentile_threshold: float = 80.0

    # --- 3D setting ---
    view_elevation: int = 22
    view_azimuth: int = -60
    z_gap_factor: float = 0.15
    fill_alpha: float = 0.6


# -----------------------------------------------------------------------
class ContourPlotError(Exception):
    """Custom exception for vontour plotting errors."""

    pass


# -----------------------------------------------------------------------
class ContourPlotter:
    """Main class for creating 2D contour plots with advanced features."""

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if (
            self.config.percentile_threshold < 0
            or self.config.percentile_threshold > 100
        ):
            raise ContourPlotError("Percentile threshold must be between 0 and 100")

        if self.config.fill_alpha < 0 or self.config.fill_alpha > 1:
            raise ContourPlotError("Fill lalpha must be between 0 and 1")

    def _validate_dataframe(
        self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str
    ) -> None:
        """Validate input DataFrame and columns."""
        if df.empty:
            raise ContourPlotError("DatFrame is empty.")

        required_cols = [x_col, y_col, z_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ContourPlotError(f"Missing columns:{missing_cols}")

        # --- Check for sufficient data points ----
        if len(df) < 4:
            warnings.warn("Very few data points may result in poor contour quality.")

    def _prepare_grid_data(
        self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert DataFrame to grid format for contour plotting."""
        try:
            pivot_df = df.pivot_table(
                index=y_col, columns=x_col, values=z_col, aggfunc="mean"
            )
            X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
            Z = pivot_df.values

            if self.config.interpolate:
                X, Y, Z = interpolate_grid(X, Y, Z)
            return X, Y, Z
        except Exception as e:
            raise ContourPlotError(f"Failed to prepare grid data: {e}")

    def _determine_levels(self, Z: np.ndarray, levels: Union[int, Sequence[float]]):
        """Determine contour levels from data and configuration."""
        data_min, data_max = float(np.nanmin(Z)), float(np.nanmax(Z))

        if isinstance(levels, int):
            return np.linsapce(data_min, data_max, levels)
        else:
            levels_arr = np.asarray(levels, dtype=float).ravel()
            # --- Ensure levels cover data range ---
            if levels_arr.min() > data_min:
                levels_arr = np.insert(levels_arr, 0, data_min)
            if levels_arr.max() < data_max:
                levels_arr = np.append(levels_arr, data_max)
            return levels_arr

    def _setup_axes(
        self,
        ax: plt.Axes,
        x_label: Optional[str],
        y_label: Optional[str],
        title: Optional[str],
    ) -> None:
        """Configure axes labels, ticks, and title."""
        if x_label:
            ax.set_xlabel(x_label, fontsize=self.config.font_axis_label)
        if y_label:
            ax.set_ylabel(y_label, fontsize=self.config.font_axis_label)
        if title:
            ax.set_title(title, fontsize=self.config.font_title)

        ax.tick_params(axis="x", labelsize=self.config.font_tick)
        ax.tick_params(axis="y", labelsize=self.config.font_tick)

    def plot_single_contour(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        ax: Optional[plt.Axes] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        title: Optional[str] = None,
        levels: Optional[Union[int, Sequence[float]]] = None,
        norm: Optional[mpl] = None,
        storytelling: bool = False,
        story_lables: Optional[Dict[float, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a single contour plot:

        Args:
            df: Input DataFrame
            x_col, y_col, z_col : Column names for axes
            ax: Optional axes to plot on
            x_label, y_label, title: Display labels
            levels: Contour levels override
            norm: Color normalization
            storytelling: Enable custom labeling
            story_labels: Custom level labels
            **kwargs: Additional configuration overrides

        Returns:
            Dictionary with plot components and metadata
        """
        # --- Apply any configuration overrides ---
        local_config = self._apply_config_overrides(kwargs)

        # --- Validate inputs ---
        self._validate_dataframe(df, x_col, y_col, z_col)

        # --- Prepare data
        X, Y, Z = self._prepare_grid_data(df, x_col, y_col, z_col)

        # --- Determine levels ---
        plot_levels = levels if levels is not None else local_config.levels
        contour_levels = self._determine_levels(Z, plot_levels)

        # --- Create axes if needed ---
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=local_config.figsize, dpi=local_config.dpi)
            careted_fig = True
        else:
            fig = ax.figure

        # --- Setup axes ---
        self._setup_axes(ax, x_label, y_label, title)

        # --- Create contour lines ---
        contour_lines = ax.contour(
            X,
            Y,
            Z,
            levels=contour_levels,
            colors=local_config.line_colors,
            linewidth=local_config.line_width,
        )
        # --- Add annotation ---
        if local_config.annotate:
            self._add_contour_annotations(ax, contour_lines, storytelling, story_lables)

        # --- Create filled contours ---
        contour_filled = None
        if local_config.highlight:
            contour_filled = self._create_filled_contours(
                ax, X, Y, Z, contour_levels, local_config, norm
            )

        # --- Add colorbar ---
        colorbar = None
        if local_config.add_colorbar and contour_filled is not None:
            colorbar = self._add_colorbar(
                fig, ax, contour_filled, storytelling, story_lables
            )

        result = {
            "figure": fig if created_fig else None,
            "axes": ax,
            "contour_lines": contour_lines,
            "contour_filled": contour_filled,
            "colorbar": colorbar,
            "levels": contour_levels,
            "data_range": (float(np.nanmin(Z)), float(np.nanmax(Z))),
        }

        return result

    def _apply_config_override(self, overrides: Dict[str, Any]) -> PlotConfig:
        """Create a local config with overrides applied."""
        config_dict = self.config.__dict__.copy()
        config_dict.update(overrides)
        return PlotConfig(**config_dict)

    def _add_contour_annotations(
        self,
        ax: plt.Axes,
        contour_lines,
        storytelling: bool,
        story_labels: Optional[Dict[float, str]],
    ) -> None:
        """Add inline labels to contour lines."""
        if storytelling and story_labels:
            ax.clabel(
                contour_lines,
                inline=True,
                fontsize=self.config.font_annotation,
                fmt=lambda v: story_labels.get(v, f"{v:.3f}"),
            )
        else:
            ax.clabel(
                contour_lines,
                inline=True,
                fontsize=self.config.font_annotation,
                fmt="%.2f",
            )

    def _create_filled_contours(
        self,
        ax: plt.Axes,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        levels: np.ndarray,
        config: PlotConfig,
        norm: Optional[mpl.colors.Normalize],
    ):
        """Create filled contour regions."""
        if config.highlight:
            # --- Use highlight region from utils ---
            return highlight_region(
                ax,
                X,
                Y,
                Z,
                percent=config.percentile_threshold,
                levels=levels,
                cmap=config.cmap,
            )
        else:
            # --- Standard filled contours
            if norm is None:
                data_min, data_max = float(np.nanmin(Z)), float(np.nanmax(Z))
                norm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)

            return ax.contourf(
                X, Y, Z, levels=levels, cmap=config.cmap, norm=norm, extend="both"
            )

    def _add_colorbar(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        contour_filled,
        storytelling: bool,
        story_labels: Optional[Dict[str, float]],
    ):
        """Add colorbar to the plot."""
        colorbar = fig.colorbar(contour_filled, ax=ax)

        if storytelling and story_labels:
            colorbar.set_ticks(list(story_labels.keys()))
            colorbar.set_ticklabels(list(story_labels.values()))

        colorbar.ax.tick_params(labelsize=self.config.font_colorbar)
        return colorbar


# -----------------------------------------------------------------------
class MultiContourPlotter(ContourPlotter):
    """ Extended plotter for multiple contour subplots."""

    def plot_multiple_contours(
        self, 
        datasets: List[pd.DataFrame],
        x_col: str, 
        y_col: str, 
        z_col: str, 
        ncols: int = 2, 
        share_normalization: bool = True, 
        titles: Optional[List[str]] = None, 
        x_labels: Optional[List[str]] = None, 
        y_labels: Optional[List[str]] = None, 
        figsize: Optional[Tuple[float, float]] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create multiple contour plots in a grid layout.

        Args: 
            datasets : List of DataFrames to plot 
            x_col, y_col, z_col : Column names 
            ncols: Number of columns in grid 
            share_normalization: Use shared color scale 
            titles, x_labels, y_labels : List of labels for each subplot 
            figsize: Figure size override 
            ** kwargs: Additional configuration 

        Returns: 
            Dictionary with figure, axes, and individual results 
        
        """
        if not datasets:
            raise ContourPlotError("No datasets provided")
        
        nplots = len(datasets)
        nrows = (nplots + ncols - 1)// ncols

        #  --- Determine figure size --- 
        if figsize is None: 
            single_size =  self.config.figsize
            figsize = (single_size[0] *ncols, single_size[1]* nrows)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dfp=self.config.dpi)
        axes = np.atleast_1d(axes).ravel()

        results = []

        # --- Claculate shared normalization if requested ---
        shared_norm = None
        if share_normalization: 
            share_norm = self._calculate_shared_normalization(datasets, z_col)
        
        # --- Plot each subplot ---
        for i, df in enumerate(datasets):
            ax = axes[i]

            # --- Get labels for this subplot --- 
            title = titles[i] if titles and i < len(titles) else None
            x_label = x_labels[i] if x_labels and i < len(x_labels) else None
            y_label = y_labels[i] if y_labels and i < len(y_labels) else None

            # --- Plot single contour --- 
            result = self.plot_single_contour(
                df, x_col, y_col, z_col, 
                ax=ax, 
                title=title, 
                x_label=x_label, 
                y_label=y_label, 
                norm = shared_norm, 
                add_colorbar=False, 
                **kwargs
            )
            results.append(result)

        # --- Turn off unused axes --- 
        for j in range(nplots, len(axes)):
            axes[j].axis("off")

        # --- Add shared colorbar if requested ---
        colorbar = None
        if self.config.add_colorbar:
            colorbar = self._add_shared_colorbar(fig, axes[:nplots], results, shared_norm)

        return {
            "figure": fig, 
            "axes": axes[:nplots],
            "results": results, 
            "colorbar": colorbar, 
            "shared_norm": shared_norm
        }

    def _calculate_shared_normalization(
            self, 
            datasets: List[pd.DataFrame],
            z_col: str
    ) -> mpl.colors.Normalize:
        """
        Calculate shared normalization across all datasets.
        """
        all_values = []
        for df in datasets: 
            all_values.extend(df[z_col].dropna().tolist())
        
        if not all_values: 
            raise ContourPlotError("No valid data found across datasets")
        
        return mpl.colors.Normalize(vmin=min(all_values), vmax=max(all_values))
    
    def _add_shared_colorbar(self, fig, axes, results, shared_norm):
        """Add a shared colorbar for multiple subplots."""
        # --- Find a valid filled contour for colorbar reference ---
        filled_example= next(
            (r["contour_filled"] for r in results if r["contour_filled"] is not None), 
            None
        )

        if filled_example is not None:
            return fig.colorbar(
                filled_example,
                ax = axes, 
                orientation="vertical", 
                shrink=0.8, 
            )
        return None
# -----------------------------------------------------------------------
class Contour3Dstacker: 
    """Specialized class for creating 3D stacked visualizations."""





