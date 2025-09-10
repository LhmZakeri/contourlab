import warnings

warnings.simplefilter("always", UserWarning)
from dataclasses import dataclass
from typing import Union, Sequence, Tuple, Optional, Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

from utils import highlight_region
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
    line_widths: float = 1.0

    # --- Behavior flags ---
    interpolate: bool = True
    highlight: bool = False
    annotate: bool = True
    add_colorbar: bool = False
    contour_filled: bool = False

    # --- Highlighting ---
    percentile_threshold: float = 80.0

    # --- 3D setting ---
    view_elevation: int = 22
    view_azimuth: int = -60
    z_gap_factor: float = 0.15
    fill_alpha: float = 0.6


# -----------------------------------------------------------------------
class ContourPlotError(Exception):
    """Custom exception for ccontour plotting errors."""

    pass


# -----------------------------------------------------------------------
class ConoturPlotter:
    """Main class for creating 2D contour plots with advanced features."""

    # -------------------------------------------------------------------
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self._validate_configue()

    # -------------------------------------------------------------------
    def _validate_configue(self) -> None:
        """Validate configuration parameters."""
        if (
            self.config.percentile_threshold < 0
            or self.config.percentile_threshold > 100
        ):
            raise ContourPlotError("Percentile threshold must be between 0 and 100.")

        if self.config.fill_alpha < 0 or self.config.fill_alpha > 1:
            raise ContourPlotError("Fill alpha must be between 0 and 1 ")

    # -------------------------------------------------------------------
    def _validate_dataframe(
        self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str
    ) -> None:
        """Validate input DataFrame and columns."""
        if df.empty:
            raise ContourPlotError("DataFrame is empty.")

        required_cols = [x_col, y_col, z_col]
        missing_col = next(
            (col for col in required_cols if col not in df.columns), None
        )
        if missing_col:
            raise ContourPlotError(f"Missing column: {missing_col}")

        if len(df) < 4:
            warnings.warn("Very few  datapoints may result in poor contour quality")

    # -------------------------------------------------------------------

    def _prepare_data_grid(
        self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
        config: PlotConfig = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert DataFrame to grid format for contour plotting."""
        try:
            pivot_df = df.pivot_table(
                index=y_col, columns=x_col, values=z_col, aggfunc="mean"
            )
            X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
            Z = pivot_df.values

            if config.interpolate:
                xi = np.linspace(X.min(), X.max(), 200)  # finer grid
                yi = np.linspace(Y.min(), Y.max(), 200)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = griddata((X.ravel(), Y.ravel()), Z.ravel(), (Xi, Yi), method='cubic')
            else:
                Xi, Yi, Zi = X, Y, Z

            return Xi, Yi, Zi

        except Exception as e:
            raise ContourPlotError(f"Failed to prepare grid data: {e}")

    # -------------------------------------------------------------------

    def _determine_levels(
        self,
        Z: np.ndarray,
        levels: Optional[Union[int, Sequence[float]]] = None,
        levels_step: Optional[float] = None,
    ) -> np.ndarray:
        """
        Determine contour levels from data and configuration.

        Args:
            Z: Data points values
            levels : the number of contour lines or the array of contours' values
        Returns:
            levels_arr : np.ndarray of levels
        """
        data_min, data_max = float(np.nanmin(Z)), float(np.nanmax(Z))

        if levels is not None and levels_step is not None:
            raise ContourPlotError("Specify either levels or step, not both.")

        if isinstance(levels, int):
            levels_arr = np.linspace(data_min, data_max, levels)

        elif isinstance(levels, (list, tuple)):
            levels_arr = np.asarray(levels, dtype=float).ravel()

        elif isinstance(levels_step, (int, float)):
            levels_arr = np.arange(data_min, data_max + levels_step, step=levels_step)

        else:
            warnings.warn(
                "No levels or step provided. Using default: 10 contour levels."
            )
            levels_arr = np.linspace(data_min, data_max, self.config.levels)

        # --- Ensure levels cover data range ----------------------------
        if levels_arr.min() > data_min:
            levels_arr = np.insert(arr=levels_arr, obj=0, values=data_min)
        if levels_arr.max() < data_max:
            levels_arr = np.insert(arr=levels_arr, obj=len(levels_arr), values=data_max)

        return levels_arr

    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    def _apply_config_override(self, overrides: Dict[str, Any]) -> PlotConfig:
        """Create a local config with overrides applied."""
        config_dict = self.config.__dict__.copy()
        config_dict.update(overrides)
        return PlotConfig(**config_dict)
    # -------------------------------------------------------------------
    def _add_acontour_annotations(
        self, 
        ax: plt.Axes,
        contour_lines,
        colorbar_labels,
    ) -> None: 
        """Add inline labels to contour lines."""
        if colorbar_labels:
            ax.clabel(
                contour_lines,
                inline = True,
                fontsize= self.config.font_annotation, 
                fmt = lambda v: colorbar_labels.get(v, f"{v:.3f}"),
            )
        else:
            ax.clabel(
                contour_lines, 
                inline = True,
                fontsize=self.config.font_annotation,
                fmt= "%.2f",
            )
    # -------------------------------------------------------------------
    def _create_filled_contours(
        self,
        ax: plt.Axes,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        levels: np.ndarray,
        config : PlotConfig, 
        norm : Optional[Normalize],
    ):
        """Create filled contour regions."""
        if config.highlight:
            return highlight_region(
                ax, 
                X, 
                Y, 
                Z, 
                percent = config.percentile_threshold,
                levels=levels,
                cmap=config.cmap,
            )
        return ax.contourf(
            X, Y, Z, levels=levels, cmap=config.cmap, norm=norm, extend="both"
        )

    # -------------------------------------------------------------------
    def _add_colorbar(
        self, 
        fig: plt.Figure,
        ax: plt.Axes, 
        contour_filled, 
        colorbar_labels: Optional[Dict[str, float]]= None,
    ):
        """Add colorbar to the plot"""
        colorbar = fig.colorbar(contour_filled, ax=ax)

        if colorbar_labels:
            colorbar.set_ticks(list(colorbar_labels.keys()))
            colorbar.set_ticklabels(list(colorbar_labels.values()))
        
        colorbar.ax.tick_params(labelsize=self.config.font_colorbar)
        return colorbar
    # -------------------------------------------------------------------
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
        levels_step: Optional[float] = None,
        norm: Optional[Normalize] = None,
        colorbar_labels: Optional[Dict[float, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a single contour plot:

        Args:
            df: Input DataFrame
            x_col, y_col, z_col : Columns' names for axes
            ax: Optional axes to plot on
            x_label, y_label, title : Display labels
            levels : Contour levels
            norm: Color normalization
            **kwargs: Additional configuration

        Returns:
            Dictionary with plot components and metadata
        """
        # ---------------------------------------------------------------
        local_config = self._apply_config_override(kwargs)
        self._validate_dataframe(df, x_col, y_col, z_col)
        X, Y, Z = self._prepare_data_grid(df, x_col, y_col, z_col, config=local_config)
        contour_levels = self._determine_levels(Z, levels, levels_step)
        
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize= local_config.figsize, dpi=local_config.dpi)
            created_fig = True
        else:
            fig = ax.figure

        self._setup_axes(ax, x_label, y_label, title)

        contour_lines = ax.contour(
            X, 
            Y, 
            Z, 
            levels = contour_levels,
            colors = local_config.line_colors, 
            linewidths = local_config.line_widths,
        )
        # ---------------------------------------------------------------
        if local_config.annotate:
            self._add_acontour_annotations(ax, contour_lines, colorbar_labels)
        
        contour_filled = None
        if local_config.contour_filled and local_config.highlight:
            raise ContourPlotError("Choose between highlight or filled method.")
        if local_config.contour_filled or local_config.highlight:
            contour_filled = self._create_filled_contours(
                ax, X, Y, Z, 
                contour_levels,
                local_config,
                norm,
            )
        
        colorbar = None
        if local_config.add_colorbar and contour_filled is not None:
            colorbar = self._add_colorbar(
                fig, ax, contour_filled, colorbar_labels
            )
        
        results = {
            "figure": fig if created_fig else None,
            "axes": ax,
            "contour_lines": contour_lines,
            "contour_filled": contour_filled, 
            "colorbar": colorbar, 
            "levels": contour_levels, 
            "data_range": (float(np.nanmin(Z)), float(np.nanmax(Z))),
        }

        return results
# -----------------------------------------------------------------------
class MultiContourPlotter(ConoturPlotter):    
    """ Extended plotter for multiple contour subplots."""

    def _calculate_shared_normalization(
        self, 
        datasets: List[pd.DataFrame],
        z_col: str
    )-> Normalize:
        
        """Calculate shared normalization across all datasets."""
        all_values = []
        for df in datasets:
            all_values.extend(df[z_col].dropna().tolist())
        
        return Normalize(vmin=np.nanmin(all_values), vmax = np.nanmax(all_values))
    # -------------------------------------------------------------------
    def _add_shared_colorbar(
            self,
            fig: plt.Figure,
            axes: Union[plt.Axes, np.ndarray],
            results: List[Dict[str, Any]],
            shared_norm: Optional[Normalize] = None,
            label: Optional[str] = None,
            **kwargs):
        """Add a shared colorbar for multiple subplots."""
        filled_example = next(
            (r["contour_filled"] for r in results if r["contour_filled"] is not None),
            None
        )
        line_example = next(
            (r["contour_lines"] for r in results if r["contour_lines"] is not None), 
            None
        )
        mappable = filled_example or line_example
        
        if mappable is None: 
            warnings.warn("No contour plots availabel for shared colorbar.")

        cbar = fig.colorbar(
            mappable,
            ax = axes, 
            orientation=kwargs.get("orientation", "vertical"),
            shrink=kwargs.get("shrink", 0.8),
            norm = shared_norm,
        )

        if label:
            cbar.set_label(label, fontsize=self.config.font_colorbar)
        
        cbar.ax.tick_params(labelsize=self.config.font_colorbar)
        return cbar
    # -------------------------------------------------------------------
    def plot_multiple_contours(
        self,
        datasets: Union[pd.DataFrame,List[pd.DataFrame]],
        x_col : str,
        y_col: str,
        z_col: str,
        ncols: int = 2,
        shared_normalization: bool = True, 
        titles: Optional[List[str]]= None,
        x_labels: Optional[List[str]]=None,
        y_labels: Optional[List[str]]=None,
        show: bool= False,
        savepath: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if len(datasets) == 0: 
            raise ContourPlotError("No dataset is provided.")
        for df in datasets:
            if df.empty:
                raise ContourPlotError("One of the DataFrames is empty.")
        
        nplots = len(datasets)
        nrows = (nplots + ncols - 1)// ncols

        figsize=kwargs.get("figsize")
        if figsize is None:
            single_size = self.config.figsize
            figsize = (single_size[0]*ncols, single_size[1]*nrows)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                                  dpi=self.config.dpi)
        axes = np.atleast_1d(axes).ravel()

        results = []

        shared_norm = None
        if shared_normalization:
            shared_norm = self._calculate_shared_normalization(datasets, z_col)

        for i, df in enumerate(datasets):
            ax = axes[i]

            title = titles[i] if titles and i < len(titles) else None
            x_label = x_labels[i] if x_labels and i < len(x_labels) else None
            y_label = y_labels[i] if y_labels and i < len(y_labels) else None

            result = self.plot_single_contour(
                df, x_col, y_col, z_col, 
                ax = ax,
                title=title,
                x_label=x_label, 
                y_label=y_label, 
                norm = shared_norm,
                add_colorbar=False,
                **kwargs
            )   
            results.append(result)

        # --- Turn off unused axes --------------------------------------
        for j in range(nplots, len(axes)):
            axes[j].axis("off")
        
        # --- Add shared colorbar if requested --------------------------
        colorbar = None
        if self.config.add_colorbar:
            colorbar = self._add_shared_colorbar(fig, axes[:nplots], results, shared_norm)

        # --- Finalize output(show or save)------------------------------
        if savepath: 
            fig.savefig(savepath, dpi=self.config.dpi, bbox_inches="tight")
        if show:
            plt.show()
            
        return{
            "figure": fig,
            "axes": axes[:nplots],
            "results": results,
            "colorbar": colorbar,
            "shared_norm": shared_norm
        }

# =======================================================================
if __name__ == "__main__":

    plot_config = PlotConfig(percentile_threshold=80, fill_alpha=0.5)

    datadir = "/home/elham/EikonalOptim/data/new_fk_table_sigma51full.txt"
    data = pd.read_csv(datadir, sep="\s+")
    
    mcp = MultiContourPlotter()
    mcp.plot_multiple_contours(
        datasets=[data, data],
        x_col="wavelength",
        y_col="aniso_phase",
        z_col="simdur",
        figsize=(10, 12),
        titles=["(5, 5)","(5, 5)"],
        y_labels=["one", "two"], 
        levels=np.arange(560, 610), 
        annotate=False,
        show=True, 
        highlight = True,
        contour_filled = False,
        percentile_threshold=80, 
        shared_normalization=False, 
        #add_colorbar= True, 
        interpolate=True)





