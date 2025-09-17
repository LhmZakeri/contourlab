import warnings
from dataclasses import dataclass
from typing import Union, Sequence, Tuple, Optional, List, Dict, Any

import pandas as pd 
import numpy as np 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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
class ContourPlotter:
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
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str, 
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

                # Fallback : if result is all NaN (common for cubic on sparse grids), retry 
                # with 'nearest
                if Zi is None or np.all(~np.isfinite(Zi)):
                    Zi = griddata((X.ravel(), Y.ravel()), Z.ravel(), (Xi, Yi), method='nearest')
            else:
                Xi, Yi, Zi = X, Y, Z

            return Xi, Yi, Zi

        except Exception as e:
            raise ContourPlotError(f"Failed to prepare grid data: {e}")
    # -------------------------------------------------------------------
    def _apply_config_override(self, overrides: Dict[str, Any]) -> PlotConfig:
        """Create a local config with overrides applied."""
        config_dict = self.config.__dict__.copy()
        config_dict.update(overrides)
        return PlotConfig(**config_dict)
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
    def _add_shared_colorbar(self, fig, axes, results, norm , colorbar_labels_set=None):
        mappable = None
        for res in results:
            if "contour_filled" in res and res["contour_filled"] is not None:
                mappable = res["contour_filled"]
                break
            elif "contour_lines" in res and res["contour_lines"] is not None:
                mappable = res["contour_lines"]
                break
        if mappable is None:
            raise ContourPlotError("No contour object found for shared colorbar.")

        cbar = fig.colorbar(
            mappable,
            ax=axes,
            orientation="vertical",
            fraction=0.05,
            pad=0.05
        )
        if colorbar_labels_set:
            first_labels_dict = colorbar_labels_set[0]

            tick_positions = mappable.levels

            tick_labels = [label for _, label in sorted(first_labels_dict.items())]
            if len(tick_positions) == len(tick_labels):
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels(tick_labels)
            else: 
                ContourPlotError("Number of levels and level labels are not the same.")
                

        return cbar
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
        X, Y, Z,
        contour_levels,
        ax: Optional[plt.Axes] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        title: Optional[str] = None,
        norm: Optional[Normalize] = None,
        colorbar_labels: Optional[Dict[float, str]] = None,
        individual_colorbars: Optional[bool] = True,
        **kwargs
        )-> Dict[str, Any]:        
        """
        Create a single contour plot
        """
        local_config = self._apply_config_override(kwargs)
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
        if individual_colorbars and local_config.contour_filled is not None:
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
class MultiContourPlotter(ContourPlotter):    
    """ Extended plotter for multiple contour subplots."""
    # -------------------------------------------------------------------
    def _validate_multi_plot_options(self, adaptive_levels, levels_arg,
                                      levels_step, highlight, contour_filled):
        """Validate options specific to multi-contour plots."""
        # Constraint 1: adaptive_levels vs. manual levels 
        if adaptive_levels and (levels_arg is not None or levels_step is not None):
            raise ContourPlotError(
                "You cannot use 'adaptive_levels=True' in combination with "
                "either 'levels_step'."
                "Choose one method for setting the contour levels."
            )

        # Constraint 2: filled vs. highlight
        if contour_filled and highlight:
            raise ContourPlotError(
                "You cannot set both 'contour_filled=True' and 'highlight=True'. "
                "Please choose one method for filling the contours."
            )
        
        # Constraint 3: highlight requires levels
        if highlight and levels_arg is None: 
            raise ContourPlotError(
                "When 'highlight=True', you must provide the 'levels' argument. "
                "The highlighting function needs specific levels to work correctly."
            )
    # -------------------------------------------------------------------
    def _collect_gridded_data(
        self, 
        datasets: List[pd.DataFrame], 
        x_col: str, 
        y_col: str, 
        z_col: str, 
        verbose = False,
    ) -> List[np.ndarray]:
        """
        Run all datasets through _prepare_data_grid and collect Z arrays.
        This ensures normalization and level selection use gridded values.
        """
        X_all, Y_all, Z_all = [], [], []
        for i, df in enumerate(datasets):
            X, Y, Z = self._prepare_data_grid(
                df=df, x_col=x_col, y_col=y_col, z_col=z_col, config=self.config
            )
            X_all.append(X)
            Y_all.append(Y)
            Z_all.append(Z)
            if verbose:
                print(f"Dataset {i}: grid shape {Z.shape}, range [{np.nanmin(Z):.2f}, {np.nanmax(Z):.2f}]")
        return X_all, Y_all, Z_all
    # -------------------------------------------------------------------
    def _calculate_shared_normalization(
        self, 
        Z_all: List[np.ndarray],
        verbose = False,
    )-> Normalize:
        """Calculate shared normalization across all gridded Z arrays."""
        all_values = np.concatenate([Z.ravel() for Z in Z_all if Z is not None])
        global_min, global_max = np.nanmin(all_values), np.nanmax(all_values)
        if verbose:
            print(f"Shared normalization range (from gridded Z): [{global_min:.2f}, {global_max:.2f}]")
        return Normalize(vmin=global_min, vmax=global_max)
    # -------------------------------------------------------------------
    def _calculate_robust_shared_normalization(
        self, 
        Z_all: List[np.ndarray], 
        vmin_percentile: float = 2.0, 
        vmax_percentile: float = 98.0,
        verbose: bool = False,
    ) -> Normalize: 
        """Robust normalization from gridded Z values using percentiles."""
        all_values = np.concatenate([Z.ravel() for Z in Z_all if Z is not None])
        vmin = np.percentile(all_values, vmin_percentile)
        vmax = np.percentile(all_values, vmax_percentile)
        if verbose:
            print(f"Robust normalization {vmin_percentile}-{vmax_percentile} percentiles: [{vmin:.2f},{vmax:.2f}]")
        return Normalize(vmin=vmin, vmax=vmax)
    # -------------------------------------------------------------------
    def _create_adaptive_levels(
        self, 
        Z_all: List[np.ndarray], 
        num_levels: int = 10, 
        method: str = 'quantile', 
        versbose: bool = False,
    ) -> np.ndarray:
        """Create levels from combined gridded Z arrays."""
        all_values = np.concatenate([Z.ravel() for Z in Z_all if Z is not None])

        if method == 'quantile': 
            levels = np.quantile(all_values, np.linspace(0, 1, num_levels))
        elif method == 'linear':
            levels = np.linspace(np.nanmin(all_values), np.nanmax(all_values), num_levels)
        elif method == 'log':
            if np.nanmin(all_values) > 0 : 
                levels = np.logspace(np.log10(np.nanmin(all_values)),
                                     np.log10(np.nanmax(all_values)), num_levels)
            else:
                levels = np.linspace(np.nanmin(all_values), np.nanmax(all_values), num_levels)
        if versbose:
            print(30*"--")
            print(f"\nCreated {len(levels)} levels using '{method}' method")
            print(f"Level range: [{levels[0]:.2f}, {levels[-1]:.2f}]")
            print(f"levels:\n{levels}")
        return levels
    # -------------------------------------------------------------------
    def plot_multiple_contours(
        self,
        datasets: Union[pd.DataFrame,List[pd.DataFrame]],   
        x_col : str,
        y_col: str,
        z_col: str,
        ncols: int = 2,
        shared_normalization: bool = True, 
        robust_normalization: bool = True,
        adaptive_levels: bool = True, 
        level_method : str = 'quantile',
        titles: Optional[List[str]]= None,
        x_labels: Optional[List[str]]=None,
        y_labels: Optional[List[str]]=None,
        colorbar_labels_set: Optional[List[Dict]] = None,
        show: bool= False,
        savepath: Optional[str] = None,
        verbose = False,
        **kwargs,
    )-> Dict[str, Any]:
        if len(datasets) == 0: 
            raise ContourPlotError("No dataset is provided.")
        for df in datasets:
            if df.empty:
                raise ContourPlotError("One of the DataFrames is empty.")
        # --- Validate options before proceeding ------------------------
        levels_arg = kwargs.get('levels')
        levels_step = kwargs.get('levels_step')
        highlight = kwargs.get('highlight', self.config.highlight)
        contour_filled = kwargs.get('contour_filled', self.config.contour_filled)

        self._validate_multi_plot_options(
            adaptive_levels, levels_arg, levels_step, highlight, contour_filled
        )
        # --- Collect gridded Z for normalization/levels ----------------
        X_all, Y_all, Z_all  = self._collect_gridded_data(datasets, x_col, y_col, z_col, verbose=verbose)
        
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
        add_colorbar_arg = kwargs.pop('add_colorbar', self.config.add_colorbar)
        shared_colorbar = shared_normalization and add_colorbar_arg
        individual_colorbars = not shared_normalization and add_colorbar_arg
        # --- Shared normalization from gridded Z ----------------------
        shared_norm = None
        if shared_normalization:
            if robust_normalization:
                shared_norm = self._calculate_robust_shared_normalization(Z_all)
            else:
                shared_norm = self._calculate_shared_normalization(Z_all, verbose=verbose)
        # --- Levels from gridded Z ------------------------------------
        levels_arg = kwargs.pop('levels', None)
        levels_step = kwargs.pop('levels_step', None)

        final_levels = None
        if adaptive_levels and levels_arg is None and levels_step is None:
            final_levels = self._create_adaptive_levels(Z_all, method=level_method,
                                                         versbose=verbose)
        elif levels_arg is not None and levels_step is None:
            final_levels = levels_arg

        # --- Plot each subplot ----------------------------------------
        for i, df in enumerate(datasets):
            ax = axes[i]   

            title = titles[i] if titles and i < len(titles) else None
            x_label = x_labels[i] if x_labels and i < len(x_labels) else None
            y_label = y_labels[i] if y_labels and i < len(y_labels) else None

            current_labels = None
            if colorbar_labels_set and i < len(colorbar_labels_set):
                current_labels = colorbar_labels_set[i]
            
            current_levels_for_plot = None
            if isinstance(final_levels, list) and len(final_levels) == len(datasets):
                current_levels_for_plot = final_levels[i]
            else:
                current_levels_for_plot = final_levels     

            result = self.plot_single_contour(
                    X_all[i], Y_all[i], Z_all[i], 
                    ax = ax,
                    title=title,
                    x_label=x_label,
                    y_label=y_label,
                    norm = shared_norm,
                    add_colorbar=add_colorbar_arg,
                    contour_levels=current_levels_for_plot,
                    colorbar_labels=current_labels,
                    individual_colorbars=individual_colorbars,
                    **kwargs
                )
            results.append(result)
        
        for j in range(nplots, len(axes)):
            axes[j].axis("off")
        
        colorbar = None
        if shared_colorbar:
            colorbar = self._add_shared_colorbar(fig, axes[:nplots], results, shared_norm, colorbar_labels_set)

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
    # -------------------------------------------------------------------
    #  Adjust config setting  
    # -------------------------------------------------------------------
    plot_config = PlotConfig(
        # --- Figure setting ---
        figsize= (6, 5), 
        dpi= 100,
        # --- Font settings ---
        font_axis_label = 12,
        font_tick = 10,
        font_annotation= 8,
        font_title= 14,
        font_colorbar = 10,
        # --- Contour settings ---
        levels = 10,            
        cmap = "Blues", 
        line_colors = "k",
        line_widths = 1.0,
        # --- Behavior flags ---
        #interpolate= True,
        #highlight = False,
        #annotate= True,
        #add_colorbar= False,
        #contour_filled = False,
        # --- Highlighting ---
        #percentile_threshold= 80.0,
        # --- 3D setting ---
        view_elevation = 22,
        view_azimuth = -60,
        z_gap_factor = 0.15,
        fill_alpha = 0.6,
    )

    datadirs = [
            "/home/elham/EikonalOptim/data/crn_table_sigma42.txt",
            "/home/elham/EikonalOptim/data/crn_table_sigma51.txt",
            "/home/elham/EikonalOptim/data/crn_table_sigma63.txt",
            "/home/elham/EikonalOptim/data/crn_table_sigma82.txt",
            "/home/elham/EikonalOptim/data/crn_table_sigma93.txt",
        ]
    
    dataset = []
    levelsset = []
    label_list = []
    for datadir in datadirs:
        data = pd.read_csv(datadir, sep="\s+")
        dataset.append(data)
        
        cp = ContourPlotter()
        _, _, Z= cp._prepare_data_grid(data, "aniso_phase", "wavelength", "simdur", config=plot_config)
        Zmax = np.nanmax(Z)

        levels = np.array(
            [
                Zmax * (0.97),
                Zmax * (0.975),
                Zmax * (0.98),
                Zmax * (0.985), 
                Zmax * (0.99),
                Zmax * (0.995),
                Zmax,
            ],
            dtype = float,            
        )
        levelsset.append(levels)
        colorbar_labels = {
            levels[6]: "Max",
            levels[5]: "Max-0.5%",
            levels[4]: "Max-1.0%",
            levels[3]: "Max-1.5%",
            levels[2]: "Max-2.0%",
            levels[1]: "Max-2.5%",
            levels[0]: "Max-3.0%",
        }
        label_list.append(colorbar_labels)
    
    mcp = MultiContourPlotter()
    mcp.plot_multiple_contours(
        datasets=dataset,
        x_col = "aniso_phase",
        y_col = "wavelength",
        z_col = "simdur",
        ncols = 3,
        interpolate=True, 

        shared_normalization= False,
        robust_normalization = False, # shared_normalization should be True also 
        adaptive_levels = False, 
        level_method = 'quantile', #'Log'/'linear'/'quantile',

        levels=levelsset,#levelsset / 5, 
        #levels_step = 1,

        titles=["Sigma (4, 2)", "Sigma (5, 5)", "Sigma (6, 2)", "Sigma (8, 4)", "Sigma (9, 3)"],        
        #x_labels
        y_labels=["Mean of Success Rate", None, None, "Mean of Success Rate", None], 

        annotate=False, 
        highlight = True,
        percentile_threshold=60,        
        contour_filled = False, 

        #colorbar_labels_set=label_list,  
        add_colorbar=True, 
        show=True,
        verbose = False,  
    )