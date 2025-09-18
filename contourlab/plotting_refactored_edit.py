import warnings
from dataclasses import dataclass
from typing import Union, Sequence, Tuple, Optional, List, Dict, Any

import pandas as pd 
import numpy as np 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    # -------------------------------------------------------------------
    def __str__(self) -> str:
        """
        Provides a well-formatted string representation of the configuration.
        """
        config_str =(
            f"--- Plot Configuration ---------\n"
            f"----[Figure Setting]------------\n"
            f"  -Figure Size:       {self.figsize}\n"
            f"  - DPI:              {self.dpi}\n"
            f"----[font Setting]--------------\n"
            f"  - Axis Label:       {self.font_axis_label}\n"
            f"  - Ticks:            {self.font_tick}\n"
            f"  - Annotations:      {self.font_annotation}\n"
            f"  - Title:            {self.font_title}\n"
            f"  - Colorbar:         {self.font_colorbar}\n"
            f"----[Contour Setting]-----------\n"
            f"  - Levels:           {self.levels}\n"
            f"  - Colormap:         {self.cmap}\n"
            f"  - Line Colors:      {self.line_colors}\n"
            f"  - line Widths:      {self.line_widths}\n"
            f"----[Behavior Flags]------------\n"
            f"  - Interpolate:      {self.interpolate}\n"
            f"  - Highlight:        {self.highlight}\n"
            f"  - Annotate:         {self.annotate}\n"
            f"  - Add Colorbar:     {self.add_colorbar}\n"
            f"  - Contour Filled:   {self.contour_filled}\n"
            f"----[Highlighting Settings]-----\n"
            f"  - Threshold:        {self.percentile_threshold}%\n"
            f"----[3D Settings]---------------\n"
            f"  - View Elevation:   {self.view_elevation}\n"
            f"  - View Azimuth:     {self.view_azimuth}\n"
            f"  - Z_Gap Factor:     {self.z_gap_factor}\n"
            f"  - Fill Alpha:       {self.fill_alpha}\n"
            f"--------------------------------\n"
        )
        return config_str
# -----------------------------------------------------------------------
class ContourPlotError(Exception):
    """Custom exception for ccontour plotting errors."""

    pass


# -----------------------------------------------------------------------
class ContourPlotter:
    """Main class for creating 2D contour plots with advanced features.
    
    This class handles data validation, grid preparation, and the generation
    of both filled and line contour plots. It uses a configuration object
    to style plots and manage behavior flags.
    """

    # -------------------------------------------------------------------
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize the ContourPlotter with a given configuration.

        Parameters:
        config: An optional PlotConfig object to customize plot settings.
                       If None, a default configuration is used.
        """
        self.config = config or PlotConfig()
        self._validate_configue()
    # -------------------------------------------------------------------
    def _validate_configue(self) -> None:
        """
        Validate configuration parameters to ensure they are within valid ranges.
        Raises:
        ContourPlotError: If percentile_threshold or filled_alpha are out
          of range.
        """
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
        """Validates the input DataFrame and required columns.
        
        Parameters:
            df: The pandas DataFrame to validate.
            x_col: The name of the column for the x-axis
            y_col: The name of the column for the y-axis
            z_col: The name of the column for the z-axis 
        Raises:
            ContourPlotError: If the DataFrame is empty or a required column is 
            missing.
            UserWarning: If the number of data points is very low.
        
        """
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
        """
        Converts scattered DataFrame data into a grid format suitable for contour
        plotting.

        Interpolation is performed if specified in the configuration.

        Paramters:
        df: The pandas DataFrame containing the data.
        x_col: The column name for x-coordiantes.
        y_col: The column name for y_coordinates.
        z_col: The column name for z_coordinates.
        config: A plotConfig object specifying plot settings.

        Returns: 
        A tuple of three Numpy arrays (X, y, Z) representing the gridded data.

        Raises:
        ContourPlotError: If the grid data preparation fails.

        """
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
        """
        Creates a new PlotConfig instance by applying overrides to the current
        config.

        Parameters:
        Override a dictionary of config attributes to override.

        Returns:
        A new PlotConfig object with the applied overrides.

        """
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
        """
        Configures axis labels, ticks, and the title for a given subplot.

        Parameters:
        ax: The matplotlib Axes object to configure.
        x_label: The label for the x-axis.
        y_label: The label for the y-axis.
        title: The title of the subplot.
        
        """
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
        """
        Adds a single, shared colorbar for a set of subplots. 

        This method finds a suitable mappable object from the list of plot results 
        and creates a single colorbar that applies to all subplots.

        Parameters:
        fig: The matplotlib Figure object
        axes: A list of matplotlib Axes objects.
        results: A list of dictionaries containing plot results for each subplot.
        norm: The normalization object for the color scale.
        colorbar_labels_set: Optional dictionary of labels for the colorbar ticks.

        Raises:
        ContourPlotError: If no mappable contour object is found.

        Returns:
        The created matplotlib Colorbar object.
        """
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
        """
        Creates filled contour regions on an Axes object.

        This method can either use standard contour filling or a 
        highlight function based on the configuration.

        Params:
        ax: The matplotlib Axes object for plotting.
        X: The 2D array of x-coordinates.
        Y: The 2D array of y-coordinates.
        Z: The 2D array of z-values.
        levels: The conotur levels.
        config: The PlotConfig object.
        norm: The normalization object.

        Returns:
        The matplotlib contourf or highlight object.
        """
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
        """
        Add inline labels to contour lines for better readability.
        
        Params:
        ax: The matplotlib Axes object.
        contour_lines: The contour lines object returned by ax.contour()
        colorbar_labels: Optional dictionary of labels for the contour lines.
        """
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
        """
        Adds a colorbar to a single plot.
        
        Parameters:
        fig: The matplotlib Figure object.
        ax: The matplotlib Axes object.
        contour_filled: The filled contour object.
        colorbar_labels: Optional dictionary of labels for the colorbar ticks.

        Returns:
        The created matplotlib Colorbar object.

        """
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
        Creates and styles a single 2D plot.

        This method generates contour lines and, optionally, filled contours
        or hyighlighted regions. It returns a dictionary containing the plot objects and
        data for further use (e.g., 3D stacking).

        Parameters: 
        X: 2D array of x-coordinates.
        Y: 2D array of y-coordinates.
        Z: 2D array of z-values.
        contour_levels: The levels for the contourlines.
            can accept int = number of contour lines , 
            or specific normalization (matplotlib.colors.Normalize object)
        ax: Optional matplotlib Axes object to plot on.
        x_label: Optional label for the x-axis.
        y_label: Optional label for the y-axis. 
        tiltle: Optional title for the plot. 
        norm : Optional normalization object for the colormap.
        colorbar_labels: Optional dictionary of labels for the colorbar. 
        individual_colorbars: A boolean flag to determine whether to add individual colorbars
                                or not.
        kwargs: Additional keyword arguments to override PlotConfig settings.

        Returns: 
            A dictionary containing the figure, axes, and contour objects.
        
        Raises: 
            ContourPlotError: If both contour_filled and highlight flags are set.

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
    """ 
    An extension of ContourPlotter for creating multiple contour subplots. 

    This class enables the creation of side-by-side contour plots with advanced
    features like shared normalization and adaptive level generation across 
    multiple datasets. 
    """
    # -------------------------------------------------------------------
    def _validate_multi_plot_options(self, adaptive_levels, levels_arg,
                                      levels_step, highlight, contour_filled):
        """
        Validates configuration options specific to multi-contour plots.

        This ensures that mutually exclusive options (e.g., adaptive levels
        and manual levels) are not used together.

        Parameters:
        adaptive_levels: Flag for generating levels automatically using 'log',
                        'quantile', or 'linear' method
        levels_arg : User-specified levels.
        levels_step: User-specified level step. ( to have certain step size between
                     contours levels)
        highlight: Flag to highlight a specific region.
        contour_filled: Flag to create filled contours with cmap colormap.
        
        Raises:             
        ContourPlotError:
        If conflicting options are selected.
        
        """
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
        Converts a list of DataFrame into a list of gridded Z-value arrays.

        This step is neccessary for calculting shared normalization and adaptive
        levels based on the full data range. 

        Parameters:
        datasets: A list of pandas DataFrames.
        x_col: The column name for x-coordinates.
        y_col: The column name for y-coordinates.
        z_col: The column name for z-values.
        verbose: If True, prints status messages. 

        Returns: 
        A list of Numpy arrays for X, Y, and Z.
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
        """
        Calculate a shared normalization object for all plots.

        The min and max values are determined from the combined range of 
        all gridded Z arrays. 

        Parameters:
        Z_all: A list of gridded Z-value arrays.
        verbose: If True, prints the calculated range. 

        Returns:
        A matplotlib Normalize object.
        """
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
        """
        Calculates a robust normalization using percentiles to exclude outliers.

        Parameters:
        Z_all: A list of gridded Z-value arrays.
        vmin_percentile: The lower percentile for normalization.
        vmax_percentile: The upper percentile for normalizaton.
        verbose: If True, prints the calculated percentile range.

        Returns:
        A matplotlib Normalize object.
        """
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
        """
        Creates a uniform set of contour levels from the combined data. 

        This allows for consistent coloring amd contour lines across all 
        subplots.

        Parameters:
        Z_all: Alist of gridded Z-value arrays. 
        num_levels: The desired number of contour levels.
        method: The method for creating levels ('quantile', 'log', 'linear')
        verbose: If True, prints the generated levels.

        Returns:
        A Numpy array of the generated contour levels. 
        """
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
        """
        Creates a multi-panel grid of contour plots from a list of datasets.

        This method coordinates the plotting of multiple datasets in a single 
        figure, with options for shared color scales and dynamically generated
        contour levels.

        Parameters:
        datasets: A single DataFrame or a list of DataFrames to plot.
        x_col: The column name for x-coordinates.
        y_col: The column name for y-coordinates.
        z_col: The column name for z-values.
        ncols: The number of columns in the subplot grid. 
        shared_normalization: If True, uses a single color for all plots.
        robust_normalization: If True, uses percentiles to ignore outliers.
        adaptive_levels:If True, generates contour levels automatically. 
        level_method: The method of adaptive levels ('quantile', 'linear', 'log')
        titles: Optional list of titles for each subplot.
        x_labels: Optional list of x-axis labels.
        y_labels: Optional list of y-axis labels.
        colorbar_labels_set : Optional list of dictionaries for colorbar labels.
        show: If True, displays the plot.
        savepath: Optional path to save the figure.
        verbose: If True, prints progress messages. 
        kwargs: Additional keyword arguments to pass to plot_single_contour 

        Raises:
        ContourPlotError:
        If any of the input datasets are empty or if there are conflicts
        inplotting options. 

        Returns:
        A dictionary containing the figure, axes, plot results, and metadata.

        """
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
            "shared_norm": shared_norm,
            "gridded_data":{
                "X_all": X_all,
                "Y_all": Y_all,
                "Z_all": Z_all,
            }, 
            "final_levels": final_levels,
        }      
# -----------------------------------------------------------------------
class Contour3Dstacker:
    """Specialized class for creating 3D stacked visualizations."""
    # -------------------------------------------------------------------
    def __init__(self, config:Optional[PlotConfig] = None, verbose=False):
        self.config = config or PlotConfig()
        if verbose:
            print(f"Contour3Dstacker created with the following configuration:\n{self.config}")
    # -------------------------------------------------------------------
    def _calculate_spatial_extents(self, valid_contours: List[Tuple[int, Any]])-> Dict[str, float]:
        """Calculate the spatial bounds of all contour data."""
        x_min, x_max = float("inf"), -float("inf")
        y_min, y_max = float("inf"), -float("inf")
        
        for _, plot_data_dict in valid_contours:
            cs = plot_data_dict.get("contour_lines")
            if cs is None:
                cs = plot_data_dict.get("contour_filled")
            # If no contour data was found, skip this plot
            if cs is None:
                continue

            if not hasattr(cs, "collections") or not cs.collections:
                continue
    # --- Skip in valid paths in countours ----------------------------    
            for coll in cs.collections:
                for path in coll.get_paths():
                    vertices = path.vertices
                    if vertices.size == 0 or vertices.shape[0] < 2:
                        continue    

                    x_min = min(x_min, vertices[:, 0].min())
                    x_max = max(x_max, vertices[:, 0].max())
                    y_min = min(y_min, vertices[:, 1].min())
                    y_max = max(y_max, vertices[:, 1].max())

        # The final checks and return statement are now outside the loop
        if not all(np.isfinite([x_min, x_max, y_min, y_max])):
            raise ContourPlotError("Could not determine valid spatial extents")
        return{
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "x_range": x_max - x_min, 
            "y_range": y_max - y_min, 
        }
    # -------------------------------------------------------------------
    def _calculate_z_positions(self, n_contours:int, z_positions:
                               Optional[List[float]],
                                extents: Dict[str, float])->List[float]:
        """Calculate Z positions for stacking contours."""
        if z_positions is not None:
            if len(z_positions) != n_contours:
                raise ContourPlotError("Number of z_positions must match number of contours.")
            return list(z_positions)
            
        # --- Auro-calculate based on spatial scale ---------------------
        z_gap = self.config.z_gap_factor* max(extents["x_range"], extents["y_range"])
        return[i*z_gap for i in range(n_contours)]
    # -------------------------------------------------------------------
    def _setup_3d_view(self, ax: plt.Axes, extents:Dict[str, float], 
                       z_offsets:List[float]) -> None:
        "Configure 3D axes view and propoerties."
        ax.set_proj_type("ortho")

        # --- Set aspect ratio ------------------------------------------
        z_range =  max(z_offsets) - min(z_offsets) if len(z_offsets) > 1 else 1.0
        # (x_aspect_ratio, y_aspect_ratio, z_aspect_ratio)
        ax.set_box_aspect((extents["x_range"], extents["y_range"], max(z_range, 1.0)))

        # --- Set view angle --------------------------------------------
        ax.view_init(elev= self.config.view_elevation, azim=self.config.view_azimuth)

        # --- Set limits with margins -----------------------------------
        margin_x = 0.02 * extents["x_range"]
        margin_y = 0.02 * extents["y_range"]
        ax.set_xlim(extents["x_min"] - margin_x, extents["x_max"] + margin_x)
        ax.set_ylim(extents["y_min"] - margin_y, extents["y_max"] + margin_y)

    # -------------------------------------------------------------------
    def _render_line_contours(self, ax:plt.Axes, valid_contours:
                              List[Tuple[int, Any]], z_offsets: List[float], **kwargs) -> None:
        """Render conoturs as lines in 3D."""
        for (_, plot_data_dict), z_pos in zip(valid_contours, z_offsets):
            cs = plot_data_dict.get('contour_lines')

            if cs is None:
                continue

            if not hasattr(cs, 'collections'):
                continue

            for coll in cs.collections:
                edge_colors = coll.get_edgecolor()
                color = edge_colors[0] if len(edge_colors) > 0 else 'k'

                for path in coll.get_paths():
                    vertices = path.vertices
                    if vertices.shape[0] < 2:
                            continue
                    
                    x, y = vertices[:, 0], vertices[:, 1]
                    z = np.full_like(x, z_pos, dtype=float)
                    ax.plot(x, y, z, color=color, linewidth=self.config.line_widths)
    # -------------------------------------------------------------------
    def _render_filled_contours(
        self, 
        ax: plt.Axes, 
        valid_contours: List[Tuple[int, Any]], 
        z_offsets: List[float], 
        **kwargs,
    ) -> None:
        """Render filled contours in 3D by reusing the 2D filled collections directly."""

        for (_, plot_data_dict), z_pos in zip(valid_contours, z_offsets):
            cs_filled = plot_data_dict.get("contour_filled")
            cs_lines = plot_data_dict.get("contour_lines")

            if cs_filled is None:
                continue
            if not hasattr(cs_filled, "collections") or not cs_filled.collections:
                continue

            for coll in cs_filled.collections:
                facecolor = coll.get_facecolor()
                if len(facecolor) == 0:
                    continue
                color = facecolor[0]

                for path in coll.get_paths():
                    vertices = path.vertices
                    if len(vertices) < 3:  # must have area
                        continue
                    x, y = vertices[:, 0], vertices[:, 1]
                    z = np.full_like(x, z_pos)

                    verts = [list(zip(x, y, z))]
                    poly = Poly3DCollection(
                        verts,
                        facecolor=color,
                        alpha=self.config.fill_alpha,
                        linewidths=0
                    )
                    ax.add_collection3d(poly)

            # Colored contour outline color 
            if cs_lines is not None and hasattr(cs_lines, "collections"):
                for coll in cs_lines.collections:
                    for path in coll.get_paths():
                        v = path.vertices 
                        if len(v) < 2:
                            continue
                        x, y = v[:, 0], v[:, 1]
                        z = np.full_like(x, z_pos+ 1e-3)

                        ax.plot(x, y, z, color = self.config.line_colors,
                                 linewidth=self.config.line_widths)

    # -------------------------------------------------------------------
    def _add_slice_boxes(self, ax:plt.Axes, extents: Dict[str, float],
                          z_offsets:List[float]) -> None:
        """Add rectangular boxes to show slice boundaries."""
        for z_pos in z_offsets:
            ax.plot(
                [extents["x_min"], extents["x_max"], extents["x_max"], extents["x_min"], extents["x_min"]],
                [extents["y_min"], extents["y_min"], extents["y_max"], extents["y_max"], extents["y_min"]], 
                [z_pos]*5, 
                color = "k", 
                linewidth = 1, 
                alpha = 0.7,
            )
    # -------------------------------------------------------------------
    def _style_3d_plot(self, ax:plt.Axes, mode:str) -> None: 
        """Apply final styling to 3D plot."""
        ax.set_xlabel("X Axis", fontsize=self.config.font_axis_label)
        ax.set_ylabel("Y Axis", fontsize=self.config.font_axis_label)
        ax.set_zlabel("Z layers", fontsize=self.config.font_axis_label)
        ax.set_title(f"3D stacked contours ({mode} mode)", fontsize=self.config.font_title)
        plt.tight_layout()

    # -------------------------------------------------------------------
    def stack_contours(
            self, 
            contour_sets: List[Any],
            gridded_data: Dict[str, List[np.ndarray]],            
            z_positions: Optional[List[float]] = None,
            figsize: Optional[Tuple[float, float]] = None,
            mode: str = "line",
            final_levels: Optional[List[float]] = None,
            show_slice_boxes : bool = True, 
            show: bool = False,
            savepath: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]: 
        """
        stack 2D contour plots in 3D space.
        """
        if not contour_sets:
            raise ContourPlotError("No contour sets provided.")
        
        # --- Filter out None contour sets ------------------------------
        valid_contours = [(i, cs) for i, cs in enumerate(contour_sets) if cs is not None]
        if not valid_contours:
            raise ContourPlotError("No valid contour sets found")
        
        # --- Setup figure ----------------------------------------------
        figsize = figsize or (10, 8)
        fig = plt.figure(figsize=figsize, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection="3d")

        # --- Calculate spatial extents and Z positions -----------------
        extents = self._calculate_spatial_extents(valid_contours)
        z_offsets = self._calculate_z_positions(len(valid_contours), z_positions, extents)

        # --- Configure 3D view -----------------------------------------
        self._setup_3d_view(ax, extents, z_offsets)

        # --- Render contours based on mode -----------------------------
        if mode == "line":
            self._render_line_contours(ax, valid_contours, z_offsets, **kwargs)
        elif mode == "filled":
            self._render_filled_contours(ax, valid_contours, z_offsets)

        else:
            raise ContourPlotError(f"Unknown rendering mode: {mode}")
        # --- Add slice boxes if requested ------------------------------
        if show_slice_boxes:
            self._add_slice_boxes(ax, extents, z_offsets)
        # --- Final Styling ---------------------------------------------
        self._style_3d_plot(ax, mode)
        if savepath:
            fig.savefig(savepath, dpi=self.config.dpi, bbox_inches="tight")
        if show:
            plt.show()
        
        return{
            "figure": fig, 
            "axes": ax, 
            "z_positions": z_offsets,
            "extents": extents,
            "mode": mode
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
        interpolate= True,
        highlight = False,
        annotate= True,
        add_colorbar= False,
        contour_filled = False,
        # --- Highlighting ---
        percentile_threshold= 80.0,
        # --- 3D setting ---
        view_elevation = 22,
        view_azimuth = -60,
        z_gap_factor = 0.15,
        fill_alpha = 0.7,
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
    results = mcp.plot_multiple_contours(
        datasets=dataset,
        x_col = "aniso_phase",
        y_col = "wavelength",
        z_col = "simdur",
        ncols = 3,
        interpolate=True, 

        shared_normalization= True,
        robust_normalization = False, # shared_normalization should be True also 
        adaptive_levels = False, 
        level_method = 'quantile', #'Log'/'linear'/'quantile',

        #levels=levelsset,#levelsset / 5, 
        levels = 10,
        #levels_step = 1,

        titles=["Sigma (4, 2)", "Sigma (5, 5)", "Sigma (6, 2)", "Sigma (8, 4)", "Sigma (9, 3)"],        
        #x_labels
        y_labels=["Mean of Success Rate", None, None, "Mean of Success Rate", None], 

        annotate=False, 
        highlight = False,
        percentile_threshold=60,        
        contour_filled = True, 

        #colorbar_labels_set=label_list,  
        add_colorbar=True, 
        show=True,
        verbose = False,  
    )
    c3d = Contour3Dstacker(plot_config, verbose=False)
    c3d.stack_contours(results['results'],
                    #z_positions=np.arange(5),
                    mode= "filled",#'line',"filled"
                    show=True, 
                    figsize=(12, 10), 
                    show_slice_boxes=True,
                    cmap = "Blues",
                    shared_norm = results['shared_norm'],
                    gridded_data=results["gridded_data"], 
                    show_line = True, 
                    final_levels=results["final_levels"],
                    line_colors = 'k',
                    )

