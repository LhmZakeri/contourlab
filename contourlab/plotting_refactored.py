from typing import Union, Sequence, Optional, Tuple, Dict, Any, List
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass, field
import matplotlib.cm as cm 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=self.config.dpi)
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
    def __init__(self, config: Optional[PlotConfig] = None)
        self.config = config or PlotConfig()

    def stack_contours(
        self, 
        contour_sets: List[Any], 
        z_positions: Optional[List[float]] = None,
        mode: str = "line",
        figsize: Optional[Tuple[float, float]] = None,
        show_slice_boxes: bool = True, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        stack 2D contour plots in 3D space. 

        Args: 
            contour_sets: List of contour objects to stack 
            z_positions: Z-axis positions for each contour 
            mode: "line" or "filled" rendering mode
            figsize: Figure size
            show_slice_boxes: whether to show slice boundary boxes

        Returns: 
            Dictionary with 3D figure components

        """
        if not contour_sets:
            raise ContourPlotError("No contour sets provided")
        
        # --- Filter out None contour sets --- 
        valid_contours = [(i, cs) for i, cs in enumerate(contour_sets) if cs is not None]
        if not valid_contours:
            raise ContourPlotError("No valid contour sets found")
        
        # --- Setup figure --- 
        figsize = figsize or (10, 8)
        fig = plt.figure(figsize=figsize, dpi=self.config.dpi)
        ax= fig.add_subplot(111, projection = "3d")

        # --- Calculate spatial extents and Z positions ---
        extents = self._calculate_spatial_extents(valid_contours)
        z_offsets = self._calculate_z_positions(len(valid_contours), z_positions, extents)

        # --- Configure 3D view --- 
        self._setup_3d_view(ax, extents, z_offsets)

        # --- Render contours based on mode ---
        if mode == "line":
            self._render_line_contours(ax, valid_contours, z_offsets)
        elif mode == "filled":
            self._render_filled_contours(ax, valid_contours, z_offsets)
        else:
            raise ContourPlotError(f"Unknown rendering mode:{mode}")
        

        # --- Add slice boxes if requested --- 
        if show_slice_boxes: 
            self._add_slice_boxes(ax, extents, z_offsets)

        # --- Final styling --- 
        self._style_3d_plot(ax, mode)

        return{
            "figure": fig, 
            "axes": ax, 
            "z_positions": z_offsets,
            "extents": extents, 
            "mode": mode
        }
    
    def _calculate_spatial_extents(self, valid_contours: List[Tuple[int, Any]]) -> Dict[str, float]:
        """Calculate the spatial bounds of all contour data."""
        x_min, x_max = float("inf"), -float("inf")
        y_min, y_max = float("inf"), -float("inf")

        for _, cs in valid_contours:
            if not hasattr(cs, "collections") or not cs.collections:
                continue

            for coll in cs.collections:
                for path in coll.get_paths():
                    vertices = path.vertices
                    if vertices.size == 0 or vertices.shape[0] < 2:
                        continue

                    x_min = min(x_min, vertices[:, 0].min())
                    x_max = max(x_max, vertices[:, 0].max())
                    y_min = min(y_min, vertices[:, 1].min())
                    y_max = max(y_max, vertices[:, 1].max())

            if not all(np.isfinite([x_min, x_max, y_min, y_max])):
                raise ContourPlotError("Could not determine valid spatial extents")
            
            return{
                "x_min": x_min, "x_max": x_max, 
                "y_min": y_min, "y_max": y_max,
                "x_range": x_max - x_min, 
                "y_range": y_max - y_min
            }


    def _calculate_z_positions(self, n_contours:int, z_positions: 
                               Optional[List[float]], extents: Dict[str, float]) -> List[float]:
        """Calculate Z positions for stacking contours."""
        if z_positions is not None:
            if len(z_positions) != n_contours:
                raise ContourPlotError("Number of z_positions must match number of contours.")
            return list(z_positions)
        
        # --- Auto-calculate based on spatial scale ---
        z_gap = self.config.z_gap_factor * max(extents["x_ramge"], extents["y_range"])
        return [i * z_gap for i in range(n_contours)]
    
    def _setup_3d_view(self, ax: plt.Axes, extents:Dict[str, float], z_offsets:List[float]) -> None:
        """Configure 3D axes view and properties."""
        ax.set_proj_type("ortho")

        # --- Set aspect ratio ---
        z_range = max(z_offsets) - min(z_offsets) if len(z_offsets) > 1 else 1.0 
        ax.set_box_aspect((extents["x_range"], extents["y_range"], max(z_range, 1.0)))

        # --- Set view angle --- 
        ax.view_init(elev=self.config.view_elevation, azim = self.config.view_azimuth)

        # --- Set limits with margins --- 
        margin_x = 0.02 * extents["x_range"]
        margin_y = 0.02 * extents["y_range"]
        ax.set_xlim(extents["x_min"] - margin_x, extents["x_max"] + margin_x)
        ax.set_ylim(extents["y_min"] - margin_y, extents[y_max] + margin_y)

    def _render_line_contours(self, ax:plt.Axes, valid_contours: List[
        Tuple[int, Any]], z_offsets: List[float]) -> None:
        """Render contours as lines in 3D."""
        for (original_idx, cs), z_pos in zip(valid_contours, z_offsets):
            if not hasattr(cs, 'collections'):
                continue

            for coll in cs.collections:
                edge_colors = coll.get_edgecolor()
                color = edge_colors[0] if len(edge_colors) > 0 else 'k'

                for path in coll.get_path():
                    vertices = path.vertices 
                    if vertices.shape(0) < 2:
                        continue

                    x, y = vertices[:, 0], vertices[:, 1]
                    z = np.full_like(x, z_pos, dtype=float)
                    ax.plot(x, y, z, color=color, linewidth=self.config.line_width)

    def _render_filled_contours(self, ax: plt.Axes, valid_contours: List[
        Tuple[int, Any]], z_offsets: List[float], **kwargs)-> None:
        """Render contours as filled polygons in 3D"""

        show_lines = kwargs.get("show_lines", True)
        cmap_name = kwargs.get("cmap", None)

        for (original_idx, cs), z_pos in zip(valid_contours, z_offsets):
            if not hasattr(cs, "collections") or not hasattr(cs, "levels"):
                continue 
            # --- Setup colormap if specified --- 
            cmap_obj = None
            norm = None
            if cmap_name: 
                cmap_obj = cm.get_cmap(cmap_name)
                norm = plt.Normalize(vmin=min(cs.levels), vmax=max(cs.levels))

            # --- Render each contour level --- 
            z_step = self.config.z_gap_factor * 0.1 # small z increment per level 
            for level_idx, coll in enumerate(cs.collections):
                z_level = z_pos + level_idx * z_step 

                # --- Determine face color ---
                if cmap_obj and norm: 
                    level_val = cs.levels[level_idx] if level_idx < len(cs.levels) else cs.levels[-1]
                    facecolor = cmap_obj(norm(level_val))
                else:
                    face_colors = coll.get_facecolor()
                    facecolor = face_colors[0] if len(face_colors) > 0 else (0.5, 0.5, 0.5, 1.0)

                # --- Create 3D polygons ---
                for path in coll.get_paths():
                    vertices = path.vertices
                    if vertices.shape[0] < 3:
                        continue

                    # --- Create 3D vertices ---
                    verts_3d = [(vx, vy, z_level) for vx, vy in vertices]
                    poly = Poly3DCollection([verts_3d])
                    poly.set_facecolor(facecolor)
                    poly.set_alpha(self.config.fill_alpha)
                    poly.set_edgecolor("None")
                    ax.add_collection3d(poly)

                    # --- Add contour lines if requested --- 
                    if show_lines: 
                        x, y = vertices[:, 0], vertices[:, 1]
                        z = np.full_like(x, z_level, dtype=float)
                        ax.plot(x, y, z, color="k", linewidth=0.5)
    
    def _add_slice_boxes(self, ax: plt.Axes, extents:Dict[str, float], z_offsets:List[float]) -> None:
        """Add rectangular boxes to show slice boundaries."""
        for z_pos in z_offsets:
            ax.plot(
                [extents["x_min"], extents["x_max"], extents["x_max"], extents["x_min"], extents["xmin"]],
                [extents["y_min"], extents["y_min"], extents["y_max"], extents["y_max"], extents["y_min"]],
                [z_pos] *5, 
                color = "k", 
                linewidth= 1, 
                alpha= 0.7
            )                

    def _style_3d_plot(self, ax:plt.Axes, mode: str) -> None:
        """Apply final styling to 3D plot."""
        ax.set_xlabel("X Axis", fontsize=self.config.font_axis_label)
        ax.set_ylabel("Y Axis", fontsize=self.config.font_axis_label)
        ax.set_zlabel("Z layers", fontsize=self.config.font_axis_label)
        ax.set_title(f"3D stacked Contours ({mode} mode)", fontsize=self.config.font_title)
        plt.tight_layout()

    def quick_contour_plot(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, **kwargs) -> Dict[str, Any]:
        """Quick function to create a single contour plot with minimal setup."""
        plotter = ContourPlotter()
        return plotter.plot_single_contour(df, x_col, y_col, z_col, **kwargs)
    
    def quick_multi_contour_plot(datasets: List[pd.DataFrame], x_col:str, y_col: str, z_col: str, **kwargs) -> Dict[str, Any]:
        """Qiuck function to create multiple contour plots."""
        plotter = MultiContourPlotter()
        return plotter.plot_multiple_contours(datasets, x_col, y_col, z_col, **kwargs)
    
    def quick_3d_stack(contour_sets: List[Any], **kwargs) -> Dict[str, any]:
        """Quick function to create 3D stacked contours."""
        stacker = Contour3Dstacker()
        return stacker.stack_contours(contour_sets, **kwargs)

