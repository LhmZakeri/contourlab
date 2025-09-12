import warnings
from dataclasses import dataclass
from typing import Union, Sequence, Tuple, Optional

import pandas as pd 
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
# =======================================================================
if __name__ == "__main__":

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