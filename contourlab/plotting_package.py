# Import the backward compatible wrappers
from .backward_compatible_api import (
    plot_contour,
    plot_multiple_contours, 
    stack_contours_in_z,
    plot_multiple_contour_subplots,
    create_3d_contour_stack
)

# Also make the new classes available for advanced users
from .plotting_refactored import (
    ContourPlotter,
    MultiContourPlotter, 
    Contour3DStacker,
    PlotConfig,
    ContourPlotError,
    quick_contour_plot,
    quick_multi_contour_plot,
    quick_3d_stack
)

# Version info
__version__ = "0.2.0"
__author__ = "Elham Zakeri"

# Define what gets imported with "from plotting_package import *"
__all__ = [
    # Original API functions (backward compatible)
    "plot_contour",
    "plot_multiple_contours", 
    "stack_contours_in_z",
    
    # New API classes (for advanced usage)
    "ContourPlotter",
    "MultiContourPlotter",
    "Contour3DStacker", 
    "PlotConfig",
    "ContourPlotError",
    
    # Quick convenience functions
    "quick_contour_plot",
    "quick_multi_contour_plot", 
    "quick_3d_stack",
]