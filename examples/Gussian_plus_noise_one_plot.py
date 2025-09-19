import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contourlab.plotting_refactored_edit import *

if __name__ == "__main__":
    # --- Create a grid of (x, y) points ------------------------------------------
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)

    # --- Example surface (Gussian + some noise) -----------------------------------
    Z = np.exp(-(X**2 + Y**2) / 5) + 0.1 * np.random.randn(*X.shape)

    df = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "z": Z.ravel()})

    config = PlotConfig()
    cp = ContourPlotter(config)
    mcp = MultiContourPlotter(config)
    res_multi = mcp.plot_multiple_contours(

    datasets=[df],
    x_col="x",
    y_col="y",
    z_col="z",
    interpolate = True,
    levels=15,
    ncols=1,
    cmap="Blues",
    annotate=True,
    adaptive_levels=False,
    highlight = True,
    percentile_threshold=90,
    contour_filled = False,
    add_colorbar=True,
    show=True,
    verbose = False,
    )


  



