import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contourlab.plotting import *


def make_x_shifted_normal(X, Y, shift):
    Z = np.exp(-((X-shift)**2 + Y**2) / 5) + 0.05 * np.random.randn(*X.shape)
    return Z
# ---------------------------------------------------------------------------------
def make_dataframe(X, Y, Z):
    return pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "z": Z.ravel()})
# =================================================================================
if __name__ == "__main__":

    # --- Create a grid of (x, y) points ------------------------------------------
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)

    # --- Example surface (Gussian + some noise) -----------------------------------
    Z1 = make_x_shifted_normal(X, Y, 0.05)
    Z2 = make_x_shifted_normal(X, Y, 0.1)
    Z3 = make_x_shifted_normal(X, Y, 0.15)
    Z4 = make_x_shifted_normal(X, Y, 0.2)
    Z5 = make_x_shifted_normal(X, Y, 0.25)

    df1 = make_dataframe(X, Y, Z1)
    df2 = make_dataframe(X, Y, Z2)
    df3 = make_dataframe(X, Y, Z3)
    df4 = make_dataframe(X, Y, Z4)
    df5 = make_dataframe(X, Y, Z5)

    config = PlotConfig()
    cp = ContourPlotter(config)
    mcp = MultiContourPlotter(config)
    res_multi = mcp.plot_multiple_contours(

    datasets=[df1, df2, df3, df4, df5],
    x_col="x",
    y_col="y",
    z_col="z",
    ncols=3,
    interpolate = True,
    shared_normalization=True,
    robust_normalization = False,
    adaptive_levels = False,
    level_method = 'quantile',
    levels=5,
    #levels_step = 1,
    cmap="coolwarm",
    annotate=True,
    highlight = False,
    percentile_threshold=60,
    contour_filled = False,
    add_colorbar=False,
    show=True,
    verbose = False,
    )


    c3d = Contour3Dstacker(config, verbose=False)
    c3d.stack_contours(res_multi['results'],
    z_positions=2*np.arange(5),
    mode= "line" ,#'line',"filled"
    show=True,
    figsize=(12, 10),
    show_slice_boxes=True,
    shared_norm = res_multi['shared_norm'],
    gridded_data=res_multi["gridded_data"],
    show_line = True,
    final_levels=res_multi["final_levels"],
    line_colors = 'k',
    verbose=False,
    )


