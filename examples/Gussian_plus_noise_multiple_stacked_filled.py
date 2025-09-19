import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contourlab.plotting_refactored_edit import MultiContourPlotter, Contour3Dstacker, ContourPlotter, PlotConfig

if __name__ == "__main__":
    # --- Create a grid of (x, y) points ------------------------------------------
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # --- Example surface (Gussian + some noise) -----------------------------------
    Z = np.exp(-(X**2 + Y**2) /0.5)
    Z1 = 5 * np.exp(-((X-0.5)**2 + (Y+0.3)**2) / 0.3) + \
    2 * np.exp(-((X+0.8)**2 + (Y-0.2)**2) / 0.8) #+ \
    #0.5 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    Z2 = np.exp(-(X**2 + Y**2) /0.2)

    df = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "z": Z.ravel()})
    df1 = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "z": Z1.ravel()})
    df2 = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "z": Z2.ravel()})

    config = PlotConfig()
    cp = ContourPlotter(config)
    mcp = MultiContourPlotter(config)
    res_multi = mcp.plot_multiple_contours(

    datasets=[df, df1, df2],
    x_col="x",
    y_col="y",
    z_col="z",
    ncols=2,
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
    contour_filled = True,
    add_colorbar=False,
    show=True,
    verbose = False,
    )


    c3d = Contour3Dstacker(config, verbose=False)
    c3d.stack_contours(res_multi['results'],
    z_positions=3.5*np.arange(3),
    mode= "filled" ,#'line',"filled"
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