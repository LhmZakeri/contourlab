import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contourlab.plotting import plot_multiple_contours, stack_contours_in_z

# --- Create a grid of (x, y) points ------------------------------------------
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)

# --- Example surface (Gussian + some noise) -----------------------------------
Z = np.exp(-(X**2 + Y**2) / 5) + 0.05 * np.random.randn(*X.shape)

df = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "z": Z.ravel()})
df2 = df.copy()
df2["z"] *= 1.5

res_multi = plot_multiple_contours(
    [df, df2],
    x_col="x",
    y_col="y",
    z_col="z",
    share_norm=True,  # force both plots to use the same scale
    ncols=2,
    cmap="plasma",
)

contours_list = [r["contour"] for r in res_multi["results"]]

res_3d = stack_contours_in_z(
    contours_list,
    z_gap=0.2,
    elev=30, 
    azim=-45
)
plt.show()
