import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contourlab.plotting import plot_contour

# --- Create a grid of (x, y) points ------------------------------------------
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)

# --- Example surface (Gussian + some noise) -----------------------------------
Z = np.exp(-(X**2 + Y**2) / 5) + 0.05 * np.random.randn(*X.shape)

df = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "z": Z.ravel()})

res = plot_contour(
    df,
    x_col="x",
    y_col="y",
    z_col="z",
    levels=15,  # number of contour levels
    interp=True,  # smoother surface
    add_colorbar=True,  # show colorbar
    cmap="Blues",
    show_stats=True
)

plt.show()
