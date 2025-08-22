from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.axes import Axes
from typing import Union, Tuple, Iterable

# -----------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, pd.Series]
Levels = Union[int, Iterable[float]]

def interpolate_grid(
    X: ArrayLike,
    Y: ArrayLike,
    Z: ArrayLike,
    resolution: int = 100,
    method: str = "cubic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate Z over a finer (X, Y) grid.

    Parameters:
    X, Y, Z : array-like
        Input coordinates and values
    resolution : int
        Grid resolution
    method : str
        Interpolation method ('linear', 'cubic', 'nearest').

    Returns:
    Xf, Yf, Zf : np.ndarray
        Interpolated grid.
    """
    Xf = np.linspace(np.nanmin(X), np.nanmax(X), resolution)
    Yf = np.linspace(np.nanmin(Y), np.nanmax(Y), resolution)
    Xf, Yf = np.meshgrid(Xf, Yf)
    Zf = griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xf, Yf), method=method)
    return Xf, Yf, Zf

# -----------------------------------------------------------------------------
def plot_contour(
        ax, 
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        levels: Levels = 10,
        interp: bool = True,
        highlight: bool = True,
        annot: bool = True,
        title: str | None = None,
): 
    """Draw contour lines and optional filled highlight on the provided Axes.
    
    Notes
    -----
    Library code should not call plt.show() or mutate global rcParams.
    Returns the LineContourSet (cs) so callers can re-use levels/collections.
    """
    piv = df.pivot_table(index=y_col, columns=x_col, values=z_col)
    X, Y = np.meshgrid(
        piv.columns.to_numpy(dtype=float),
        piv.index.to_numpy(dtype=float)
    )
    z = piv.to_numpy(dtype=float)