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
    """Interpolate Z over a finer (X, Y) grid."""
    Xf = np.linspace(np.nanmin(X), np.nanmax(X), resolution)
    Yf = np.linspace(np.nanmin(Y), np.nanmax(Y), resolution)
    Xf, Yf = np.meshgrid(Xf, Yf)
    Zf = griddata((X.flatten(), Y.flatten()), Z.flatten(), (Xf, Yf), method=method)
    return Xf, Yf, Zf


# -----------------------------------------------------------------------------
def plot_contour(
    ax: Axes,
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

    Library code should not call plt.show() or mutate global rcParams.
    Returns the LineContourSet (cs).
    """
    piv = df.pivot_table(index=y_col, columns=x_col, values=z_col)

    # ensure numeric axes
    X, Y = np.meshgrid(
        piv.columns.to_numpy(dtype=float),
        piv.index.to_numpy(dtype=float),
    )
    Z = piv.to_numpy(dtype=float)  # <-- uppercase Z, consistent with usage below

    if interp:
        X, Y, Z = interpolate_grid(X, Y, Z)
        Z = np.clip(Z, 0.0, 1.0)

    cs = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=1.0)

    if annot:
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    if highlight:
        zmin, zmax = np.nanmin(Z), np.nanmax(Z)
        thresh = np.nanpercentile(Z, 80)
        if isinstance(levels, int):
            fill_levels = np.linspace(max(thresh, zmin), zmax, max(2, levels))
        else:
            lv = np.asarray(list(levels))
            lv = lv[lv >= thresh]
            fill_levels = lv if lv.size >= 2 else np.array([max(thresh, zmin), zmax])
        Zmask = np.where(Z >= thresh, Z, zmin - 1.0)
        ax.contourf(X, Y, Zmask, levels=fill_levels, cmap="Blues")  # <-- X, not Z
    else:
        ax.contourf(X, Y, Z, levels=levels, cmap="Blues")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title:
        ax.set_title(title)

    return cs
