import numpy as np
from scipy.interpolate import griddata


# -----------------------------------------------------------------------------
def interpolate_grid(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    resolution: int = 100,
    method: str = "cubic",
):
    """Interpolate data onto a finer grid for smooth contours."""
    X_fine = np.linspace(np.nanmin(X), np.nanmax(X), resolution)
    Y_fine = np.linspace(np.nanmin(Y), np.nanmax(Y), resolution)
    X_fine, Y_fine = np.meshgrid(X_fine, Y_fine)
    Z_fine = griddata(
        (X.flatten(), Y.flatten()), Z.flatten(), (X_fine, Y_fine), method=method
    )
    return X_fine, Y_fine, Z_fine


# -----------------------------------------------------------------------------
def highlight_region(
    ax, X, Y, Z, percent: float, levels: int = 10, cmap: str = "Blues"
):
    """Highlight top values in Z"""
    Zmin, Zmax = np.nanmin(Z), np.nanmax(Z)
    threshold = np.nanpercentile(Z, percent)

    if np.isscalar(levels):
        n = max(int(levels), 2)
        fill_levels = np.linspace(max(threshold, Zmin), Zmax, n)
    else:
        levels = np.asarray(levels)
        fill_levels = levels[levels >= threshold]
        if fill_levels.size < 2:
            # Ensure have at least two levels for contourf
            fill_levels = np.array([max(threshold, Zmin), Zmax])
    # Mask values below threshold
    Z_highlight = np.where(Z >= threshold, Z, Zmin - 1.0)
    return ax.contourf(X, Y, Z_highlight, levels=fill_levels, cmap=cmap)
