import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from contourlab.plotting import (
    plot_contour,
    plot_multiple_contours,
    stack_contours_in_z,
)

# -----------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "x": np.repeat(np.arange(5), 5),
            "y": np.tile(np.arange(5), 5),
            "z": np.arange(25).astype(float),
        }
    )


# -----------------------------------------------------------------------------
def test_plot_contour_basic(sample_df):
    res = plot_contour(sample_df, "x", "y", "z", interp=False, highlight=False)
    # Objext exist
    assert "contour" in res
    assert "filled" in res
    assert "colorbar" in res
    # contour lines actually have collections
    assert len(res["contour"].collections) > 0
    vmin, vmax = res["filled"].get_array().min(), res["filled"].get_array().max()
    assert vmin <= vmax


# -----------------------------------------------------------------------------


def test_plot_contour_full_coverage(sample_df):
    fig, ax = plt.subplots()
    res = plot_contour(
        sample_df, x_col="x", y_col="y", z_col="z", interp=False, levels=6, ax=ax
    )

    filled = res["filled"]
    levels = filled.levels

    # Extract Z values (drop NaNs)
    z_values = sample_df["z"].values
    z_values = z_values[~np.isnan(z_values)]

    assert z_values.min() <= levels[0]
    assert z_values.max() >= levels[-1]

    assert not np.isnan(sample_df["z"].values).any()


# -----------------------------------------------------------------------------
def test_plot_contour_without_interp(sample_df):
    res = plot_contour(
        sample_df, x_col="x", y_col="y", z_col="z", interp=False, highlight=False
    )
    assert res["filled"] is not None


# -----------------------------------------------------------------------------
def test_plot_mutiple_contours_returns(sample_df):
    # create 2 dataframe with slightly shifted z
    df2 = sample_df.copy()
    df2["z"] += 10
    res = plot_multiple_contours(
        [sample_df, df2], x_col="x", y_col="y", z_col="z", share_norm=True
    )
    assert "fig" in res and "axes" in res and "results" in res
    assert len(res["results"]) == 2
    assert res["colorbar"] is not None
    assert isinstance(res["axes"], np.ndarray)


# -----------------------------------------------------------------------------
def test_plot_multiple_contours_norm_consistency(sample_df):
    df2 = sample_df.copy()
    df2["z"] *= 2
    res = plot_multiple_contours(
        [sample_df, df2], x_col="x", y_col="y", z_col="z", share_norm=True
    )
    vmins = []
    vmaxs = []
    for r in res["results"]:
        if r["filled"] is not None:
            norm = r["filled"].norm
            vmins.append(norm.vmin)
            vmaxs.append(norm.vmax)
    assert len(set(vmins)) == 1
    assert len(set(vmaxs)) == 1


# -----------------------------------------------------------------------------
def test_plot_multiple_contours_independent_norm(sample_df):
    df2 = sample_df.copy()
    df2["z"] *= 10

    res = plot_multiple_contours(
        [sample_df, df2], x_col="x", y_col="y", z_col="z", share_norm=False
    )

    vmins, vmaxs = [], []
    for r in res["results"]:
        if r["filled"] is not None:
            norm = r["filled"].norm
            vmins.append(norm.vmin)
            vmaxs.append(norm.vmax)

    # At least one bound should differ (since ranges are different)
    assert not (len(set(vmins)) == 1 and len(set(vmaxs)) == 1)


# -----------------------------------------------------------------------------
def test_stack_contours_default_offsets(sample_df):
    # build two contour sets with different z to ensure multiple slices
    res1 = plot_contour(
        sample_df, x_col="x", y_col="y", z_col="z", annotate=False, highlight=False
    )
    df2 = sample_df.copy()
    df2["z"] *= 2
    res2 = plot_contour(
        df2, x_col="x", y_col="y", z_col="z", annotate=False, highlight=False
    )

    out = stack_contours_in_z([res1["contour"], res2["contour"]])
    offs = out["z_offsets"]
    assert len(offs) == 2
    assert offs[1] > offs[0]


# -----------------------------------------------------------------------------


def test_stack_contours_custom_offsets(sample_df):
    res = plot_contour(
        sample_df, x_col="x", y_col="y", z_col="z", annotate=False, highlight=False
    )
    out = stack_contours_in_z([res["contour"], res["contour"]], z_offsets=[0.0, 3.5])
    assert out["z_offsets"] == [0.0, 3.5]
