import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from contourlab.plotting import *

def create_example_datasets(num_datasets=3, num_points=50, noise_level=0.1):
    """Generates a list of DataFrames with varying Gaussian distributions."""
    datasets = []
    centers = [(0, 0), (2, -2), (-3, 3)]
    for i in range(num_datasets):
        x = np.linspace(-5, 5, num_points)
        y = np.linspace(-5, 5, num_points)
        X, Y = np.meshgrid(x, y)
        
        # Shift the center of the Gaussian for each dataset
        dx, dy = centers[i]
        Z = np.exp(-((X - dx)**2 + (Y - dy)**2) / 5) + noise_level * np.random.randn(*X.shape)
        
        df = pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "z": Z.ravel()})
        datasets.append(df)
    return datasets

config = PlotConfig(dpi=200)
if __name__ == "__main__":
    print("--- Running Multi-Contour Plotting Example ---")
    datasets = create_example_datasets()
    mcp = MultiContourPlotter(config)
    
    mcp.plot_multiple_contours(
        datasets=datasets,
        x_col="x",
        y_col="y",
        z_col="z",
        titles=["Slice 1 (center 0,0)", "Slice 2 (center 2,-2)", "Slice 3 (center -3,3)"],
        shared_normalization=True,  
        ncols=2,
        cmap="viridis",
        annotate=True,
        #levels=15,
        add_colorbar=True,
        verbose=False,
        contour_filled=True,
        #adaptive_levels=False,
        font_axis_label=16,
        font_title= 16, 
        font_tick=16,
        savepath='./images/Gaussain_Shared_colorbar_normalization_Automatic_levels_Adaptive.png',
        show=False,
    )
    


