import numpy as np 
import pandas as pd

from contourlab.plotting_refactored_edit import MultiContourPlotter
from contourlab.plotting_refactored_edit import Contour3Dstacker
from contourlab.plotting_refactored_edit import ContourPlotter
from contourlab.plotting_refactored_edit import PlotConfig

if __name__ == "__main__":
    # -------------------------------------------------------------------
    #  Adjust config setting  
    # -------------------------------------------------------------------
    plot_config = PlotConfig(
        # --- Figure setting ---
        figsize= (6, 5), 
        dpi= 100,
        # --- Font settings ---
        font_axis_label = 12,
        font_tick = 10,
        font_annotation= 8,
        font_title= 14,
        font_colorbar = 10,
        # --- Contour settings ---
        levels = 10,            
        cmap = "Blues", 
        line_colors = "k",
        line_widths = 1.0,
        # --- Behavior flags ---
        interpolate= True,
        highlight = False,
        annotate= True,
        add_colorbar= False,
        contour_filled = False,
        # --- Highlighting ---
        percentile_threshold= 80.0,
        # --- 3D setting ---
        view_elevation = 22,
        view_azimuth = -60,
        z_gap_factor = 0.15,
        fill_alpha = 0.7,
    )

    datadirs = [
            "/home/elham/EikonalOptim/data/crn_table_sigma42.txt",
            "/home/elham/EikonalOptim/data/crn_table_sigma51.txt",
            "/home/elham/EikonalOptim/data/crn_table_sigma63.txt",
            "/home/elham/EikonalOptim/data/crn_table_sigma82.txt",
            "/home/elham/EikonalOptim/data/crn_table_sigma93.txt",
        ]
    
    dataset = []
    levelsset = []
    label_list = []
    for datadir in datadirs:
        data = pd.read_csv(datadir, sep="\s+")
        dataset.append(data)
        
        cp = ContourPlotter()
        _, _, Z= cp._prepare_data_grid(data, "aniso_phase", "wavelength", "simdur", config=plot_config)
        Zmax = np.nanmax(Z)

        levels = np.array(
            [
                Zmax * (0.97),
                Zmax * (0.975),
                Zmax * (0.98),
                Zmax * (0.985), 
                Zmax * (0.99),
                Zmax * (0.995),
                Zmax,
            ],
            dtype = float,            
        )
        levelsset.append(levels)
        colorbar_labels = {
            levels[6]: "Max",
            levels[5]: "Max-0.5%",
            levels[4]: "Max-1.0%",
            levels[3]: "Max-1.5%",
            levels[2]: "Max-2.0%",
            levels[1]: "Max-2.5%",
            levels[0]: "Max-3.0%",
        }
        label_list.append(colorbar_labels)
    
    mcp = MultiContourPlotter()
    results = mcp.plot_multiple_contours(
        datasets=dataset,
        x_col = "aniso_phase",
        y_col = "wavelength",
        z_col = "simdur",
        ncols = 3,
        interpolate=True, 

        shared_normalization= True,
        robust_normalization = False, # shared_normalization should be True also 
        adaptive_levels = False, 
        level_method = 'quantile', #'Log'/'linear'/'quantile',

        levels=levelsset,#levelsset / 5, 
        #levels = 10,
        #levels_step = 1,

        titles=["Sigma (4, 2)", "Sigma (5, 5)", "Sigma (6, 2)", "Sigma (8, 4)", "Sigma (9, 3)"],        
        #x_labels
        y_labels=["Mean of Success Rate", None, None, "Mean of Success Rate", None], 

        annotate=True, 
        highlight = False,
        percentile_threshold=60,        
        contour_filled = True, 

        colorbar_labels_set=label_list,  
        add_colorbar=True, 
        show=True,
        verbose = True,  
    )
    c3d = Contour3Dstacker(plot_config, verbose=True)
    c3d.stack_contours(
        results['results'],
        mode="filled",
        show=True,
        figsize=(12, 10),
        show_slice_boxes=True,
        shared_norm=results['shared_norm'],
        add_colorbar=True,
        line_colors='k',
        
    )

