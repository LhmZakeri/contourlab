import numpy as np 
import pandas as pd

from contourlab import *
if __name__ == "__main__":
    # -------------------------------------------------------------------
    #  Adjust config setting  
    # -------------------------------------------------------------------
    plot_config = PlotConfig(
        # --- Figure setting ---
        figsize= (6, 5), 
        dpi= 200,
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
            "./data/crn_table_sigma42.txt",
            "./data/crn_table_sigma51.txt",
            "./data/crn_table_sigma63.txt",
            "./data/crn_table_sigma82.txt",
            "./data/crn_table_sigma93.txt",
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

        shared_normalization= False,
        robust_normalization = False, # shared_normalization should be True also 
        adaptive_levels = False, 
        level_method = 'quantile', #'Log'/'linear'/'quantile',

        levels=levelsset,#levelsset / 5, 
        #levels = 10,
        #levels_step = 1,

        titles=["Fig 1", "Fig 2", "Fig 3", "Fig 4", "Fig 5"],        
        #x_labels
        y_labels=[None, None, None, None, None], 

        annotate=True, 
        highlight = False,
        percentile_threshold=60,        
        contour_filled = True, 

        colorbar_labels_set=label_list,  
        add_colorbar=True, 
        show=True,
        verbose = True,  
        font_tick=14,
        font_axis_label=14,
        font_title=14,
        savepath = './images/Dataframe_customized_levels_subplot.png'
        
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
        font_tick=14,
        font_axis_label=14,
        font_title=14,
        savepath='./images/Dataframe_customized_levels_3DStacked.png'
        
    )

