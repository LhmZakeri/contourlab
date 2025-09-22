# **ContourLab**

A basic Python tool for creating 2D and 3D contour plots from data.

ContourLab is an initial attempt at creating a Python library to generate contour plots from tabular data. This project is currently a work-in-progress, serving as a personal exercise, and may have missing features and bugs.

***

## **Features**

* **Basic Plotting:** Generate 2D contour plots from CSV data with complete control over plot settings. You can create subplots, offering features like **shared normalization** and **color maps**, or **individual settings** for each dataset. The plots can be either **filled** (colormap) or **line-based**, with the option to include **annotations**. You can also **highlight regions** that are higher than a certain value. For each plot, you can select three columns (**x**, **y**, and **z**) where the z-values will be aggregated as a **mean** for all data points.

* **3D Stacking:** This feature allows you to render multiple datasets as a series of 2D contour plots layered in 3D space, which is an excellent way to show how a variable changes across different conditions or slices.

    The plotting relies on a specific `matplotlib` module for rendering. It uses `Poly3DCollection` to project the filled contours onto a 3D space. Because of how this module renders graphics, it may produce **visual discontinuities** or **artifacts** when stacking the filled plots. These are known limitations of the `matplotlib` library's 3D capabilities and not a bug in the code itself.

    The `Contour3Dstacker` class in the `plotting.py` file handles this feature and has two main rendering modes:

    * **line mode:** Plots the contours as simple lines in 3D space. This mode is generally cleaner and avoids the rendering issues of filled plots.
    * **filled mode:** Renders the contours as solid, filled surfaces using `Poly3DCollection`.

    The CLI command for this feature is `contourlab --csv slice1.csv slice2.csv slice3.csv --x period --y wavelength --z intensity --mode 3d-stack`. You can also specify the rendering mode with `--render-mode filled` to activate the filled plot option.
