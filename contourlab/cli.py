#!/usr/bin/env python3
"""
ContourLab CLI for 2D and 3D contour plots.

Enhanced version with better error handling and configuration options.
"""

import argparse
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import json

from contourlab.plotting import plot_contour, plot_multiple_contours, stack_contours_in_z


def load_dataframes(csv_files: List[str], separator: str = ",") -> List[pd.DataFrame]:
    """Load multiple CSV files with error handling."""
    dataframes = []
    
    for i, csv_file in enumerate(csv_files):
        try:
            if not Path(csv_file).exists():
                raise FileNotFoundError(f"File not found: {csv_file}")
            
            df = pd.read_csv(csv_file, sep=separator)
            
            if df.empty:
                print(f"Warning: File {csv_file} is empty", file=sys.stderr)
                continue
                
            dataframes.append(df)
            print(f"Loaded {csv_file}: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}", file=sys.stderr)
            sys.exit(1)
    
    if not dataframes:
        print("Error: No valid data files loaded", file=sys.stderr)
        sys.exit(1)
    
    return dataframes


def validate_columns(dataframes: List[pd.DataFrame], x_col: str, y_col: str, z_col: str):
    """Validate that required columns exist in all dataframes."""
    required_cols = [x_col, y_col, z_col]
    
    for i, df in enumerate(dataframes):
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error in file {i+1}: Missing columns {missing_cols}", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        
        # Check for sufficient data
        if len(df) < 4:
            print(f"Warning: File {i+1} has very few data points ({len(df)}). "
                 f"Contour quality may be poor.", file=sys.stderr)


def create_2d_plot(df: pd.DataFrame, args: argparse.Namespace) -> plt.Figure:
    """Create a single 2D contour plot."""
    print("Creating 2D contour plot...")
    
    res = plot_contour(
        df, 
        x_col=args.x, 
        y_col=args.y, 
        z_col=args.z,
        levels=args.levels, 
        interp=args.interp, 
        add_colorbar=True,
        title=args.title,
        xlabels=args.xlabel or args.x,
        ylabels=args.ylabel or args.y,
        cmap=args.colormap
    )
    
    return res["contour"].axes.figure


def create_multi_plot(dataframes: List[pd.DataFrame], args: argparse.Namespace) -> plt.Figure:
    """Create multiple 2D contour plots."""
    print(f"Creating {len(dataframes)} subplot(s) in {args.ncols} columns...")
    
    # Generate titles if not provided
    titles = None
    if args.titles:
        titles = args.titles.split(',')
    elif len(dataframes) > 1:
        titles = [f"Dataset {i+1}" for i in range(len(dataframes))]
    
    res = plot_multiple_contours(
        dataframes, 
        x_col=args.x, 
        y_col=args.y, 
        z_col=args.z,
        levels=args.levels, 
        ncols=args.ncols, 
        add_colorbar=True,
        titles=titles,
        share_norm=args.shared_colorscale
    )
    
    return res["fig"]


def create_3d_stack(dataframes: List[pd.DataFrame], args: argparse.Namespace) -> plt.Figure:
    """Create 3D stacked contour plot."""
    print(f"Creating 3D stack from {len(dataframes)} dataset(s)...")
    
    # Generate contour data for each dataframe
    contour_results = []
    for i, df in enumerate(dataframes):
        print(f"Processing dataset {i+1}/{len(dataframes)}...")
        res = plot_contour(
            df, 
            x_col=args.x, 
            y_col=args.y, 
            z_col=args.z,
            levels=args.levels, 
            interp=args.interp
        )
        contour_results.append(res["contour"])
    
    # Stack in 3D
    res = stack_contours_in_z(
        contour_results,
        mode=args.render_mode,
        elev=args.elevation,
        azim=args.azimuth
    )
    
    return res["fig"]


def save_or_show(figure: plt.Figure, output_path: Optional[str], dpi: int = 300):
    """Save figure to file or display it."""
    if output_path:
        try:
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"Figure saved to: {output_path}")
            
        except Exception as e:
            print(f"Error saving figure: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Displaying figure... (close window to exit)")
        plt.show()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ContourLab CLI for creating 2D and 3D contour plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single 2D plot
  contourlab --csv data.csv --x period --y wavelength --z intensity
  
  # Multiple subplots
  contourlab --csv data1.csv data2.csv --x period --y wavelength --z intensity --mode 2d-multiple
  
  # 3D stack
  contourlab --csv slice1.csv slice2.csv slice3.csv --x period --y wavelength --z intensity --mode 3d-stack
  
  # Save high-resolution output
  contourlab --csv data.csv --x period --y wavelength --z intensity --output plot.png --dpi 600
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--csv", type=str, nargs="+", required=True,
        help="CSV file(s) to plot"
    )
    parser.add_argument(
        "--x", type=str, required=True,
        help="Column name for X-axis"
    )
    parser.add_argument(
        "--y", type=str, required=True,
        help="Column name for Y-axis"
    )
    parser.add_argument(
        "--z", type=str, required=True,
        help="Column name for Z-axis (contour values)"
    )
    
    # Plot mode
    parser.add_argument(
        "--mode", type=str,
        choices=["2d", "2d-multiple", "3d-stack"],
        default="2d",
        help="Plot mode (default: 2d)"
    )
    
    # Plot configuration
    parser.add_argument(
        "--levels", type=int, default=10,
        help="Number of contour levels (default: 10)"
    )
    parser.add_argument(
        "--interp", action="store_true",
        help="Interpolate grid for smoother contours"
    )
    parser.add_argument(
        "--colormap", type=str, default="Blues",
        help="Colormap name (default: Blues)"
    )
    
    # Layout options
    parser.add_argument(
        "--ncols", type=int, default=2,
        help="Number of columns for multiple subplots (default: 2)"
    )
    parser.add_argument(
        "--shared-colorscale", action="store_true",
        help="Use shared color scale for multiple plots"
    )
    
    # Labels and titles
    parser.add_argument(
        "--title", type=str,
        help="Plot title"
    )
    parser.add_argument(
        "--xlabel", type=str,
        help="X-axis label (defaults to column name)"
    )
    parser.add_argument(
        "--ylabel", type=str,
        help="Y-axis label (defaults to column name)"
    )
    parser.add_argument(
        "--titles", type=str,
        help="Comma-separated subplot titles for multiple plots"
    )
    
    # 3D-specific options
    parser.add_argument(
        "--render-mode", type=str, choices=["line", "filled"], default="line",
        help="3D rendering mode (default: line)"
    )
    parser.add_argument(
        "--elevation", type=int, default=22,
        help="3D view elevation angle (default: 22)"
    )
    parser.add_argument(
        "--azimuth", type=int, default=-60,
        help="3D view azimuth angle (default: -60)"
    )
    
    # Input/Output options
    parser.add_argument(
        "--sep", type=str, default=",",
        help="CSV column separator (default: comma)"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output file path (if not specified, plot is displayed)"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output resolution in DPI (default: 300)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "2d" and len(args.csv) > 1:
        print("Warning: Multiple CSV files provided for 2D mode. Using first file only.", 
              file=sys.stderr)
    
    # Load and validate data
    try:
        dataframes = load_dataframes(args.csv, args.sep)
        validate_columns(dataframes, args.x, args.y, args.z)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    
    # Create plot based on mode
    try:
        if args.mode == "2d":
            figure = create_2d_plot(dataframes[0], args)
            
        elif args.mode == "2d-multiple":
            figure = create_multi_plot(dataframes, args)
            
        elif args.mode == "3d-stack":
            figure = create_3d_stack(dataframes, args)
        
        # Output result
        save_or_show(figure, args.output, args.dpi)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error creating plot: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
