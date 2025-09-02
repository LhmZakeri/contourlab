import argparse
import pandas as pd
import sys
import matplotlib.pyplot as plt
from .plotting import plot_contour, plot_multiple_contours, stack_contours_in_z


def main():
    """
    ContourLab CLI:
    - Single 2D cotour : plot_contour
    - Multiple 2D contours: plot_multiple_contours
    - 3D stacked contours: stack_contours_in_z
    """
    parser = argparse.ArgumentParser(
        description="ContourLab CLI for 2D and 3D contour plots"
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to CSV files(s). Multiple files allowed.",
    )
    parser.add_argument("--x", type=str, required=True, help="Column name for x")
    parser.add_argument("--y", type=str, required=True, help="Column name for Y")
    parser.add_argument("--z", type=str, required=True, help="Column name for Z")
    # --- Mode selection ---------------------------------------------------
    parser.add_argument(
        "--mode",
        type=str,
        choices=["2d", "2d-multiple", "3d-stack"],
        default="2d",
        help="Plot mode: single 2D, multiple 2D (default:2d), or 3D stacked contours",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save figure")
    # --- Optional settings ------------------------------------------------
    parser.add_argument(
        "--levels", type=int, default=10, help="Number of contour levels"
    )
    parser.add_argument(
        "--interp", action="store_true", help="Interpolate grid for smooth contours"
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="Number of subplot columns (for 2d-multiple)",
    )
    parser.add_argument(
        "--output", help="Save figure to file instead of showing interactively"
    )

    args = parser.parse_args()

    # --- Load CSV(s) ------------------------------------------------------
    try:
        dfs = [pd.read_csv(f) for f in args.csv]
    except Exception as e:
        print(f"Error reading CSV files(s): {e}")
        sys.exit(1)

    # --- Run mode ---------------------------------------------------------
    if args.mode == "2d":
        res = plot_contour(
            dfs[0],
            x_col=args.x,
            y_col=args.y,
            z_col=args.z,
            levels=args.levels,
            interp=args.interp,
            add_colorbar=True,
        )
        fig = res["contour"].axes.figure
     elif args.mode == "2d-multiple":
        res = plot_multiple_contours(
            dfs, 
            x_col=args.x,
            y_col=args.y,
            z_cols=args.z,
            levels=args.levels,
            ncols=args.ncols,
            add_colorbar=True,
        )    
        fig = res["fig"]
      elif args.mode=="3d-stack":
        contour_results = [
            plot_contour(
                df, 
                x_col=args.x, 
                y_col=args.y, 
                z_col= args.z,
                levels=args.levels,
                interp=args.interp,
            )["contour"]
            for df in dfs
        ]
        res = stack_contours_in_z(contour_results)
        fig = res["fig"]

     # --- Save/show --------------------------------------------------------   
     if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
     else:
        plt.show()
# ===========================================================================
if __name__ == "__main__":
    main()