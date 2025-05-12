import argparse
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import significantdigits as sd


def analyze_significant_digits(
    basis=10,
    axis=0,
    magnitude=np.array([-3, 4], dtype=np.float64),
    shift_range=np.arange(-4, 4, dtype=np.float64),
    sample_size=10000,
    confidence=0.95,
    probability=0.95,
    dtype=np.float64,
    verbose=True,
    method=sd.Method.CNH,
    error=sd.Error.Absolute,
    **kwargs,
):
    """
    Analyze significant digits using specified method and error type.

    Parameters:
    -----------
    basis : int, default=10
        The numerical base to use
    axis : int, default=0
        The axis along which to compute
    magnitude : numpy.ndarray, default=np.array([-3, 4], dtype=np.float64)
        The magnitude array for calculations
    shift_range : numpy.ndarray, default=np.arange(-4, 4, dtype=np.float64)
        Range of shift values to analyze
    sample_size : int, default=10000
        Size of random samples to generate
    confidence : float, default=0.95
        Confidence level for calculations
    probability : float, default=0.95
        Probability level for calculations
    dtype : numpy.dtype, default=np.float64
        Data type for calculations
    verbose : bool, default=True
        Whether to print detailed output
    method : sd.Method, default=sd.Method.CNH
        Method to use (CNH or General)
    error : sd.Error, default=sd.Error.Absolute
        Error type to use (Absolute or Relative)

    Returns:
    --------
    dict
        Dictionary containing the results of the analysis
    """
    # Initialize result lists
    shifts = []
    stds = []
    means = []
    digits_values = []

    for shift in shift_range:
        mean = np.power(basis, magnitude)
        std = np.power(basis, magnitude - shift)
        array = np.random.normal(
            loc=mean, scale=std, size=(sample_size, len(magnitude))
        )

        stds.append(std)
        means.append(mean)
        shifts.append(shift)

        if verbose:
            print("\nShift: ", shift)
            print("Type", array.dtype)
            print(array)
            print("-" * 20)
            print("Theoretical mean", mean)
            print("Empirical mean  ", array.mean(axis=0))
            print("Mean diff       ", mean - array.mean(axis=0))
            print("-" * 20)
            print("Theoretical std", std)
            print("Empirical std  ", array.std(axis=0))
            print("Std diff       ", std - array.std(axis=0))
            print("-" * 20)

        digits = sd.significant_digits(
            array,
            reference=mean,
            error=error,
            method=method,
            axis=axis,
            confidence=confidence,
            probability=probability,
            dtype=dtype,
            basis=basis,
        )

        digits_values.append(digits)

        if verbose:
            method_name = method.name
            error_name = error.name
            print(f"{method_name} {error_name}: ", digits)

    # Store results with appropriate key name
    result_key = ""
    if method == sd.Method.CNH:
        result_key = (
            "s_abs_cnh_values" if error == sd.Error.Absolute else "s_rel_cnh_values"
        )
    else:
        result_key = "s_abs_values" if error == sd.Error.Absolute else "s_rel_values"

    results = {
        "shifts": shifts,
        "means": means,
        "stds": stds,
        result_key: digits_values,
    }

    return results


def plot(args, results, result_key):
    method_name = args["method"].name
    error_name = args["error"].name

    # Create figure
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[
            f"Significant Digits ({method_name} method, {error_name} error)"
        ],
    )

    # For each dimension in the data
    for dim in range(len(args["magnitude"])):
        # Extract data for this dimension
        shifts = results["shifts"]
        digits = [d[dim] for d in results[result_key]]

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=shifts,
                y=digits,
                mode="lines+markers",
                name=f'Magnitude {args["magnitude"][dim]}',
            ),
            row=1,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title=f"Significant Digits Analysis - Basis: {args['basis']}",
        xaxis_title="Shift",
        yaxis_title="Significant Digits",
        legend_title="Dimensions",
        width=900,
        height=600,
    )

    # Show plot or save to file
    if args.get("output") and args["output"].endswith((".png", ".jpg", ".svg", ".pdf")):
        fig.write_image(args["output"])
        print(f"Plot saved to {args['output']}")
    else:
        fig.show()


def dump(args, results, result_key):
    output_path = args["output"]

    # Skip if we already wrote the plot to this path
    if args.get("plot") and output_path.endswith((".png", ".jpg", ".svg", ".pdf")):
        pass
    else:
        # Convert numpy arrays to lists for serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, list) and all(
                isinstance(item, np.ndarray) for item in value
            ):
                serializable_results[key] = [item.tolist() for item in value]
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value

        # Add metadata
        serializable_results["metadata"] = {
            "method": args["method"].__name__,
            "error": args["error"].__name__,
            "basis": args["basis"],
            "confidence": args["confidence"],
            "probability": args["probability"],
            "sample_size": args["sample_size"],
        }

        # Determine file type and save accordingly
        if output_path.endswith(".json"):
            with open(output_path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Results saved to {output_path}")
        elif output_path.endswith(".csv"):
            # Create a DataFrame from the results
            data = {"shift": results["shifts"]}

            # Add columns for each dimension
            for dim in range(len(args["magnitude"])):
                col_name = f"mag_{args['magnitude'][dim]}_digits"
                data[col_name] = [d[dim] for d in results[result_key]]

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        else:
            # Default to JSON
            with open(output_path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Results saved to {output_path}")


def parse_args():
    """Parse command-line arguments for significant digits analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze significant digits with different methods and error types"
    )

    # Basic parameters
    parser.add_argument(
        "--basis", type=int, default=10, help="Numerical base (default: 10)"
    )
    parser.add_argument(
        "--axis", type=int, default=0, help="Computation axis (default: 0)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Size of random samples (default: 10000)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=0.95,
        help="Probability level (default: 0.95)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")

    # Method and error type
    parser.add_argument(
        "--method",
        type=str,
        choices=["CNH", "General"],
        default="CNH",
        help="Method to use (default: CNH)",
    )
    parser.add_argument(
        "--error",
        type=str,
        choices=["Absolute", "Relative"],
        default="Absolute",
        help="Error type (default: Absolute)",
    )

    # Data type
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="Data type (default: float64)",
    )

    # Magnitude array
    parser.add_argument(
        "--magnitude",
        type=str,
        default="-3,4",
        help='Comma-separated magnitude values (default: "-3,4")',
    )

    # Shift range
    parser.add_argument(
        "--shift-start",
        type=float,
        default=-4.0,
        help="Start of shift range (default: -4.0)",
    )
    parser.add_argument(
        "--shift-end", type=float, default=4.0, help="End of shift range (default: 4.0)"
    )
    parser.add_argument(
        "--shift-step",
        type=float,
        default=1.0,
        help="Step size for shift range (default: 1.0)",
    )

    # Output options
    parser.add_argument(
        "--output", type=str, help="Output file for results (default: no file output)"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate and display plots"
    )

    args = parser.parse_args()

    # Process arguments that need conversion
    processed_args = vars(args).copy()

    # Convert method string to enum
    if args.method == "CNH":
        processed_args["method"] = sd.Method.CNH
    else:
        processed_args["method"] = sd.Method.General

    # Convert error string to enum
    if args.error == "Absolute":
        processed_args["error"] = sd.Error.Absolute
    else:
        processed_args["error"] = sd.Error.Relative

    # Convert dtype string to numpy dtype
    processed_args["dtype"] = np.float32 if args.dtype == "float32" else np.float64

    # Convert magnitude to numpy array
    magnitude_values = [float(x) for x in args.magnitude.split(",")]
    processed_args["magnitude"] = np.array(
        magnitude_values, dtype=processed_args["dtype"]
    )

    # Create shift range
    processed_args["shift_range"] = np.arange(
        args.shift_start,
        args.shift_end + args.shift_step / 2,  # Add half step to ensure end is included
        args.shift_step,
        dtype=processed_args["dtype"],
    )

    # Remove original args that were converted
    for key in ["shift_start", "shift_end", "shift_step"]:
        processed_args.pop(key, None)

    return processed_args


def main():
    """
    Main function to run significant digits analysis from command line arguments.

    Assumes parse_args and analyze_significant_digits functions are defined elsewhere.
    """
    # Parse command line arguments
    args = parse_args()

    print(
        f"Running analysis with {args['method'].name} method and {args['error'].name} error"
    )
    print(f"Shift range: {args['shift_range'][0]} to {args['shift_range'][-1]}")

    # Run the analysis
    results = analyze_significant_digits(**args)

    # Determine which result key contains our data
    result_key = ""
    if args["method"] == sd.Method.CNH:
        result_key = (
            "s_abs_cnh_values"
            if args["error"] == sd.Error.Absolute
            else "s_rel_cnh_values"
        )
    else:
        result_key = (
            "s_abs_values" if args["error"] == sd.Error.Absolute else "s_rel_values"
        )

    # Generate plot if requested
    if args.get("plot"):
        plot(args, results, result_key)

    # Save results if output file specified
    if args.get("output"):
        dump(args, results, result_key)

    return results


if __name__ == "__main__":
    main()
