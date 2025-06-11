import argparse
import sys
import ast

import significantdigits as sd
from significantdigits._significantdigits import (
    _assert_is_confidence,
    _assert_is_probability,
    _default_confidence,
    _default_probability,
)
from significantdigits.export import input_formats, input_types, output_formats


def safe_eval(s):
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f"Error while parsing {s}")
        print(e)
        sys.exit(1)


def process_args(args):
    args.metric = sd.Metric.map[args.metric]
    args.method = sd.Method.map[args.method]
    args.error = sd.Error.map[args.error]
    args.input_type = input_types[args.input_type]

    if args.input_format == "stdin":
        args.inputs = safe_eval(args.inputs)
        if args.reference:
            args.reference = safe_eval(args.reference)

    if args.probability is not None:
        _assert_is_probability(args.probability)
    else:
        args.probability = _default_probability[args.metric]

    if args.confidence is not None:
        _assert_is_confidence(args.probability)
    else:
        args.confidence = _default_confidence[args.metric]

    if args.probability:
        _assert_is_probability(args.probability)

    if args.confidence:
        _assert_is_confidence(args.confidence)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Significant digits", prog="significantdigits"
    )
    parser.add_argument(
        "--metric",
        required=True,
        type=str.lower,
        choices=sd.Metric.names,
        help="Metric to compute",
    )
    parser.add_argument(
        "--method",
        type=str.lower,
        default=sd.Method.CNH.name,
        choices=sd.Method.names,
        help="Method to use",
    )
    parser.add_argument(
        "--error",
        type=str.lower,
        default=sd.Error.Relative.name,
        choices=sd.Error.names,
        help="Error to use",
    )
    parser.add_argument(
        "--probability", type=float, help="Probability for significance or contribution"
    )
    parser.add_argument("--confidence", type=float, help="Confidence level to use")
    parser.add_argument(
        "--axis", default=0, help="Axis along which samples are displayed"
    )
    parser.add_argument("--basis", default=2, type=int, help="Basis")
    parser.add_argument(
        "--input-type",
        choices=input_types.keys(),
        default="binary64",
        help="Input types",
    )
    parser.add_argument(
        "--input-format",
        choices=input_formats.keys(),
        required=True,
        help="Input format",
    )
    parser.add_argument("-i", "--inputs", required=True, help="Inputs")
    parser.add_argument(
        "-r",
        "--reference",
        help='Reference. Use "mean" to use the mean value of the input',
    )
    parser.add_argument(
        "--output-format",
        default="stdin",
        choices=output_formats.keys(),
        help="Output format",
    )
    parser.add_argument("-o", "--output", default="stdin", help="Output")
    return parser


def parse_args(args=None):
    parser = create_parser()
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    process_args(args)
    return args
