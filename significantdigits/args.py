import argparse
import sys
import ast

import significantdigits
from significantdigits.export import input_formats, input_types, output_formats


def safe_eval(s):
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f'Error while parsing {s}')
        print(e)
        sys.exit(1)


def process_args(args):

    args.metric = significantdigits.Metric.map[args.metric]
    args.method = significantdigits.Method.map[args.method]
    args.error = significantdigits.Error.map[args.error]
    args.input_type = input_types[args.input_type]

    if args.input_format == 'stdin':
        args.inputs = safe_eval(args.inputs)
        if args.reference:
            args.reference = safe_eval(args.reference)

    if args.probability is not None:
        significantdigits.assert_is_probability(args.probability)
    else:
        args.probability = significantdigits.default_probability[args.metric]

    if args.confidence is not None:
        significantdigits.assert_is_confidence(args.probability)
    else:
        args.confidence = significantdigits.default_confidence[args.metric]

    if args.probability:
        significantdigits.assert_is_probability(args.probability)

    if args.confidence:
        significantdigits.assert_is_confidence(args.confidence)


def create_parser():
    parser = argparse.ArgumentParser(description="Significant digits",
                                     prog="significantdigits")
    parser.add_argument('--metric',
                        required=True,
                        type=str.lower,
                        choices=significantdigits.Metric.names,
                        help='Metric to compute')
    parser.add_argument('--method',
                        type=str.lower,
                        default=significantdigits.Method.CNH.name,
                        choices=significantdigits.Method.names,
                        help='Method to use')
    parser.add_argument('--error',
                        type=str.lower,
                        default=significantdigits.Error.Relative.name,
                        choices=significantdigits.Error.names,
                        help='Error to use')
    parser.add_argument('--probability',
                        type=float,
                        help='Probability for significance or contribution')
    parser.add_argument('--confidence',
                        type=float,
                        help='Confidence level to use')
    parser.add_argument('--axis',
                        default=0,
                        help='Axis along which samples are displayed')
    parser.add_argument('--base',
                        default=2,
                        type=int,
                        help='Base')
    parser.add_argument('--input-type',
                        choices=input_types.keys(),
                        default='binary64',
                        help='Input types')
    parser.add_argument('--input-format',
                        choices=input_formats.keys(),
                        required=True,
                        help='Input format')
    parser.add_argument('-i', '--inputs',
                        required=True,
                        help='Inputs')
    parser.add_argument('-r', '--reference',
                        help='Reference. Use "mean" to use the mean value of the input')
    parser.add_argument('--output-format',
                        default='stdin',
                        choices=output_formats.keys(),
                        help='Output format')
    parser.add_argument('-o', '--output',
                        default='stdin',
                        help='Output')
    return parser


def parse_args(args=None):
    parser = create_parser()
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    process_args(args)
    return args
