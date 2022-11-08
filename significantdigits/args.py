import argparse
import sys

import significantdigits

input_formats = ['stdin', 'npy']
output_formats = ['stdin', 'npy']


def check_args(args):

    args.metric = significantdigits._Metric_map[args.metric]
    args.method = significantdigits._Method_map[args.method]
    args.error = significantdigits._Error_map[args.error]

    if args.probability is not None:
        significantdigits.assert_is_probability(args.probability)
    else:
        args.probability = significantdigits.default_probability[args.metric]

    if args.confidence is not None:
        significantdigits.assert_is_confidence(args.probability)
    else:
        args.confidence = significantdigits.default_confidence[args.metric]

    if args.input_format == 'stdin':
        try:
            args.inputs = list(map(float, args.inputs))
        except ValueError as e:
            print(f'Excepted numbers as inputs: {args.inputs}')
            print(e)
            sys.exit(2)

    if args.probability:
        significantdigits.assert_is_probability(args.probability)

    if args.confidence:
        significantdigits.assert_is_confidence(args.confidence)


def create_parser():
    parser = argparse.ArgumentParser(description="Significant digits",
                                     prog="significantdigits")
    parser.add_argument('--metric',
                        required=True,
                        choices=significantdigits._Metric_names,
                        help='Metric to compute')
    parser.add_argument('--method',
                        default=significantdigits.Method.CNH.name,
                        choices=significantdigits._Method_names,
                        help='Method to use')
    parser.add_argument('--error',
                        default=significantdigits.Error.Relative.name,
                        choices=significantdigits._Error_names,
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
    parser.add_argument('--input-format',
                        choices=input_formats,
                        required=True,
                        help='Input format')
    parser.add_argument('-i', '--inputs',
                        nargs='+',
                        required=True,
                        help='Inputs')
    parser.add_argument('-r', '--reference',
                        nargs='+',
                        help='Reference. Use "mean" to use the mean value of the input')
    parser.add_argument('--output-format',
                        default='stdin',
                        choices=output_formats,
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
    check_args(args)
    return args
