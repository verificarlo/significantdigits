#!/usr/bin/python3

import sys
import significantdigits.sigdigits as sigdigits
import significantdigits.io as sio
import argparse
import numpy as np


input_formats = ['stdin', 'npy']
output_formats = ['stdin', 'npy']


def check_args(args):

    args.metric = sigdigits._Metric_map[args.metric]
    args.method = sigdigits._Method_map[args.method]
    args.error = sigdigits._Error_map[args.error]

    if args.probability is not None:
        sigdigits.assert_is_probability(args.probability)
    else:
        args.probability = sigdigits.default_probability[args.metric]

    if args.confidence is not None:
        sigdigits.assert_is_confidence(args.probability)
    else:
        args.confidence = sigdigits.default_confidence[args.metric]


def parse_args():
    parser = argparse.ArgumentParser(description="Significant digits",
                                     prog="significantdigits")
    parser.add_argument('--metric',
                        choices=sigdigits._Metric_names,
                        help='Metric to compute')
    parser.add_argument('--method',
                        default=sigdigits.Method.CNH.name,
                        choices=sigdigits._Method_names,
                        help='Method to use')
    parser.add_argument('--error',
                        default=sigdigits.Error.Relative.name,
                        choices=sigdigits._Error_names,
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

    args = parser.parse_args()
    check_args(args)
    return args


def main(args):

    parser = sio.get_parser(args.input_format)()
    exporter = sio.get_exporter(args.output_format)(args.output)

    inputs = parser.parse(args.inputs)
    if args.reference:
        if args.reference == ["mean"]:
            reference = np.mean(inputs, axis=args.axis)
        else:
            reference = parser.parse(args.reference)
    else:
        reference = args.reference

    if args.metric == sigdigits.Metric.Significant:
        s = sigdigits.significant_digits(array=inputs,
                                         reference=reference,
                                         axis=args.axis,
                                         base=args.base,
                                         error=args.error,
                                         method=args.method,
                                         probability=args.probability,
                                         confidence=args.confidence)
    elif args.metric == sigdigits.Metric.Contributing:
        s = sigdigits.contributing_digits(array=inputs,
                                          reference=reference,
                                          axis=args.axis,
                                          base=args.base,
                                          error=args.error,
                                          method=args.method,
                                          probability=args.probability,
                                          confidence=args.confidence)
    else:
        print(f'Error: unkown metric {args.metric}')
        sys.exit(1)

    exporter.export(s)


if __name__ == '__main__':
    args = parse_args()
    main(args)
