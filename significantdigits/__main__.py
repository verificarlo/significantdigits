#!/usr/bin/python3

import sys

import numpy as np

import significantdigits.export as export
from significantdigits.sigdigits import *
import significantdigits.args as argparse


def main(args):
    parser = export.get_parser(args.input_format)()
    exporter = export.get_exporter(args.output_format)(args.output)

    inputs = parser.parse(args.inputs)
    if args.reference:
        reference = parser.parse(args.reference)
    else:
        reference = np.mean(inputs, axis=args.axis)

    if args.metric == Metric.Significant:
        s = significant_digits(array=inputs,
                               reference=reference,
                               axis=args.axis,
                               base=args.base,
                               error=args.error,
                               method=args.method,
                               probability=args.probability,
                               confidence=args.confidence)
    elif args.metric == Metric.Contributing:
        s = contributing_digits(array=inputs,
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
    args = argparse.parse_args()
    main(args)
