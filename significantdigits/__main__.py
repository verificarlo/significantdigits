#!/usr/bin/python3

import sys


from significantdigits import export
from significantdigits import (
    significant_digits, contributing_digits, Metric)
from significantdigits import args as argparse
from significantdigits import stats


def main():
    args = argparse.parse_args()
    parser = export.get_parser(args.input_format)()
    exporter = export.get_exporter(args.output_format)(args.output)

    inputs = parser.parse(args.inputs)
    if args.reference:
        reference = parser.parse(args.reference)
    else:
        reference = stats.mean(inputs, axis=args.axis)

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
    main()
