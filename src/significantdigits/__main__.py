#!/usr/bin/python3

import sys


from significantdigits import export
from significantdigits import significant_digits, contributing_digits, Metric
from significantdigits import args as argparse
from significantdigits import stats


def main():
    args = argparse.parse_args()
    Parser = export.input_formats[args.input_format]
    Exporter = export.output_formats[args.output_format]

    parser = Parser()
    exporter = Exporter(args.output)

    dtype = args.input_type

    inputs = parser.parse(args.inputs, dtype=dtype)

    if args.reference:
        reference = parser.parse(args.reference, dtype=dtype)
    else:
        reference = stats.mean(inputs, axis=args.axis, dtype=dtype)

    if args.metric == Metric.Significant:
        s = significant_digits(
            array=inputs,
            reference=reference,
            axis=args.axis,
            basis=args.basis,
            error=args.error,
            method=args.method,
            probability=args.probability,
            confidence=args.confidence,
            dtype=dtype,
        )
    elif args.metric == Metric.Contributing:
        s = contributing_digits(
            array=inputs,
            reference=reference,
            axis=args.axis,
            basis=args.basis,
            error=args.error,
            method=args.method,
            probability=args.probability,
            confidence=args.confidence,
            dtype=dtype,
        )
    else:
        print(f"Error: unkown metric {args.metric}")
        sys.exit(1)

    exporter.export(s)


if __name__ == "__main__":
    main()
