import significantdigits.io.numpy as io_numpy
import significantdigits.io.stdin as io_stdin

_input_formats = dict(stdin=io_stdin.StdinParser,
                      npy=io_numpy.NumpyParser)

_output_formats = dict(stdin=io_stdin.StdinExporter,
                       npy=io_numpy.NumpyExporter)


def get_parser(input_format):
    return _input_formats[input_format]


def get_exporter(output_format):
    return _output_formats[output_format]
