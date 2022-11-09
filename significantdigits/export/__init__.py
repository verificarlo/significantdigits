import numpy as np

import significantdigits.export.numpy as io_numpy
import significantdigits.export.stdin as io_stdin

input_formats = dict(stdin=io_stdin.StdinParser,
                     npy=io_numpy.NumpyParser)

output_formats = dict(stdin=io_stdin.StdinExporter,
                      npy=io_numpy.NumpyExporter)

input_types = dict(binary16=np.float16,
                   binary32=np.float32,
                   binary64=np.float64)
