from significantdigits.export.generic import Exporter, Parser

import numpy as np


class StdinParser(Parser):
    def parse(self, *args, **kwargs):
        values = kwargs.get("values", *args)
        dtype = kwargs.get("dtype", np.float64)
        return np.asarray(values, dtype=dtype)


class StdinExporter(Exporter):
    def __init__(self, *args, **kwargs):
        filename = kwargs.get("filename", args[0])
        self.filename = filename

    def export(self, *args, **kwargs):
        values = kwargs.get("values", args)
        print(values)
