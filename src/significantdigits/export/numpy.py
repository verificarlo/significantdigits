from significantdigits.export.generic import Exporter, Parser

import numpy as np


class NumpyParser(Parser):
    def parse(self, *args, **kwargs):
        filename = kwargs.get("filename", args[0])
        return np.load(filename, allow_pickle=True)


class NumpyExporter(Exporter):
    def __init__(self, *args, **kwargs):
        filename = kwargs.get("filename", args[0])
        self.filename = filename

    def export(self, *args, **kwargs):
        values = kwargs.get("values", args)
        np.save(self.filename, values)
