from significantdigits.io.generic import Exporter, Parser

import numpy as np


class NumpyParser(Parser):

    def parse(self, filename):
        return np.load(filename[0])


class NumpyExporter(Exporter):

    def __init__(self, filename):
        self.filename = filename

    def export(self, values):
        np.save(self.filename, values)
