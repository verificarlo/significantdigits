from significantdigits.io.generic import Exporter, Parser

import numpy as np


class StdinParser(Parser):

    def parse(self, values):
        return np.array(values)


class StdinExporter(Exporter):

    def __init__(self, filename):
        self.filename = filename

    def export(self, values):
        print(values)
