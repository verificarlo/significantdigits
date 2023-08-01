import os
import shutil
import numpy as np

outputs_dir = "outputs_test"


class Singleton(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Setup(metaclass=Singleton):
    def __init__(self):
        self.rng = np.random.RandomState()
        self.outputs_dir = "outputs_test"
        self.prepare_outputs_dir()

    def prepare_outputs_dir(self):
        shutil.rmtree(self.outputs_dir, ignore_errors=True)
        os.mkdir(self.outputs_dir)

    def get_numpy_data_path(self, filename):
        filename = os.path.extsep.join((filename, "npy"))
        path = os.path.join(outputs_dir, filename)
        return os.path.realpath(path)

    def get_report_path(self, filename):
        filename = os.path.extsep.join((filename, "csv"))
        path = os.path.join(outputs_dir, filename)
        return os.path.realpath(path)
