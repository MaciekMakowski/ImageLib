import numpy as np
from enum import Enum


class Filter(Enum):
    IDENTITY = default = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    SHARP = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    BLUR = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    GAUSS_BLUR = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

    def value(self) -> np.array:
        return np.array(self._value_)