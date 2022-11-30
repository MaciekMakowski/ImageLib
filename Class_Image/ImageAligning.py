import numpy as np

from BaseImage import _BaseImage, _ColorModel


class _ImageAligning(_BaseImage):
    """
    klasa odpowiadająca za wyrównywanie hostogramu
    """
    values: np.ndarray

    def __init__(self, data: np.ndarray = None, path: str = None, model = _ColorModel.gray) -> None:
        super().__init__(data, path, model)
        pass

    def align_image(self, tail_elimination: bool = True) -> '_BaseImage':
        """
        metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        """
        var_max = np.max(self.data)
        var_min = np.min(self.data)
        self.data = (self.data - var_min) * (255 / (var_max - var_min))
        return self
