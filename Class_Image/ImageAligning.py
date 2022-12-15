import numpy as np

from BaseImage import BaseImage, ColorModel


class ImageAligning(BaseImage):
    """
    klasa odpowiadająca za wyrównywanie hostogramu
    """
    values: np.ndarray

    def __init__(self, data: np.ndarray = None, path: str = None, model=None) -> None:
        super().__init__(data, path, model)
        pass

    def align_layer(self, layer) -> 'np.ndarray':
        values = np.zeros_like(layer)
        px_max = np.max(layer)
        px_min = np.min(layer)
        divider = px_max - px_min
        values = (layer - px_min) * (255 / divider)
        return values



    def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
        """
        metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        """
        if self.color_model == ColorModel.gray:
            self.data = self.align_layer(self.data)
            return self
        if self.color_model == ColorModel.rgb:
            r, g, b = self.get_layers()
            r_layer = self.align_layer(r)
            g_layer = self.align_layer(g)
            b_layer = self.align_layer(b)
            self.data = np.dstack((r_layer, g_layer, b_layer)).astype('uint8')
            return self
        pass
