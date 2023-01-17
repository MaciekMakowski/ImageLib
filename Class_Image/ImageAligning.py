import numpy as np
import cv2
from BaseImage import BaseImage, ColorModel
from GrayScale import GrayScaleTransform

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



    def align_image(self, tail_elimination) -> 'BaseImage':
        """
        metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        """
        if self.color_model == ColorModel.gray:
            self.data = self.align_layer(self.data)

        if self.color_model == ColorModel.rgb:
            r, g, b = self.get_layers()
            r_layer = self.align_layer(r)
            g_layer = self.align_layer(g)
            b_layer = self.align_layer(b)
            self.data = np.dstack((r_layer, g_layer, b_layer)).astype('uint8')

        if tail_elimination == True:
            self.data = np.quantile(self.data, 0.95)
            self.data = np.quantile(self.data, 0.05)
        return self

        pass

    def clahe(self) -> BaseImage:
        if self.color_model != ColorModel.gray:
            img = GrayScaleTransform(data=self.data, model=ColorModel.rgb).to_gray()
            self.data = img.data
            self.color_model = img.color_model

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.data = clahe.apply(self.data)
        return self
