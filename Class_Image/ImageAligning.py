import numpy as np
import cv2
from BaseImage import BaseImage, ColorModel
import matplotlib.pyplot as plt

class ImageAligning(BaseImage):
    """
    klasa odpowiadająca za wyrównywanie hostogramu
    """
    values: np.ndarray

    def __init__(self, data: np.ndarray = None, path: str = None, model=None) -> None:
        super().__init__(data, path, model)
        pass

    def align_layer(self, layer, tail_elimination) -> 'np.ndarray':
        values = np.zeros_like(layer)
        px_max = np.max(layer)
        px_min = np.min(layer)
        layer = layer.astype(np.float64)
        if tail_elimination == True:
            px_max = np.quantile(layer, 0.95)
            px_min = np.quantile(layer, 0.05)
        divider = px_max - px_min
        values = ((layer - px_min) / divider) * 255
        values[values < 0] = 0
        values[values > 255] = 255
        return values



    def align_image(self, tail_elimination) -> 'BaseImage':
        """
        metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        """
        if self.color_model == ColorModel.gray:
            self.data = self.align_layer(self.data)

        if self.color_model == ColorModel.rgb:
            r, g, b = self.get_layers()
            r_layer = self.align_layer(r, tail_elimination)
            g_layer = self.align_layer(g, tail_elimination)
            b_layer = self.align_layer(b, tail_elimination)

            self.data = np.dstack((r_layer, g_layer, b_layer)).astype('uint8')
        return self

        pass

    def clahe(self) -> BaseImage:
        if self.color_model == ColorModel.gray:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            self.data = clahe.apply(self.data)
        else:
            imgLab = cv2.cvtColor(self.data, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            imgLab[..., 0] = clahe.apply(imgLab[...,0])

            self.data = cv2.cvtColor(imgLab, cv2.COLOR_LAB2BGR)
        return self

    def show_clahe_plot(self):

        if self.color_model == ColorModel.gray:
            plt.subplot(221)
            plt.imshow(self.data, cmap='gray')
            plt.subplot(222)
            plt.hist(self.data.ravel(), bins=256, range=(0, 256), color='gray')
            plt.subplot(223)
            plt.imshow(self.data, cmap='gray')
            plt.subplot(224)
            plt.hist(self.clahe().data.ravel(), bins=256, range=(0, 256), color='gray')

            plt.show()

        else:
            plt.subplot(221)
            plt.imshow(self.data)
            plt.subplot(222)
            plt.hist(self.data[..., 0].ravel(), bins=256, range=(0, 256), color='b')
            plt.hist(self.data[..., 1].ravel(), bins=256, range=(0, 256), color='g')
            plt.hist(self.data[..., 2].ravel(), bins=256, range=(0, 256), color='r')
            plt.subplot(223)
            plt.imshow(self.clahe().data)
            plt.subplot(224)
            plt.hist(self.clahe().data[..., 0].ravel(), bins=256, range=(0, 256), color='b')
            plt.hist(self.clahe().data[..., 1].ravel(), bins=256, range=(0, 256), color='g')
            plt.hist(self.clahe().data[..., 2].ravel(), bins=256, range=(0, 256), color='r')
            plt.show()
