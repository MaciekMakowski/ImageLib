from Image import BaseImage, GrayScaleTransform, ColorModel
from ImageDiffMethod import ImageDiffMethod
from Histograms import Histogram
import numpy as np


class ImageComparison(BaseImage):
    """
    Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    """

    def histogram(self) -> Histogram:
        """
        metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        """
        return Histogram(self.data)
        pass

    def compare_to(self, other: BaseImage, method: ImageDiffMethod) -> float:
        """
        metoda zwracajaca mse lub rmse dla dwoch obrazow
        """
        result = 0
        if self.color_model != ColorModel.gray:
            Img1 = GrayScaleTransform(data=self.data, model=ColorModel.rgb).to_gray()
        else:
            Img1 = self.data
        hist1 = Histogram(Img1.data)
        if other.color_model != ColorModel.gray:
            Img2 = GrayScaleTransform(data=other.data, model=other.color_model).to_gray()
        else:
            Img2 = other.data
        hist2 = Histogram(Img2.data)

        if len(hist2.values) == len(hist1.values):
            n = len(hist2.values)
            result_arr = np.power(hist1.values - hist2.values, 2)
            result = np.sum(result_arr/n)
            if method == ImageDiffMethod.rmse:
                result = np.sqrt(result)
            return result
        print("Obrazy muszą mieć takie same wymiary")
        return 0

    def similarity(self, other: BaseImage) -> float:
        result = self.compare_to(other, ImageDiffMethod.rmse)
        if 100 - round(result / 100, 2) < 0:
            return 0
        else:
            return 100 - round(result / 100, 2)

        pass