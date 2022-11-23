from Image import _BaseImage, _GrayScaleTransform, _ColorModel
from ImageDiffMethod import _ImageDiffMethod
from Histogram import _Histogram
import numpy as np


class _ImageComparison(_BaseImage):
    """
    Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    """

    def histogram(self) -> _Histogram:
        """
        metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        """
        return _Histogram(self.data)
        pass

    def compare_to(self, other: _BaseImage, method: _ImageDiffMethod) -> float:
        """
        metoda zwracajaca mse lub rmse dla dwoch obrazow
        """
        result = 0
        Img1 = _GrayScaleTransform(data=self.data,model=_ColorModel.rgb).to_gray()
        Img2 = _GrayScaleTransform(data=other.data, model=other.color_model).to_gray()
        hist1 = _Histogram(Img1.data)
        hist2 = _Histogram(Img2.data)
        if len(hist2.values) == len(hist1.values):
            n = len(hist2.values)
            result_arr = np.power(hist1.values - hist2.values, 2)
            result = np.sum(result_arr/n)
            if method == _ImageDiffMethod.rmse:
                result = np.sqrt(result)
            return result
        print("Obrazy muszą mieć takie same wymiary")
        return 0

    def similarity(self, other: _BaseImage) -> float:
        result = self.compare_to(other, _ImageDiffMethod.rmse)
        return abs(100 - round(result / 100, 2))

        pass