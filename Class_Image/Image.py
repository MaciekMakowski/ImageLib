from GrayScale import GrayScaleTransform, ColorModel, BaseImage, np
from ImageComparison import ImageComparison, ImageDiffMethod
from ImageAligning import ImageAligning
from ImageFiltration import ImageFiltration
from Filtres import Filter
from Histograms import Histogram
class Image(GrayScaleTransform, ImageComparison, ImageAligning, ImageFiltration):
    """
    klasa stanowiaca glowny interfejs biblioteki
    w pozniejszym czasie bedzie dziedziczyla po kolejnych klasach
    realizujacych kolejne metody przetwarzania obrazow
    """
    def __init__(self, data: np.ndarray = None, path: str = None, model: ColorModel = None) -> None:
        super().__init__(data, path, model)
        pass