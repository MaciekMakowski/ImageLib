import matplotlib.pyplot as plt

from GrayScale import GrayScaleTransform, ColorModel, BaseImage, np
from ImageComparison import ImageComparison, ImageDiffMethod
from ImageAligning import ImageAligning
from ImageFiltration import ImageFiltration
from Filtres import Filter
from Histograms import Histogram
from Thresholding import Thresholding
from EdgeDetection import EdgeDetection
import cv2

class Image(GrayScaleTransform, ImageComparison, ImageAligning, ImageFiltration, Thresholding, EdgeDetection):
    """
    klasa stanowiaca glowny interfejs biblioteki
    w pozniejszym czasie bedzie dziedziczyla po kolejnych klasach
    realizujacych kolejne metody przetwarzania obrazow
    """

    def __init__(self, data: np.ndarray = None, path: str = None, model: ColorModel = None) -> None:
        super().__init__(data, path, model)
        pass

    def findAndColor(self, r, g, b):
        if self.color_model != ColorModel.gray:
            img = GrayScaleTransform(data=self.data, model=ColorModel.rgb).to_gray()
            self.color_model = img.color_model
        else:
            img = self
        theresh = Image(data=img.data, model=ColorModel.gray).th_adaptive(17,8)
        canny = Image(data=theresh.data, model=ColorModel.gray).canny(20, 50, 3)
        lines = cv2.HoughLinesP(canny.data, 2, np.pi / 180, 30)
        result_lines_img = cv2.cvtColor(img.data, cv2.COLOR_GRAY2RGB)
        for line in lines:
            x0, y0, x1, y1 = line[0]
            cv2.line(result_lines_img, (x0, y0), (x1, y1), (r, g, b), 8)

        self.data = result_lines_img
        return self