from BaseImage import BaseImage, ColorModel
import cv2

from GrayScale import GrayScaleTransform


class EdgeDetection(BaseImage):

    def canny(self, th0: int, th1: int, kernel_size: int ) -> BaseImage:
        if self.color_model != ColorModel.gray:
            img = GrayScaleTransform(data=self.data, model=ColorModel.rgb).to_gray()
        edges = cv2.Canny(self.data, th0, th1, kernel_size)
        self.data = edges

        return self