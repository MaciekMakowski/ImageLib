from BaseImage import BaseImage, ColorModel
import matplotlib.pyplot as plt
import cv2

from GrayScale import GrayScaleTransform


class EdgeDetection(BaseImage):

    def canny(self, th0: int, th1: int, kernel_size: int ) -> BaseImage:
        if self.color_model != ColorModel.gray:
            img = GrayScaleTransform(data=self.data, model=ColorModel.rgb).to_gray()
            self.color_model = img.color_model
        else:
            img = self
        edges = cv2.Canny(img.data, th0, th1, kernel_size)
        self.data = edges

        return self

    def find_circles(self):
        img_gray = GrayScaleTransform(data=self.data, model=ColorModel.rgb).to_gray()
        circles = cv2.HoughCircles(img_gray.data, method=cv2.HOUGH_GRADIENT, dp=2, minDist=60, minRadius=20, maxRadius=100)

        for (x, y, r) in circles.astype(int)[0]:
            cv2.circle(self.data, (x, y), r, (0, 255, 0), 4)

        plt.imshow(self.data)
        plt.show()
