import numpy as np
import cv2
from BaseImage import BaseImage, ColorModel
from GrayScale import GrayScaleTransform


class Thresholding(BaseImage):
    def threshold(self, value: int) -> BaseImage:
        if self.color_model != ColorModel.gray:
            img = GrayScaleTransform(data=self.data, model=ColorModel.rgb).to_gray()

        img.data[img.data > value] = 255
        img.data[img.data < value] = 0
        self.data = img.data
        self.color_model = img.color_model
        return self

    def otsu(self) -> BaseImage:
        if self.color_model != ColorModel.gray:
            img = GrayScaleTransform(data=self.data, model=ColorModel.rgb).to_gray()
            self.color_model = img.color_model
        else:
            img = self
        self.data = cv2.threshold(img.data, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.data = self.data[1]
        return self

    def th_adaptive(self, bS, c) -> BaseImage:
        if self.color_model != ColorModel.gray:
            img = GrayScaleTransform(data=self.data, model=ColorModel.rgb).to_gray()
            self.color_model = img.color_model
        else:
            img = self
        self.data = cv2.adaptiveThreshold(img.data, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=bS, C=c)
        return self


