from ColorModel import _ColorModel
from BaseImage import _BaseImage
from GrayScale import _GrayScaleTransform
from Image import *

file2 = _GrayScaleTransform('data/lena.jpg', _ColorModel.rgb)
file2.show_img()
file2.to_sepia((1.9, 0.1))
file2.show_img()
