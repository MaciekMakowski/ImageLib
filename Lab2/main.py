from BaseImage import _BaseImage, _ColorModel
from Histogram import _Histogram
from GrayScale import _GrayScaleTransform
file1 = _BaseImage(path='data/lena3.jpg', model=_ColorModel.rgb)
file2 = _BaseImage(data=file1.data, model=_ColorModel.rgb)
file3 = _GrayScaleTransform(data=file1.data, model=_ColorModel.rgb)
file2.show_img()
file2.to_hsl()
file2.show_img()
file2.to_rgb()
file2.show_img()
