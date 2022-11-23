from Image import _Image, _ColorModel, _ImageDiffMethod, _ImageComparison
from Histogram import _Histogram
file2 = _Image(path='data/as.jpg', model=_ColorModel.rgb)
file1 = _Image(path='data/as.jpg', model=_ColorModel.rgb)
file2.to_hsv()
file2.to_rgb()
file2.show_img()
print(file2.similarity(file1))

