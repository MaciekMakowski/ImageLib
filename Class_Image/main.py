from Image import _Image, _ColorModel, _ImageDiffMethod, _ImageComparison
from Histogram import _Histogram
file2 = _Image(path='data/lol.jpg', model=_ColorModel.rgb)
file1 = _Image(path='data/lol.jpg', model=_ColorModel.rgb)
file1.show_img()
file1.to_hsv()
file1.show_img()
file1.to_rgb()
file1.show_img()
print(file1.similarity(file2))
