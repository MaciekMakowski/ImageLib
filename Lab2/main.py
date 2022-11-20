from Image import _Image, _ColorModel, _ImageDiffMethod, _ImageComparison
from Histogram import _Histogram
file1 = _Image(path='data/lena3.jpg', model=_ColorModel.rgb)
file2 = _Image(path='data/lena3.jpg', model=_ColorModel.rgb)
file2.to_hsi()
print(file1.similarity(file2))

