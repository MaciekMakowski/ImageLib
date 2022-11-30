from Image import _Image, _ColorModel, _ImageDiffMethod, _ImageComparison
from ImageAligning import _ImageAligning
from Histogram import _Histogram
file1 = _Image(path='data/lena.jpg', model=_ColorModel.rgb)
file1.to_gray()
hist1 = _Histogram(file1.data)
file2_alg = _ImageAligning(data=file1.data)
file2_alg.align_image()
hist2 = _Histogram(file2_alg.data)
hist1.plot()
hist2.plot()
