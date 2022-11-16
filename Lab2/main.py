from BaseImage import _BaseImage, _ColorModel
from Histogram import _Histogram
file2 = _BaseImage('data/lena.jpg', _ColorModel.rgb)
file2.to_hsi()
file2.show_img()
hist = _Histogram(file2.data)
hist.plot()
file2.to_rgb()
file2.show_img()
