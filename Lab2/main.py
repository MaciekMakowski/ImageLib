from Image import _GrayScaleTransform, _ColorModel

file2 = _GrayScaleTransform('data/lena.jpg', _ColorModel.rgb)
file2.show_img()
file2.to_sepia((1.1, 0.9))
file2.show_img()
