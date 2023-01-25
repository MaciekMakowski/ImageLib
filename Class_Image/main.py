from Image import Image, ColorModel, ImageDiffMethod, ImageComparison,np, ImageAligning, Histogram, Filter
import cv2
import matplotlib.pyplot as plt
file1 = Image(path='data/lena.jpg', model=ColorModel.rgb)
file1.conv_2d(kernel=Filter.GAUSS_BLUR.value(), prefix=1/16)
file1.show_img()
file1.clahe()
file1.show_img()
# file2 = file1.conv_2d(file1, Filter.W45.value())

# Do widgeta w Jupyterze
# from ipywidgets import interact
#
# def test_print(first: int, second: str) -> None:
#   print(f'First param: {first}, second param: {second}')
#
# interact(test_print, first=range(1, 11), second=['one', 'two'])

