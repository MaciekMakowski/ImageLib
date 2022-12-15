from Image import Image, ColorModel, ImageDiffMethod, ImageComparison,np, ImageAligning, Histogram, Filter

file1 = Image(path='data/lena3.jpg', model=ColorModel.rgb)
file1.show_comp_with_filter(image=file1, kernel=Filter.SHARP.value())
