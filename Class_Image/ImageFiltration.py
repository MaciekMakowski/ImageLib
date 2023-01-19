from typing import Optional
from BaseImage import BaseImage, ColorModel, np, plt
from Filtres import Filter
from GrayScale import GrayScaleTransform

class ImageFiltration(BaseImage):

    def conv_layer(self, data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        img_rows, img_columns = data.shape
        data_line = np.reshape(data, (1, data.size))
        kernel_line = np.reshape(kernel, (1, kernel.size))
        layer = np.convolve(data_line[0],kernel_line[0],'same')
        layer = np.reshape(layer, (img_rows, img_columns))
        return layer

    def conv_2d(self, kernel: np.ndarray, prefix: Optional[float] = None) -> BaseImage:
        """
         kernel: filtr w postaci tablicy numpy
         prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
         metoda zwroci obraz po procesie filtrowania
         """
        if prefix is None:
            prefix = 1
        if self.color_model == ColorModel.gray:
            self.data = self.conv_layer(self.data, kernel) * prefix
            new_image = BaseImage(data=self.data, model=ColorModel.gray)
            return new_image
        if self.data.shape[2] == 3:
            r_data, g_data, b_data = self.get_layers()
            r = self.conv_layer(r_data, kernel) * prefix
            g = self.conv_layer(g_data, kernel) * prefix
            b = self.conv_layer(b_data, kernel) * prefix
            r[r > 255] = 255
            g[g > 255] = 255
            b[b > 255] = 255
            r[r < 0] = 0
            g[g < 0] = 0
            b[b < 0] = 0
            new_data = np.dstack((r, g, b)).astype('uint8')
            new_image = BaseImage(data=new_data, model=ColorModel.rgb)
            return new_image

        pass

    def show_comp_with_filter(self, kernel: np.ndarray, prefix: Optional[float] = None):
        fig = plt.figure(figsize=(5, 3))
        rows = 1
        columns = 2
        fig.add_subplot(rows, columns, 1)
        plt.imshow(self.data)
        plt.axis('off')
        plt.title("Before")

        fig.add_subplot(rows, columns, 2)
        plt.imshow(self.conv_2d(kernel=kernel, prefix=prefix).data)
        plt.axis('off')
        plt.title("After")
        plt.show()
        pass

    def create_margins(self):
        margin0 = self.conv_2d(Filter.W0.value())
        margin45 = self.conv_2d(Filter.W45.value())
        margin90 = self.conv_2d(Filter.W90.value())
        margin135 = self.conv_2d(Filter.W135.value())
        new_image = BaseImage(data=margin0.data+margin45.data+margin90.data+margin135.data, model=ColorModel.gray)
        new_image.data[ new_image.data > 255] = 255
        return new_image
