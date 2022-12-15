from typing import Optional
from BaseImage import BaseImage, ColorModel, np, plt


class ImageFiltration:

    def conv_layer(self,data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        img_rows, img_columns = data.shape
        data_line = np.reshape(data, (1, data.size))
        kernel_line = np.reshape(kernel, (1, kernel.size))
        layer = np.convolve(data_line[0],kernel_line[0],'same')
        layer = np.reshape(layer, (img_rows, img_columns))
        return layer

    def conv_2d(self, image:BaseImage, kernel: np.ndarray, prefix: Optional[float] = None) -> BaseImage:
        """
         kernel: filtr w postaci tablicy numpy
         prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
         metoda zwroci obraz po procesie filtrowania
         """
        if prefix is None:
            prefix = 1
        if image.color_model == ColorModel.gray:
            image.data = self.conv_layer(image.data, kernel) * prefix
            new_image = BaseImage(data=image.data, model=ColorModel.gray)
            return new_image
        if image.data.shape[2] == 3:
            r_data, g_data, b_data = image.get_layers()
            r = self.conv_layer(r_data, kernel) * prefix
            g = self.conv_layer(g_data, kernel) * prefix
            b = self.conv_layer(b_data, kernel) * prefix
            r[r > 255] = 255
            g[g > 255] = 255
            b[b > 255] = 255
            r[r < 0] = 0
            g[g < 0] = 0
            b[b < 0] = 0
            new_data = np.dstack((r, g, b)).astype('int64')
            new_image = BaseImage(data=new_data, model=ColorModel.rgb)
            return new_image

        pass

    def show_comp_with_filter(self, image:BaseImage, kernel: np.ndarray, prefix: Optional[float] = None):
        fig = plt.figure(figsize=(5, 3))
        rows = 1
        columns = 2
        fig.add_subplot(rows, columns, 1)
        plt.imshow(image.data)
        plt.axis('off')
        plt.title("Before")

        fig.add_subplot(rows, columns, 2)
        plt.imshow(self.conv_2d(image=image, kernel=kernel, prefix=prefix).data)
        plt.axis('off')
        plt.title("After")
        plt.show()
        pass