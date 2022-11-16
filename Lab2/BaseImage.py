from matplotlib.image import imread
from matplotlib.image import imsave
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from ColorModel import _ColorModel
from math import acos, pi


class _BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu
    color_model: _ColorModel  # atrybut przechowujacy biezacy model barw obrazu

    def __init__(self, path: str, model: _ColorModel) -> None:
        """
        inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        """
        self.data = imread(path)
        self.color_model = model
        pass

    def save_img(self, path: str) -> None:
        """
        metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
         """
        imsave(path, self.data)
        pass

    def show_img(self) -> None:
        """
        metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        """
        if self.color_model == _ColorModel.gray:
            imshow(self.data, cmap='gray')
        else:
            imshow(self.data)
        plt.show()
        pass

    def get_layer(self, layer_id: int) -> '_BaseImage':
        return self.data[:, :, layer_id]
        pass

    def calculate_H(self, r, g, b):
        calc = np.power(r.astype(int), 2) + np.power(g.astype(int), 2) + np.power(b.astype(int), 2) - r.astype(int) * \
               g.astype(int) - r.astype(int) * b.astype(int) - g.astype(int) * b.astype(int)
        H = np.where(g >= b, np.arccos((r.astype(int) - g.astype(int) / 2 - b.astype(int) / 2) / np.sqrt(calc)
                                     ) * 180 / pi,
                     360 - np.arccos(((r.astype(int) - g.astype(int) / 2 - b.astype(int) / 2) / np.sqrt(calc))
                                   ) * 180 / pi)
        return H

    def get_layers(self):
        return np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

    def to_hsv(self) -> '_BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model != _ColorModel.rgb:
            self.to_rgb()
        r, g, b = self.get_layers()
        MAX = np.max([r, g, b], axis=0)
        MIN = np.min([r, g, b], axis=0)
        V = MAX / 255
        H = self.calculate_H(r, g, b)
        S = np.where(MAX > 0, 1 - MIN / MAX, 0)
        V = MAX / 255
        self.data = np.dstack((H, S, V))
        self.color_model = _ColorModel.hsv

        print('Przekonwertowano na HSV')
        return self
        pass

    def to_hsi(self) -> '_BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model != _ColorModel.rgb:
            self.to_rgb()
        r, g, b = np.float32(self.get_layers())
        MIN = np.min([r, g, b], axis=0)
        calc = np.power(r, 2) + np.power(g, 2) + np.power(b, 2) - r * \
               g - r * b - g * b
        H = np.where(g >= b, np.arccos((r - g / 2 - b / 2) / np.sqrt(calc)
                                       ) * 180 / pi,
                     360 - np.arccos(((r.astype(int) - g.astype(int) / 2 - b.astype(int) / 2) / np.sqrt(calc))
                                     ) * 180 / pi)
        I = (r + g + b) / 3
        S = np.where(I > 0, 1 - MIN / I, 0)
        self.data = np.dstack((H, S, I))
        self.color_model = _ColorModel.hsi

        print('Przekonwertowano na HSI')
        return self
        pass

    def to_hsl(self) -> '_BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        r, g, b = self.get_layers()
        MAX = np.max([r, g, b], axis=0)
        MIN = np.min([r, g, b], axis=0)
        D = (MAX - MIN)/255
        H= self.calculate_H(r, g, b)
        L = (0.5 * (MAX.astype(int) + MIN.astype(int))) / 255
        S = np.where(L>0, D / (1 - abs (2 * L - 1)), 0)

        self.data = np.dstack((H, S, L))
        self.color_model = _ColorModel.hsl

        print('Przekonwertowano na HSL')
        return self
        pass

    def to_rgb(self) -> '_BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        """Konwersja do Z HSV DO RGB"""
        if self.color_model == _ColorModel.hsv:
                H, S, V = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
                MAX = 255 * V
                MIN = MAX * (1 - S)
                z = (MAX - MIN) * (1 - abs(((H / 60) % 2) - 1))
                r = np.where(H >= 300, MAX,
                             np.where(H >= 240, z + MIN, np.where(H >= 120, MIN, np.where(H >= 60, z + MIN, MAX))))
                g = np.where(H >= 300, MIN,
                             np.where(H >= 240, MIN, np.where(H >= 120, MAX, np.where(H >= 60, MAX, z + MIN))))
                b = np.where(H >= 300, z + MIN, np.where(H >= 240, MAX, np.where(H >= 120, z + MIN, MIN)))
                self.color_model = _ColorModel.rgb
                self.data = np.dstack((r, g, b)).astype(np.uint16)

                print('Przekonwertowano na z HSV na RGB')

        """Konwersja do Z HSI DO RGB"""
        if self.color_model == _ColorModel.hsi:
            H, S, I = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

            rows = self.data.shape[0]
            columns = self.data.shape[1]

            r = np.zeros((rows, columns))
            g = np.zeros((rows, columns))
            b = np.zeros((rows, columns))

            for i in range(rows):
                for j in range(columns):

                    if b[i, j] == g[i, j] == r[i, j]:
                        H[i, j] = 0

                    if H[i, j] == 0:
                        r[i, j] = I[i, j] + 2 * (I[i, j] * S[i, j])
                        g[i, j] = I[i, j] - I[i, j] * S[i, j]
                        b[i, j] = I[i, j] - I[i, j] * S[i, j]

                    if 0 <= H[i, j] <= 120:
                        b[i, j] = I[i, j] * (1 - S[i, j])
                        r[i, j] = I[i, j] * (
                                1 + (S[i, j] * np.cos(np.radians(H[i, j]))) / np.cos(np.radians(60) - np.radians(H[i, j])))
                        g[i, j] = 3 * I[i, j] - (r[i, j] + b[i, j])

                    if 120 < H[i, j] <= 240 or H[i, j] == 120:
                        r[i, j] = I[i, j] * (1 - S[i, j])
                        g[i, j] = I[i, j] * (
                                1 + (S[i, j] * np.cos(np.radians(H[i, j]))) / np.cos(np.radians(60) - np.radians(H[i, j])))
                        b[i, j] = 3 * I[i, j] - (r[i, j] + g[i, j])

                    if 240 < H[i, j] <= 360 or H[i, j] == 240:
                        g[i, j] = I[i, j] * (1 - S[i, j])
                        b[i, j] = I[i, j] * (
                                1 + (S[i, j] * np.cos(np.radians(H[i, j]))) / np.cos(np.radians(60) - np.radians(H[i, j])))
                        r[i, j] = 3 * I[i, j] - (g[i, j] + b[i, j])

                    elif r[i, j] > 255:
                        r[i, j] = 255
            self.data = np.dstack((r, g, b)).astype(np.uint16)
            self.color_model = _ColorModel.rgb

            print('Przekonwertowano na z HSI na RGB')

        """Konwersja do Z HSL DO RGB"""
        if self.color_model == _ColorModel.hsl:
            H, S, L = self.get_layers()
            d = S * (1 - abs(2 * L - 1))
            MIN = 255 * (L - 0.5 * d)
            x = d * (1 - abs(H / 60 % 2 - 1))
            r = np.where(H >= 300, 255 * d + MIN, np.where(H >= 240, 255 * x + MIN, np.where(H >= 180, MIN,
                                                                                             np.where(H >= 120, MIN,
                                                                                                      np.where(H >= 60,
                                                                                                               255 * x + MIN,
                                                                                                               255 * d + MIN)))))
            g = np.where(H >= 300, MIN, np.where(H >= 240, MIN, np.where(H >= 180, 255 * x + MIN,
                                                                         np.where(H >= 120, 255 * d + MIN,
                                                                                  np.where(H >= 60, 255 * d + MIN,
                                                                                           255 * x + MIN)))))
            b = np.where(H >= 300, 255 * x + MIN, np.where(H >= 240, 255 * d + MIN, np.where(H >= 180, 255 * d + MIN,
                                                                                             np.where(H >= 120,
                                                                                                      255 * x + MIN,
                                                                                                      MIN))))
            g[g > 255] = 255
            b[b > 255] = 255
            r[r > 255] = 255
            r[r < 0] = 0
            g[g < 0] = 0
            b[b < 0] = 0
            self.color_model = _ColorModel.rgb
            self.data = np.dstack((r, g, b)).astype(np.int16)

            print('Przekonwertowano na z HSL na RGB')

        if self.color_model == _ColorModel.gray:
            print("Nie można przekonwertować z Modelu 2w do 3w")

        return self
        pass
