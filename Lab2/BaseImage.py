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

    def __init__(self,data:np.ndarray = None, path: str = None, model: _ColorModel = None) -> None:
        """
        inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        """
        if np.size(data) > 1:
            self.data = data
        elif path != None:
            self.data = imread(path)
        else:
            print("Podaj dane, lub ścieżkę do pliku")
            return
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
        data = self.data
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
        calc = np.power(r.astype(int), 2) + np.power(g.astype(int), 2) + np.power(b.astype(int), 2) - (r.astype(int) * \
               g.astype(int)) - (r.astype(int) * b.astype(int)) - (g.astype(int) * b.astype(int))
        calc_sqrt = np.sqrt(calc)
        calc_sqrt[calc_sqrt == 0] = 1
        H = np.where(g >= b, np.arccos((r - (g / 2) - (b / 2)) / calc_sqrt
                                     ) * 180 / pi,
                     360 - np.arccos(((r - (g / 2) - (b / 2)) / calc_sqrt)
                                   ) * 180 / pi)
        return H

    def get_layers(self) -> np.ndarray:
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
        H = self.calculate_H(r, g, b)
        diff = MIN / MAX
        diff[np.isnan(diff)] = 1
        S = np.where(MAX == 0, 0, (1 - diff))
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
        R, G, B = np.float32(self.get_layers())
        sum = R + G + B
        sum[sum == 0] = 0.0001
        r = R / sum
        g = G / sum
        b = B / sum
        MIN = np.min([r, g, b], axis=0)
        H = self.calculate_H(R,G,B)
        I = (R + G + B) / (3 * 255)
        S = 1 - 3 * MIN
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
        H = self.calculate_H(r, g, b)
        L = (0.5 * (MAX.astype(int) + MIN.astype(int))).astype(int) / 255
        calc = (1 - abs(2 * L - 1))
        calc[calc == 0] = 0.0001
        S = np.where(L > 0, D / calc, 0)
        S[S > 1] = 1
        S[S < 0] = 0.00001
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
                H, S, V = self.get_layers()
                MAX = 255 * V
                MIN = MAX * (1 - S)
                z = (MAX - MIN) * (1 - abs(((np.radians(H) / np.radians(60)) % 2) - 1))
                r = np.where(H >= 300, MAX,
                             np.where(H >= 240, z.astype(int) + MIN.astype(int),
                                      np.where(H >= 120, MIN,
                                               np.where(H >= 60, z.astype(int) + MIN.astype(int), MAX))))
                g = np.where(H >= 300, MIN,
                             np.where(H >= 240, MIN,
                                      np.where(H >= 120, MAX,
                                               np.where(H >= 60, MAX, z + MIN))))
                b = np.where(H >= 300, z + MIN,
                            np.where(H >= 240, MAX,
                                      np.where(H >= 120, z + MIN, MIN)))
                # Normalize r g b
                g[g > 255] = 255
                b[b > 255] = 255
                r[r > 255] = 255
                r[r < 0] = 0
                g[g < 0] = 0
                b[b < 0] = 0
                self.color_model = _ColorModel.rgb
                self.data = np.dstack((r, g, b)).astype(np.uint8)

                print('Przekonwertowano na z HSV na RGB')

        """Konwersja do Z HSI DO RGB"""
        if self.color_model == _ColorModel.hsi:
            H, S, I = self.get_layers()
            h = H * np.pi / 180
            s = S
            i = I
            rows = self.data.shape[0]
            columns = self.data.shape[1]
            r = np.zeros((rows, columns))
            g = np.zeros((rows, columns))
            b = np.zeros((rows, columns))
            for k in range(rows):
                for j in range(columns):
                    if h[k, j] < np.pi * 2 / 3:
                        x = i[k, j] * (1 - s[k, j])
                        y = i[k, j] * (1 + s[k, j] * np.cos(h[k, j]) / np.cos(np.pi / 3 - h[k, j]))
                        z = 3 * i[k, j] - (x + y)
                        r[k, j] = y
                        g[k, j] = z
                        b[k, j] = x
                    if np.pi * 2 / 3 <= h[k, j] < np.pi * 4 /3 :
                        h[k, j] = h[k, j] - np.pi * 2 /3
                        x = i[k, j] * (1 - s[k, j])
                        y = i[k, j] * (1 + s[k, j] * np.cos(h[k, j]) / np.cos(np.pi / 3 - h[k, j]))
                        z = 3 * i[k, j] - (x + y)
                        r[k, j] = x
                        g[k, j] = y
                        b[k, j] = z

                    if np.pi * 4 /3 < h[k, j] < np.pi * 2:
                        h[k, j] = h[k, j] - np.pi * 4 / 3
                        x = i[k, j] * (1 - s[k, j])
                        y = i[k, j] * (1 + s[k, j] * np.cos(h[k, j]) / np.cos(np.pi / 3 - h[k, j]))
                        z = 3 * i[k,j] - (x + y)
                        r[k, j] = z
                        g[k, j] = x
                        b[k, j] = y
            # [0...1] to [0...255]
            r[r > 1] = 1
            g[g > 1] = 1
            b[b > 1] = 1
            r = r * 255
            g = g * 255
            b = b * 255
            #Normalize r g b
            g[g > 255] = 255
            b[b > 255] = 255
            r[r > 255] = 255
            r[r < 0] = 0
            g[g < 0] = 0
            b[b < 0] = 0
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
            #Normalize r g b
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
