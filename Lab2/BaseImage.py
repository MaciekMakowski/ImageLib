from matplotlib.image import imread
from matplotlib.image import imsave
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from ColorModel import _ColorModel


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

    def calculate_HandS(self, r, g, b):
        MAX = np.max([r, g, b], axis=0)
        MIN = np.min([r, g, b], axis=0)
        S = np.where(MAX > 0, 1 - MIN / MAX, 0)
        calc = np.power(r, 2) + np.power(g, 2) + np.power(b, 2) - r * g - r * b - g * b
        H = np.where(g >= b, np.cos((r - g / 2.0 - b / 2.0) / np.sqrt(calc)) ** (-1),
                     360 - np.cos((r - g / 2.0 - b / 2.0) / np.sqrt(calc)) ** (-1))
        H = H / 360
        S[S > 1] = 1
        return H, S

    def get_layers(self):
        return np.squeeze(np.dsplit(self.data, self.data.shape[-1])) / 255

    def to_hsv(self) -> '_BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model != _ColorModel.rgb:
            self.to_rgb()
        r, g, b = self.get_layers()
        MAX = np.max([r, g, b], axis=0)
        V = MAX / 255
        H, S = self.calculate_HandS(r, g, b)

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
        r, g, b = self.get_layers()
        MAX = np.max([r, g, b], axis=0)
        MIN = np.min([r, g, b], axis=0)
        H, S = self.calculate_HandS(r, g, b)
        I = (r + g + b) / 3.0

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
        H, S = self.calculate_HandS(r, g, b)
        L = (0.5 * (MAX + MIN)) / 255

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
            z = (MAX - MIN) * (1 - np.fabs(((H / 60) % 2) - 1))
            r = np.where(H >= 300, MAX,
                         np.where(H >= 240, z + MIN, np.where(H >= 120, MIN, np.where(H >= 60, z + MIN, MAX))))
            g = np.where(H >= 300, z + MIN,
                         np.where(H >= 240, MIN, np.where(H >= 120, MAX, np.where(H >= 60, MAX, z + MIN))))
            b = np.where(H >= 300, MIN, np.where(H >= 240, MAX, np.where(H >= 120, z + MIN, MIN)))
            r[r > 1] = 1
            g[g > 1] = 1
            b[b > 1] = 1
            r[r < 0] = 0
            g[g < 0] = 0
            b[b < 0] = 0
            self.color_model = _ColorModel.rgb
            self.data = np.dstack((r, g, b))

            print('Przekonwertowano na z HSV na RGB')

        """Konwersja do Z HSI DO RGB"""
        if self.color_model == _ColorModel.hsi:
            H, S, I = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
            r = np.where(H > 240, I + I * S * (1 - np.cos(H - 240) / np.cos(300 - H)), np.where(H == 240, I - I * S,
                                                                                                np.where(H > 120,
                                                                                                         I - I * S,
                                                                                                         np.where(
                                                                                                             H == 120,
                                                                                                             I - I * S,
                                                                                                             np.where(
                                                                                                                 H > 0,
                                                                                                                 I + I * S * np.cos(
                                                                                                                     H) / np.cos(
                                                                                                                     60 - H),
                                                                                                                 I + 2 * I * S)))))
            g = np.where(H > 240, I - I * S, np.where(H == 240, I - I * S,
                                                      np.where(H > 120, I + I * S * np.cos(H - 120) / np.cos(180 - H),
                                                               np.where(H == 120, I + 2 * I * S, np.where(H > 0,
                                                                                                          I + I * S * (
                                                                                                                      1 - np.cos(
                                                                                                                  H) / np.cos(
                                                                                                                  60 - H)),
                                                                                                          I - I * S)))))
            b = np.where(H > 240, I + I * S * np.cos(H - 240) / np.cos(300 - H), np.where(H == 240, I + 2 * I * S,
                                                                                          np.where(H > 120,
                                                                                                   I + I * S * (
                                                                                                               1 - np.cos(
                                                                                                           H - 120) / np.cos(
                                                                                                           180 - H)),
                                                                                                   np.where(H == 120,
                                                                                                            I - I * S,
                                                                                                            np.where(
                                                                                                                H > 0,
                                                                                                                I - I * S,
                                                                                                                I - I * S)))))

            self.data = np.dstack((r, g, b))
            self.color_model = _ColorModel.rgb

            print('Przekonwertowano na z HSI na RGB')

        """Konwersja do Z HSL DO RGB"""
        if self.color_model == _ColorModel.hsl:
            H, S, L = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
            d = S * (1 - np.fabs(2 * L - 1))
            MIN = 255 * (L - 0.5 * d)
            x = d * (1 - np.fabs(H / 60) % 2 - 1)
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
            g[g > 1] = 1
            b[b > 1] = 1
            r[r > 1] = 1
            r[r < 0] = 0
            g[g < 0] = 0
            b[b < 0] = 0
            self.color_model = _ColorModel.rgb
            self.data = np.dstack((r, g, b))

            print('Przekonwertowano na z HSL na RGB')

        if self.color_model == _ColorModel.gray:
            print("Nie można przekonwertować z Modelu 2w do 3w")

        return self
        pass
