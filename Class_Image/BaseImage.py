from matplotlib.image import imread
from matplotlib.image import imsave
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from ColorModel import ColorModel
from math import acos, pi
import cv2


class BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu
    color_model: ColorModel  # atrybut przechowujacy biezacy model barw obrazu

    def __init__(self, data:np.ndarray = None, path: str = None, model: ColorModel = None) -> None:
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
        if self.color_model == ColorModel.gray:
            imshow(self.data, cmap='gray')
        else:
            imshow(self.data)
        plt.show()
        pass

    def show_img_in_rgb_range(self) -> None:
        """
        metoda wyświetlająca obraz w modelach HSL. HSI, HSL  w przedziale liczb dostępnych w imshow()
        """
        if self.color_model == ColorModel.hsv or self.color_model == ColorModel.hsi or self.color_model == ColorModel.hsl:
            layer1, layer2, layer3 = self.get_layers()
            layer1 = layer1 / 360
            image_in_range = np.dstack((layer1, layer2, layer3))
            imshow(image_in_range)
            plt.show()

    def show_in_all_models(self):
        if self.color_model == ColorModel.rgb:
            fig = plt.figure(figsize=(8, 5))
            rows = 1
            columns = 4
            fig.add_subplot(rows, columns, 1)
            plt.imshow(self.to_hsv().data)
            plt.axis('off')
            plt.title("HSV")

            fig.add_subplot(rows, columns, 2)
            plt.imshow(self.to_hsi().data)
            plt.axis('off')
            plt.title("HSI")

            fig.add_subplot(rows, columns, 3)
            plt.imshow(self.to_hsl().data)
            plt.axis('off')
            plt.title("HSL")

            fig.add_subplot(rows, columns, 4)
            plt.imshow(self.to_rgb().data)
            plt.axis('off')
            plt.title("RGB")
            plt.show()
        else:
            print("Podaj obraz w modelu RGB")

        pass

    def get_layer(self, layer_id: int) -> 'BaseImage':
        """
        metoda zwracająca tylko jedną wybraną warstę obrazu
        """
        return self.data[:, :, layer_id]
        pass

    def RGBandBGR(self):
        l1, l2, l3 = self.get_layers()
        tmp = l1
        l1 = l3
        l3 = tmp
        self.data = np.dstack((l1, l2, l3))
        return self.data


    def calculate_H(self, r, g, b):
        """
        metoda zwracająca H dla modeli kolorów HSV, HSI i HSL
        """
        r = r.astype(int)
        g = g.astype(int)
        b = b.astype(int)
        calc = np.power(r, 2) + np.power(g, 2) + np.power(b, 2) - (r *
                                    g) - (r * b) - (g * b)
        calc_sqrt = np.sqrt(calc)
        calc_sqrt[calc_sqrt == 0] = 1
        H = np.where(g >= b, np.arccos((r - (g / 2) - (b / 2)) / calc_sqrt
                                       ) * 180 / pi,
                     360 - np.arccos(((r - (g / 2) - (b / 2)) / calc_sqrt)
                                     ) * 180 / pi)
        return H

    def get_layers(self) -> np.ndarray:
        """
        metoda zwracająca podzielone warstwy obrazu
        """
        return np.squeeze(np.dsplit(self.data, self.data.shape[-1]))


    def to_hsv(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model != ColorModel.rgb:
            self.to_rgb()
        r, g, b = self.get_layers()
        MAX = np.max([r, g, b], axis=0)
        MAX[MAX == 0] = 255
        MIN = np.min([r, g, b], axis=0)
        H = self.calculate_H(r, g, b)
        S = np.where(MAX == 0, 0, (1 - MIN/MAX))
        V = MAX / 255
        self.data = np.dstack((H, S, V))
        self.color_model = ColorModel.hsv

        print('Przekonwertowano na HSV')
        return self
        pass

    def to_hsi(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model != ColorModel.rgb:
            self.to_rgb()
        R, G, B = np.float32(self.get_layers())
        sum = R + G + B
        sum[sum == 0] = 255
        r = R / sum
        g = G / sum
        b = B / sum
        MIN = np.min([r, g, b], axis=0)
        H = self.calculate_H(R, G, B)
        I = (R + G + B) / (3 * 255)
        S = 1 - 3 * MIN
        self.data = np.dstack((H, S, I))
        self.color_model = ColorModel.hsi
        print('Przekonwertowano na HSI')
        return self
        pass

    def to_hsl(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model != ColorModel.rgb:
            self.to_rgb()
        r, g, b = self.get_layers()
        MAX = np.max([r, g, b], axis=0)
        MIN = np.min([r, g, b], axis=0)
        D = (MAX - MIN) / 255
        H = self.calculate_H(r, g, b)
        L = (0.5 * (MAX.astype(int) + MIN.astype(int))).astype(int) / 255
        calc = (1 - abs(2 * L - 1))
        calc[calc == 0] = 1
        S = np.where(L > 0, D / calc, 0)
        S[S > 1] = 1
        S[S < 0] = 0.001
        self.data = np.dstack((H, S, L))
        self.color_model = ColorModel.hsl

        print('Przekonwertowano na HSL')
        return self
        pass


    def to_rgb(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        """Konwersja do Z HSV DO RGB"""
        if self.color_model == ColorModel.hsv:
                H, S, V = self.get_layers()
                C = V * S
                X = C * (1 - abs(((H / 60) % 2)-1))
                m = V - C
                r = np.where(H >= 300, C, np.where(H >= 240, X, np.where(H >= 120, 0, np.where(H >= 60, X, C))))
                g = np.where(H >= 240, 0, np.where(H >= 180, X, np.where(H >= 60, C, X)))
                b = np.where(H >= 300, X, np.where(H >= 180, C, np.where(H >= 120, X, 0)))
                r = (r + m) * 255
                g = (g + m) * 255
                b = (b + m) * 255
                # Normalize r g b
                g[g > 255] = 255
                b[b > 255] = 255
                r[r > 255] = 255
                r[r < 0] = 0
                g[g < 0] = 0
                b[b < 0] = 0
                r = r.astype(int)
                g = g.astype(int)
                b = b.astype(int)
                self.color_model = ColorModel.rgb
                self.data = np.dstack((r, g, b)).astype(np.uint8)

                print('Przekonwertowano na z HSV na RGB')

        """Konwersja do Z HSI DO RGB"""
        if self.color_model == ColorModel.hsi:
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
                    if np.pi * 2 / 3 <= h[k, j] < np.pi * 4 / 3:
                        h[k, j] = h[k, j] - np.pi * 2 /3
                        x = i[k, j] * (1 - s[k, j])
                        y = i[k, j] * (1 + s[k, j] * np.cos(h[k, j]) / np.cos(np.pi / 3 - h[k, j]))
                        z = 3 * i[k, j] - (x + y)
                        r[k, j] = x
                        g[k, j] = y
                        b[k, j] = z

                    if np.pi * 4 / 3 < h[k, j] < np.pi * 2:
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
            r[r < 0] = 0
            g[g < 0] = 0
            b[b < 0] = 0
            r = r.astype(int)
            g = g.astype(int)
            b = b.astype(int)
            self.data = np.dstack((r, g, b))
            self.color_model = ColorModel.rgb

            print('Przekonwertowano na z HSI na RGB')

        """Konwersja do Z HSL DO RGB"""
        if self.color_model == ColorModel.hsl:
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
            r = r.astype(int)
            g = g.astype(int)
            b = b.astype(int)
            self.color_model = ColorModel.rgb
            self.data = np.dstack((r, g, b))

            print('Przekonwertowano na z HSL na RGB')

        if self.color_model == ColorModel.gray:
            print("Nie można przekonwertować z Modelu 2w do 3w")

        return self
        pass
