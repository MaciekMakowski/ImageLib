from matplotlib.image import imread
from matplotlib.image import imsave
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # obraz 2d

class BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu
    color_model: ColorModel  # atrybut przechowujacy biezacy model barw obrazu
    def __init__(self, path: str) -> None:
        """
        inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        """
        self.data = imread(path)
        self.color_model= ColorModel.rgb
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
        plt.imshow(self.data)
        plt.show()
        pass

    def get_layer(self, layer_id: int) -> 'BaseImage':
        """
        metoda zwracajaca warstwe o wskazanym indeksie
        """
        r_layer, g_layer, b_layer = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
        if(layer_id == 0):
            return r_layer
        elif (layer_id == 1):
            return g_layer
        elif(layer_id == 2):
           return b_layer
        pass

    def calculate_HandS(self,R,G,B):
        MAX = max(R, G, B)
        MIN = min(R, G, B)
        if (MAX > 0):
            S = 1 - (MIN / MAX)
        else:
            S = 0
        pattern = np.cos((int(R) - int(G / 2) - int(B / 2)) / np.sqrt(int(R ** 2) + int(G ** 2) + int(B ** 2) - (int(R) * int(G)) - (int(R) * int(B)) - (int(G) * int(B)))) ** (-1)
        if (G >= B):
            H = pattern
        else:
            H = 360 - pattern
        return H, S

    def to_hsv(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if(self.color_model != ColorModel.rgb):
            self.to_rgb()
        self.data = self.data.copy()
        for px in self.data:
            for color in px:
                V=max(color[0], color[1], color[2])/255
                H, S = self.calculate_HandS(color[0], color[1], color[2])
                color[0] = H*np.pi*2
                color[1] = S*180
                color[2] = V*180

        self.color_model = ColorModel.hsv
        print('Przekonwertowano na z HSV')
        return self
        pass

    def to_hsi(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if(self.color_model != ColorModel.rgb):
            self.to_rgb()
        self.data = self.data.copy()
        for px in self.data:
            for color in px:
                H, S = self.calculate_HandS(color[0],color[1],color[2])
                I = (int(color[0]) + int(color[1]) + int(color[2])) /3
                color[0] = H
                color[1] = S
                color[2] = I


        self.color_model = ColorModel.hsi
        print('Przekonwertowano na z HSI')
        return self
        pass

    def to_hsl(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if(self.color_model != ColorModel.rgb):
            self.to_rgb()
        self.data = self.data.copy()
        for px in self.data:
            for color in px:
                H, S = self.calculate_HandS(color[0], color[1], color[2])
                MAX = max(color[0], color[1], color[2])
                MIN = min(color[0], color[1], color[2])
                L = (0.5*(int(MAX)+int(MIN)))/255
                color[0] = H*100
                color[1] = S*100
                color[2] = L*100
        self.color_model = ColorModel.hsl
        print('Przekonwertowano na HSL')
        return self
        pass

    def to_rgb(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        self.data = self.data.copy()

        """Konwersja do Z HSV DO RGB"""
        if(self.color_model== ColorModel.hsv ):
            for px in self.data:
                for color in px:
                    MAX = (255 * (color[2]))
                    MIN = (MAX*(1-(color[1])))
                    Z = (MAX - MIN) * (1 - np.abs((((color[0])/60) % 2)-1))
                    if color[0] >= 300:
                        color[0] = MAX
                        color[1] = MIN
                        color[2] = Z + MIN
                    elif color[0] >= 240:
                        color[0] = Z + MIN
                        color[1] = MIN
                        color[2] = MAX
                    elif color[0] >= 180:
                        color[0] = MIN
                        color[1] = MAX
                        color[2] = Z + MIN
                    elif color[0] >= 120:
                        color[0] = MIN
                        color[1] = MAX
                        color[2] = Z + MIN
                    elif color[0] >= 60:
                        color[0] = Z + MIN
                        color[1] = MAX
                        color[2] = MIN
                    elif color[0] >= 0:
                        color[0] = MAX
                        color[1] = Z + MIN
                        color[2] = MIN
            print('Przekonwertowano na z HSV na RGB')
            self.color_model = ColorModel.rgb

        """Konwersja do Z HSI DO RGB"""
        if (self.color_model == ColorModel.hsi):
            for px in self.data:
                for color in px:
                    H, S, I = int(color[0]), int(color[1]), int(color[2])
                    MAX = 255 * (color[2])
                    MIN = MAX * (1 - (color[1]))
                    Z = (MAX - MIN) * (1 - np.abs(((color[0] / 60) % 2)) - 1)
                    if H >= 240:
                        color[0] = I + I * S * (1 - np.cos(H - 240) / np.cos(300 - H))
                        color[1] = I - I * S
                        color[2] = I + I * S * (np.cos(H - 240) / np.cos(300 - H))
                    elif H == 240:
                        color[0] = I - I * S
                        color[1] = I - I * S
                        color[2] = I + 2 * I * S
                    elif H > 120:
                        color[0] = I - I * S
                        color[1] = I + I * S * (np.cos(H - 120) / np.cos(180 - H))
                        color[2] = I + I * S * (1 - np.cos(H - 120) / np.cos(180 - H))
                    elif H == 120:
                        color[0] = I - I * S
                        color[1] = I + 2 * I * S
                        color[2] = I - I * S
                    elif H > 0:
                        color[0] = I + I * S * np.cos(H) / np.cos(60 - H)
                        color[1] = I + I * S * (1 - np.cos(H) / np.cos(60 - H))
                        color[2] = I - I * S
                    elif H == 0:
                        color[0] = I + 2 * I * S
                        color[1] = I - I * S
                        color[2] = I - I * S
            print('Przekonwertowano na z HSI na RGB')
            self.color_model = ColorModel.rgb

        """Konwersja do Z HSL DO RGB"""
        if (self.color_model == ColorModel.hsl):
            for px in self.data:
                for color in px:
                    H, S, L = int(color[0]), int(color[1]), int(color[2])
                    D = S * (1 - np.abs((2 * L) - 1))
                    MIN = 255 * (L - 0.5 * D)
                    X = D * (1 - np.abs(((H / 60) % 2) - 1))
                    if H >= 300:
                        color[0] = (255 * D) + MIN
                        color[1] = MIN
                        color[2] = (255 * X) + MIN
                    elif H >= 240:
                        color[0] = (255 * X) + MIN
                        color[1] = MIN
                        color[2] = (255 * D) + MIN
                    elif H >= 180:
                        color[0] = MIN
                        color[1] = (255 * X) + MIN
                        color[2] = (255 * D) + MIN
                    elif H >= 120:
                        color[0] = MIN
                        color[1] = (255 * D) + MIN
                        color[2] = (255 * X) + MIN
                    elif H >= 60:
                        color[0] = (255 * X) + MIN
                        color[1] = (255 * D) + MIN
                        color[2] = MIN
                    elif H >= 0:
                        color[0] = (255 * D) + MIN
                        color[1] = (255 * X) + MIN
                        color[2] = MIN
            print('Przekonwertowano na z HSL na RGB')
            self.color_model = ColorModel.rgb
        if(self.color_model == ColorModel.gray):
            print("Nie można przekonwertować z Modelu 2w do 3w")

        return self
        pass
file = BaseImage('data/lena.jpg')
file.to_hsv()
file.show_img()
file.to_rgb()
file.show_img()






