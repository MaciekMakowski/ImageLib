import matplotlib.pyplot as plt
import numpy as np

from BaseImage import *

class Histogram:
    """
    klasa reprezentujaca histogram danego obrazu
    """
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, values: np.ndarray) -> None:
        if values.ndim == 2:
            self.values = np.histogram(values, bins=256, range=(0, 256))[0]
        else:
            LR, LG, LB = np.squeeze(np.dsplit(values, values.shape[-1]))
            LR = np.histogram(LR, bins=256, range=(0, 255))[0]
            LG = np.histogram(LG, bins=256, range=(0, 255))[0]
            LB = np.histogram(LB, bins=256, range=(0, 255))[0]
            self.values = np.dstack((LR, LG, LB))
        pass

    def get_layers(self):
        return np.squeeze(np.dsplit(self.values, self.values.shape[-1]))

    def plot(self) -> None:
        """
        metoda wyswietlajaca histogram na podstawie atrybutu values
        """
        if self.values.ndim > 2:
            self.show_multi_histograms()
        elif self.values.ndim:
            plt.title('Gray')
            plt.plot(np.linspace(0, 255, 256), self.values,color='black')
        plt.show()
        pass

    def show_multi_histograms(self):
        plt.figure(figsize=(15, 6))
        num = 1
        space = np.linspace(0, 255, 256)
        for layer in self.get_layers():
            plt.subplot(1, 3, num)
            plt.title(self.hist_color(num))
            plt.xlim([0, 256])
            plt.plot(space, layer, color=self.hist_color(num))
            num += 1

    def to_cumulated(self):
        self.values = np.cumsum(self.values)
    def hist_color(self, num: int) -> str:
        match num:
            case 1:
                return "red"
            case 2:
                return "green"
            case 3:
                return "blue"
