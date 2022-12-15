from BaseImage import BaseImage, np, ColorModel


class GrayScaleTransform(BaseImage):
    def __init__(self, data:np.ndarray = None, path: str = None, model:ColorModel = None) -> None:
        super().__init__(data, path, model)
        pass

    def to_gray(self) -> BaseImage:
        """
        metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        """
        if self.color_model != ColorModel.rgb:
            self.to_rgb()
        r, g, b = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
        r_gray = r * 0.299
        g_gray = g * 0.587
        b_gray = b * 0.114
        gray_layer = r_gray + g_gray + b_gray
        self.data = gray_layer.astype('uint8')
        self.color_model = ColorModel.gray
        print("Przekonwertowano na 2D")
        return self
        pass

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        """
        metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta
        lub metoda 2 w przypadku przekazania argumentu w
        """
        if alpha_beta != (None, None) and w != None:
            print('Podaj tylko 1 argument')
            return
        if alpha_beta == (None, None) and w == None:
            print('Podaj agrument alpha_beta (Jako tuple) lub w (jako int)')
            return
        if w != None:
            if w < 20 or w > 40:
                print('Parametr W musi znajdować się w przedziale od 20 do 40')
                return
        if alpha_beta != None:
            if alpha_beta[0] != alpha_beta[1] == 2:
                print('Alpha + Beta musi równać się 2')
                return
        if self.color_model != ColorModel.gray:
            self.to_gray()
        L0 = self.data / 255
        L1 = self.data / 255
        L2 = self.data / 255
        if alpha_beta != (None, None):
            L0 = L0 * alpha_beta[0]
            L0[L0 > 1] = 1
            L2 = L2 * alpha_beta[1]
            L2[L2 > 1] = 1
            self.data = np.dstack((L0, L1, L2))
            self.color_model = ColorModel.rgb

        if w:
            L0 = L0 + 2 * (w/255)
            L0[L0 > 1] = 1
            L1 = L1 + (w/255)
            L1[L1 > 1] = 1
            self.data = np.dstack((L0, L1, L2))
            self.color_model = ColorModel.rgb

        return self
        pass