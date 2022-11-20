from GrayScale import _GrayScaleTransform, _ColorModel, _BaseImage, np
from ImageComparison import _ImageComparison, _ImageDiffMethod

class _Image(_GrayScaleTransform, _ImageComparison):
    """
    klasa stanowiaca glowny interfejs biblioteki
    w pozniejszym czasie bedzie dziedziczyla po kolejnych klasach
    realizujacych kolejne metody przetwarzania obrazow
    """
    def __init__(self, data: np.ndarray = None, path: str = None, model: _ColorModel = None) -> None:
        super().__init__(data, path, model)
        pass