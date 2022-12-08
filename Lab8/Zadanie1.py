from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.image import imsave
import math
from matplotlib.colors import hsv_to_rgb
from typing import Optional


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
        self.data = plt.imread(path)

    def save_img(self, path: str) -> None:
        """
        metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
        """
        imsave('file.jpg', self.data)

    def show_img(self) -> None:
        """
        metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        """
        plt.imshow(self.data)
        plt.show()

    def get_layer(self, layer_id: int) -> 'BaseImage':
        """
        metoda zwracajaca warstwe o wskazanym indeksie
        """
        layer_img = np.zeros(self.data.shape, dtype=int)
        layer_img[:, :, layer_id] = self.data[:, :, layer_id]
        print(layer_img)
        self.data = layer_img
        return self

    def to_hsv(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        new_array = []
        for row in self.data:
            new_row = []
            for pixel in row:
                r = int(pixel[0])
                g = int(pixel[1])
                b = int(pixel[2])
                if math.sqrt(float(r * r + g * g + b * b - r * g - r * b - g * b)) == 0.0:
                    h = 0
                elif g >= b:
                    h = int(math.acos(
                        (r - 0.5 * g - 0.5 * b) / (math.sqrt(float(r * r + g * g + b * b - r * g - r * b - g * b)))))
                else:
                    h = int(360 - math.acos(
                        (r - 0.5 * g - 0.5 * b) / (math.sqrt(float(r * r + g * g + b * b - r * g - r * b - g * b)))))
                maximum = np.amax(row)
                minimum = np.amin(row)
                v = maximum / 255  # B
                if maximum > 0:
                    s = 1 - minimum / maximum  # G
                else:
                    s = 0  # G
                pixel = [h, s, v]
                new_row.append(pixel)
            new_array.append(new_row)
        self.data = np.asarray(new_array)
        return self

    def to_hsi(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        new_array = []
        for row in self.data:
            new_row = []
            for pixel in row:
                r = int(pixel[0])
                g = int(pixel[1])
                b = int(pixel[2])
                if math.sqrt(float(r * r + g * g + b * b - r * g - r * b - g * b)) == 0.0:
                    h = 0
                elif g >= b:
                    h = int(math.acos(
                        (r - 0.5 * g - 0.5 * b) / (math.sqrt(float(r * r + g * g + b * b - r * g - r * b - g * b)))))
                else:
                    h = int(360 - math.acos(
                        (r - 0.5 * g - 0.5 * b) / (math.sqrt(float(r * r + g * g + b * b - r * g - r * b - g * b)))))
                maximum = np.amax(row)
                minimum = np.amin(row)
                i = (r + g + b) / 3  # B
                if maximum > 0:
                    s = 1 - minimum / maximum  # G
                else:
                    s = 0  # G
                pixel = [h, s, i]
                new_row.append(pixel)
            new_array.append(new_row)
        self.data = np.asarray(new_array)
        return self

    def to_hsl(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        new_array = []
        for row in self.data:
            new_row = []
            for pixel in row:
                r = int(pixel[0])
                g = int(pixel[1])
                b = int(pixel[2])
                if math.sqrt(float(r * r + g * g + b * b - r * g - r * b - g * b)) == 0.0:
                    h = 0
                elif g >= b:
                    h = int(math.acos(
                        (r - 0.5 * g - 0.5 * b) / (math.sqrt(float(r * r + g * g + b * b - r * g - r * b - g * b)))))
                else:
                    h = int(360 - math.acos(
                        (r - 0.5 * g - 0.5 * b) / (math.sqrt(float(r * r + g * g + b * b - r * g - r * b - g * b)))))
                maximum = np.amax(row)
                minimum = np.amin(row)
                l = (0.5 * (maximum + minimum)) / 255  # B
                if maximum > 0:
                    s = 1 - minimum / maximum  # G
                else:
                    s = 0  # G
                pixel = [h, s, l]
                new_row.append(pixel)
            new_array.append(new_row)
        self.data = np.asarray(new_array)
        self.color_model = ColorModel(3)
        return self

    def to_rgb(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model == 1:
            new_array = []
            r = 0
            g = 0
            b = 0
            for row in self.data:
                new_row = []
                for pixel in row:
                    h = pixel[0]
                    s = pixel[1]
                    v = pixel[2]
                    M = 255 * v
                    m = M * (1 - s)
                    z = (M - m) * (1 - abs(((h / 60) % 2) - 1))
                    if 0 <= h < 60:
                        r = M
                        g = z + m
                        b = m
                    if 60 <= h < 120:
                        r = z + m
                        g = M
                        b = m
                    if 120 <= h < 180:
                        r = m
                        g = M
                        b = z + m
                    if 180 <= h < 240:
                        r = m
                        g = M
                        b = z + m
                    if 240 <= h < 300:
                        r = z + m
                        g = m
                        b = M
                    if 300 <= h < 3060:
                        r = M
                        g = m
                        b = z + m
                    pixel = [r, g, b]
                    new_row.append(pixel)
                new_array.append(new_row)
            self.data = np.asarray(new_array)
            return self
        if self.color_model == 2:
            new_array = []
            r = 0
            g = 0
            b = 0
            for row in self.data:
                new_row = []
                for pixel in row:
                    h = pixel[0]
                    s = pixel[1]
                    i = pixel[2]
                    if h == 0:
                        r = i + 2 * i * s
                        g = i - i * s
                        b = i - i * s
                    if 0 < h < 120:
                        r = i + i * s * math.cos(h) / math.cos(60 - h)
                        g = i + i * s * (1 - math.cos(h) / math.cos(60 - h))
                        b = i - i * s
                    if h == 120:
                        r = i - i * s
                        g = i + 2 * i * s
                        b = i - i * s
                    if 120 < h < 240:
                        r = i - i * s
                        g = i + i * s * math.cos(h - 120) / math.cos(180 - h)
                        b = i + i * s * (1 - math.cos(h - 120) / math.cos(180 - h))
                    if h == 240:
                        r = i - i * s
                        g = i - i * s
                        b = i + 2 * i * s
                    if 240 < h < 360:
                        r = i + i * s(1 - math.cos(h - 240) / math.cos(300 - h))
                        g = i - i * s
                        b = i + i * s * math.cos(h - 240) / math.cos(300 - h)
                    pixel = [r, g, b]
                    new_row.append(pixel)
                new_array.append(new_row)
            self.data = np.asarray(new_array)
            return self
        if self.color_model == 3:
            new_array = []
            r = 0
            g = 0
            b = 0
            for row in self.data:
                new_row = []
                for pixel in row:
                    h = pixel[0]
                    s = pixel[1]
                    L = pixel[2]
                    d = s * (1 - abs(2 * L - 1))
                    m = 255 * (L - 0.5 * d)
                    x = d * (1 - abs(((h / 60) % 2) - 1))
                    if 0 <= h < 60:
                        r = 255 * d + m
                        g = 255 * x + m
                        b = m
                    if 60 <= h < 120:
                        r = 255 * x * m
                        g = 255 * d + m
                        b = m
                    if 120 <= h < 180:
                        r = m
                        g = 255 * d + m
                        b = 255 * x + m
                    if 180 <= h < 240:
                        r = m
                        g = 255 * x + m
                        b = 255 * d + m
                    if 240 <= h < 300:
                        r = 255 * x + m
                        g = m
                        b = 255 * d + m
                    if 300 <= h < 360:
                        r = 255 * d + m
                        g = m
                        b = 255 * x + m
                    pixel = [r, g, b]
                    new_row.append(pixel)
                new_array.append(new_row)
            self.data = np.asarray(new_array)
            return self


class GrayScaleTransform(BaseImage):
    """def __init__(self) -> None:
        pass"""

    def to_gray(self) -> BaseImage:
        """
        metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        """
        inputImage = self.data

        grayImage = np.zeros(inputImage.shape, dtype=float)
        R = np.array(inputImage[:, :, 0]) * 0.229
        G = np.array(inputImage[:, :, 1]) * 0.587
        B = np.array(inputImage[:, :, 2]) * 0.114

        average = (R + G + B)

        for i in range(3):
            grayImage[:, :, i] = average

        self.data = grayImage
        return self

    def to_sepia(self, alpha_beta: tuple = None, w: int = None) -> BaseImage:
        """
        metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta
        lub metoda 2 w przypadku przekazania argumentu w
        """

        temp = self.data

        if alpha_beta is not None:
            alpha = alpha_beta[0]
            beta = alpha_beta[1]
            if alpha > 1 and beta < 1 and alpha + beta == 2:
                self.to_gray()
                new_array = []
                for row in self.data:
                    new_row = []
                    for pixel in row:
                        l0 = float(pixel[0])
                        l1 = float(pixel[1])
                        l2 = float(pixel[2])
                        l0 = l0 * alpha
                        l2 = l2 * beta
                        pixel = [l0, l1, l2]
                        new_row.append(pixel)
                    new_array.append(new_row)
                self.data = np.asarray(new_array)
        if w is not None:
            if w >= 20 and w <= 40:
                self.to_gray()
                new_array = []
                for row in self.data:
                    new_row = []
                    for pixel in row:
                        l0 = pixel[0] * 255
                        l1 = pixel[1] * 255
                        l2 = pixel[2] * 255
                        l0 = l0 + 2 * w
                        l1 = l1 + w
                        if l0 > 255:
                            l0 = 255
                        if l1 > 255:
                            l1 = 255
                        pixel = [int(l0), int(l1), int(l2)]
                        new_row.append(pixel)
                    new_array.append(new_row)
                self.data = np.asarray(new_array)

        return self


class Image(GrayScaleTransform):
    """
    klasa stanowiaca glowny interfejs biblioteki
    w pozniejszym czasie bedzie dziedziczyla po kolejnych klasach
    realizujacych kolejne metody przetwarzania obrazow
    """
    """def __init__(self, ...) -> None:
        pass"""


class Histogram:
    """
    klasa reprezentujaca histogram danego obrazu
    """
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, values: np.ndarray) -> None:
        # values = BaseImage(self.data)
        self.values = values

    def plot(self) -> None:

        """
        metoda wyswietlajaca histogram na podstawie atrybutu values
        """
        if len(self.values.shape) == 0:
            plt.figure()
            plt.plot(self.values, 'gray')
        else:
            plt.figure()
            plt.plot(self.values[0], 'r')
            plt.plot(self.values[1], 'g')
            plt.plot(self.values[2], 'b')
        plt.show()


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1


class ImageComparison(BaseImage):
    """
    Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    """

    def histogram(self) -> Histogram:
        """
        metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        """
        r = 0
        g = 0
        b = 0
        new_array = []
        for row in self.data:
            new_row = []
            for pixel in row:
                r = int(pixel[0])
                g = int(pixel[1])
                b = int(pixel[2])
                if r != g != b:
                    greyscale = True
                else:
                    greyscale = False

        if greyscale is True:
            gray, bin_edges = np.histogram(self.data, bins=256, range=(0, 1))
            return Histogram(gray)
        else:
            red, bins = np.histogram(self.data[:, :, 0], bins=256, range=(0, 256))
            green, bins = np.histogram(self.data[:, :, 1], bins=256, range=(0, 256))
            blue, bins = np.histogram(self.data[:, :, 2], bins=256, range=(0, 256))
            print(red)
            rgb = np.vstack((red, blue, green))
            return Histogram(rgb)

    def compare_to(self, other: Image, method: ImageDiffMethod) -> float:
        """
        metoda zwracajaca mse lub rmse dla dwoch obrazow
        """
        n = 256
        this_histogram = self.histogram()
        other_histogram = other.histogram()

        if method == 1:
            mse = 0
            for i in range(n):
                mse = (i.this_histogram - i.other_histogram) ** 2  # histogram zamiast image i other
            mse = mse / n
            return mse
        if method == 2:
            rmse = 0
            for i in range(n):
                rmse = (i.this_histogram - i.other_histogram) ** 2
            rmse /= n
            return math.sqrt(rmse)

    def to_cumulated(self) -> 'Histogram':
        """
        metoda zwracajaca histogram skumulowany na podstawie stanu wewnetrznego obiektu
        """
        image = self.data.copy()
        h = image.histogram()
        size = np.arange(image.size())
        hs = []
        for i in range(size):
            hs[i] = h[i] + hs[i - 1]
        return hs


image = ImageComparison("Lenna_(test_image).png")
image.histogram().plot()


class ImageAligning(BaseImage):
    """
    klasa odpowiadająca za wyrównywanie histogramu
    """

    def __init__(self) -> None:
        """
        inicjalizator ...
        """
        pass

    def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
        """
        metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        """
        image = self.data.copy()
        image = image.to_gray()
        if tail_elimination is True:
            """
            histogram = image.histogram().to_cumulated()
            max_tail = histogram[max_per] #needs to be lower than 5% of highest histogram points
            # size = np.shape(image)
            # max_per = 0.95 * size
            # min_per = 0.05 * size
            min_tail = histogram[min_per] #needs to be higher than 95% of lowerst histogram points
            for row in image:
                for pixel in row: 
                    exposure[pixel] = (exposure[pixel] - min_tail * 255/(max_tail-min_tail)
            # exposure into image
            """
        # elif tail_elimination is False:
            """
            histogram = image.histogram()
            max = np.amax(histogram)
            min = np.amin(histogram)
            for row in image:
                for pixel in row: 
                    exposure[pixel] = (exposure[pixel] - min * 255/(max-min)
            #exposure into image
            """
        # return image

class ImageFiltration:
    def conv_2d(self, image, kernel: np.ndarray, prefix: Optional[float] = None) -> BaseImage:
        """
        kernel: filtr w postaci tablicy numpy
        prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
        metoda zwroci obraz po procesie filtrowania
        """

        # if the image has colours, each layer needs to get filtered

        # kernel = np.array([])
        # filtered_image =

        pass


class Image(GrayScaleTransform, ImageComparison, ImageAligning, ImageFiltration):
    pass


"""image = BaseImage("4.2.06.tiff")
# plt.imshow(image.data)
# image.show_img()
image.get_layer(0).show_img()"""

# image = GrayScaleTransform("Lenna_(test_image).png")
# image.to_sepia(w=20)
# image.to_gray()