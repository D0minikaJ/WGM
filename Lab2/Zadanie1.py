from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.pyplot import imshow
from matplotlib.image import imsave
import math
from matplotlib.colors import hsv_to_rgb


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
        imshow(self.data)

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
                R = int(pixel[0])
                G = int(pixel[1])
                B = int(pixel[2])
                if math.sqrt(float(R * R + G * G + B * B - R * G - R * B - G * B)) == 0.0:
                    H = 0
                elif G >= B:
                    H = int(math.acos(
                        (R - 0.5 * G - 0.5 * B) / (math.sqrt(float(R * R + G * G + B * B - R * G - R * B - G * B)))))
                else:
                    H = int(360 - math.acos(
                        (R - 0.5 * G - 0.5 * B) / (math.sqrt(float(R * R + G * G + B * B - R * G - R * B - G * B)))))
                M = np.amax(row)
                m = np.amin(row)
                V = M / 255  # B
                if M > 0:
                    S = 1 - m / M  # G
                else:
                    S = 0  # G
                pixel = [H, S, V]
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
                R = int(pixel[0])
                G = int(pixel[1])
                B = int(pixel[2])
                if math.sqrt(float(R * R + G * G + B * B - R * G - R * B - G * B)) == 0.0:
                    H = 0
                elif G >= B:
                    H = int(math.acos(
                        (R - 0.5 * G - 0.5 * B) / (math.sqrt(float(R * R + G * G + B * B - R * G - R * B - G * B)))))
                else:
                    H = int(360 - math.acos(
                        (R - 0.5 * G - 0.5 * B) / (math.sqrt(float(R * R + G * G + B * B - R * G - R * B - G * B)))))
                M = np.amax(row)
                m = np.amin(row)
                I = (R + G + B) / 3  # B
                if M > 0:
                    S = 1 - m / M  # G
                else:
                    S = 0  # G
                pixel = [H, S, I]
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
                R = int(pixel[0])
                G = int(pixel[1])
                B = int(pixel[2])
                if math.sqrt(float(R * R + G * G + B * B - R * G - R * B - G * B)) == 0.0:
                    H = 0
                elif G >= B:
                    H = int(math.acos(
                        (R - 0.5 * G - 0.5 * B) / (math.sqrt(float(R * R + G * G + B * B - R * G - R * B - G * B)))))
                else:
                    H = int(360 - math.acos(
                        (R - 0.5 * G - 0.5 * B) / (math.sqrt(float(R * R + G * G + B * B - R * G - R * B - G * B)))))
                M = np.amax(row)
                m = np.amin(row)
                L = (0.5 * (M + m)) / 255  # B
                if M > 0:
                    S = 1 - m / M  # G
                else:
                    S = 0  # G
                pixel = [H, S, L]
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
            for row in self.data:
                new_row = []
                for pixel in row:
                    H = pixel[0]
                    S = pixel[1]
                    V = pixel[2]
                    M = 255*V
                    m = M*(1-S)
                    z = (M-m)*(1-abs(((H/60) % 2)-1))
                    if 0 <= H < 60:
                        R = M
                        G = z + m
                        B = m
                    if 60 <= H < 120:
                        R = z + m
                        G = M
                        B = m
                    if 120 <= H < 180:
                        R = m
                        G = M
                        B = z + m
                    if 180 <= H < 240:
                        R = m
                        G = M
                        B = z + m
                    if 240 <= H < 300:
                        R = z + m
                        G = m
                        B = M
                    if 300 <= H < 3060:
                        R = M
                        G = m
                        B = z + m
                    pixel = [R, G, B]
                    new_row.append(pixel)
                new_array.append(new_row)
            self.data = np.asarray(new_array)
            return self
        if self.color_model == 2:
            new_array = []
            for row in self.data:
                new_row = []
                for pixel in row:
                    H = pixel[0]
                    S = pixel[1]
                    I = pixel[2]
                    if H == 0:
                        R = I + 2*I*S
                        G = I - I*S
                        B = I - I*S
                    if 0 < H < 120:
                        R = I + I*S*math.cos(H)/math.cos(60-H)
                        G = I + I*S*(1-math.cos(H)/math.cos(60-H))
                        B = I - I*S
                    if H == 120:
                        R = I - I*S
                        G = I + 2*I*S
                        B = I - I*S
                    if 120 < H < 240:
                        R = I - I*S
                        G = I + I*S*math.cos(H-120)/math.cos(180-H)
                        B = I + I*S*(1-math.cos(H-120)/math.cos(180-H))
                    if H == 240:
                        R = I - I*S
                        G = I - I*S
                        B = I + 2*I*S
                    if 240 < H < 360:
                        R = I + I*S(1-math.cos(H-240)/math.cos(300-H))
                        G = I - I*S
                        B = I + I*S*math.cos(H-240)/math.cos(300-H)
                    pixel = [R, G, B]
                    new_row.append(pixel)
                new_array.append(new_row)
            self.data = np.asarray(new_array)
            return self
        if self.color_model == 3:
            new_array = []
            for row in self.data:
                new_row = []
                for pixel in row:
                    H = pixel[0]
                    S = pixel[1]
                    L = pixel[2]
                    d = S * (1 - abs(2 * L - 1))
                    m = 255 * (L - 0.5 * d)
                    x = d * (1 - abs(((H / 60) % 2) - 1))
                    if 0 <= H < 60:
                        R = 255 * d + m
                        G = 255 * x + m
                        B = m
                    if 60 <= H < 120:
                        R = 255 * x * m
                        G = 255 * d + m
                        B = m
                    if 120 <= H < 180:
                        R = m
                        G = 255 * d + m
                        B = 255 * x + m
                    if 180 <= H < 240:
                        R = m
                        G = 255 * x + m
                        B = 255 * d + m
                    if 240 <= H < 300:
                        R = 255 * x + m
                        G = m
                        B = 255 * d + m
                    if 300 <= H < 360:
                        R = 255 * d + m
                        G = m
                        B = 255 * x + m
                    pixel = [R, G, B]
                    new_row.append(pixel)
                new_array.append(new_row)
            self.data = np.asarray(new_array)
            return self


image = BaseImage("4.2.06.tiff")
# plt.imshow(image.data)
# image.show_img()
image.to_hsi().show_img()


plt.show()

"""
class GrayScaleTransform(BaseImage):

    def __init__(self) -> None:
        pass

    def to_gray(self) -> BaseImage:
        
        metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        
        inputImage = self.data

        grayImage = np.zeros(inputImage.shape, dtype=int)
        R = np.array(inputImage.data[:, :, 0])
        G = np.array(inputImage.data[:, :, 1])
        B = np.array(inputImage.data[:, :, 2])

        average = (R+G+B)

        for i in range(3):
            grayImage[:, :, i] = average

        return self

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        
        metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta
        lub metoda 2 w przypadku przekazania argumentu w
        
        pass

class Image(GrayScaleTransform):
    
    klasa stanowiaca glowny interfejs biblioteki
    w pozniejszym czasie bedzie dziedziczyla po kolejnych klasach
    realizujacych kolejne metody przetwarzania obrazow
    """
    
    # def __init__(self) -> None:
        #pass