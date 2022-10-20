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
                if math.sqrt(float(R*R + G*G + B*B - R*G - R*B - G*B)) == 0.0:
                    H = 0
                elif G >= B:
                    H = int(math.acos((R - 0.5*G - 0.5*B)/(math.sqrt(float(R*R + G*G + B*B - R*G - R*B - G*B)))))
                else:
                    H = int(360 - math.acos((R - 0.5*G - 0.5*B)/(math.sqrt(float(R*R + G*G + B*B - R*G - R*B - G*B)))))
                M = np.amax(row)
                m = np.amin(row)
                V = M / 255 #B
                if M > 0:
                    S = 1 - m/M #G
                else:
                    S = 0 #G
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
                I = (R + G + B)/3  # B
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
        if self.color_model == 3:
            new_array = []
            for row in self.data:
                new_row = []
                for pixel in row:
                    H = pixel[0]
                    S = pixel[1]
                    L = pixel[2]
                    d = S*(1 - abs(2*L -1))
                    m = 255*(L - 0.5 * d)
                    x = d*(1 - abs(((H/60)%2) - 1))
                    if H >= 0 and H < 60:
                        R = 255 * d + m
                        G = 255 * x + m
                        B = m
                    if H >= 60 and H < 120:
                        R = 255 * x * m
                        G = 255 * d + m
                        B = m
                    if H >= 120 and H < 180:
                        R = m
                        G = 255 * d + m
                        B = 255 * x + m
                    if H >= 180 and H < 240:
                        R = m
                        G = 255 * x + m
                        B = 255 * d + m
                    if H >= 240 and H < 300:
                        R = 255 * x + m
                        G = m
                        B = 255 * d + m
                    if H >=300 and H < 360:
                        R =255 * d + m
                        G = m
                        B = 255 * x + m
                    pixel = [R, G, B]
                    new_row.append(pixel)
                new_array.append(new_row)
            self.data = np.asarray(new_array)
            return self



image = BaseImage("4.2.06.tiff")
#plt.imshow(image.data)
#image.show_img()
#image.get_layer(0).show_img()


plt.show()
