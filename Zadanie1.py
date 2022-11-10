from Lab2.Zadanie1 import BaseImage
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.pyplot import imshow
from matplotlib.image import imsave
import math
from matplotlib.colors import hsv_to_rgb

class GrayScaleTransform(BaseImage):
    def __init__(self) -> None:
        pass

    def to_gray(self) -> BaseImage:
        """
        metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        """
        r_layer = self.data[:, :, 0]
        g_layer = self.data[:, :, 1]
        b_layer = self.data[:, :, 2]
        img_layer = np.zeros(self.data.shape, dtype=int)
        avg = (r_layer+g_layer+b_layer)

        for i in range(3):
            img_layer[:,:,i] = avg

        self.data = img_layer
        return self

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        """
        metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta
        lub metoda 2 w przypadku przekazania argumentu w
        """
        """
        temp = self.data
        if alpha_beta is not None:
            alpha = alpha_beta[0]
            beta = alpha_beta[1]
            if alpha > 1 and beta < 1 and alpha + beta = 2:
                temp.to_gray()
                L0 = temp[:, :, 0]
                L1 = temp[:, :, 1]
                L2 = temp[:, :, 2]
                new_array = []
                for row in self.data:
                    new_row = []
                    for pixel in row:
                        L0 = int(pixel[0])
                        L1 = int(pixel[1])
                        L2 = int(pixel[2]) 
                        L0 = L0 * alpha
                        L2 = L2 * beta
                        pixel = [L0, L1, L2]
                        new_row.append(pixel)
                    new_array.append(new_row)
                self.data = np.asarray(new_array)
        if w is not None:
            if w >= 20 and w <= 40:
                temp.to_gray()
                L0 = temp[:, :, 0]
                L1 = temp[:, :, 1]
                L2 = temp[:, :, 2]
                new_array = []
                for row in self.data:
                    new_row = []
                    for pixel in row:
                        L0 = int(pixel[0])
                        L1 = int(pixel[1])
                        L2 = int(pixel[2]) 
                        L0 = L0 + 2 * w
                        L1 = L1 + w
                        pixel = [L0, L1, L2]
                        new_row.append(pixel)
                    new_array.append(new_row)
                self.data = np.asarray(new_array)
        """
        pass

class Image(GrayScaleTransform):
    """
    klasa stanowiaca glowny interfejs biblioteki
    w pozniejszym czasie bedzie dziedziczyla po kolejnych klasach
    realizujacych kolejne metody przetwarzania obrazow
    """
    def __init__(self) -> None:
        pass

image = BaseImage("4.2.06.tiff")
# plt.imshow(image.data)
# image.show_img()
# image.get_layer(0).show_img()
image.to_gray().show_img()