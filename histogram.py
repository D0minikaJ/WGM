from Lab2.Zadanie1 import BaseImage
from Lab3.Zadanie1 import Image
from Lab3.Zadanie1 import GrayScaleTransform
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.pyplot import imshow
from matplotlib.image import imsave
import math


class Histogram:
    """
    klasa reprezentujaca histogram danego obrazu
    """
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, values: np.ndarray) -> None:
        pass

    def plot(self) -> None:
        """
        metoda wyswietlajaca histogram na podstawie atrybutu values
        """
        temp = self.values
        plt.plot(temp)


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
        pass

    def compare_to(self, other: Image, method: ImageDiffMethod) -> float:
        """
        metoda zwracajaca mse lub rmse dla dwoch obrazow
        """
        """
        image1 = self.values
        image2 = other.values
        n = 256
        mse = (1/n) * (np.sum(image1 - image2)**2)
        
        """
        pass


class Image(GrayScaleTransform, ImageComparison):
    pass

