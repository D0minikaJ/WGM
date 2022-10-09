"""Utworzyć tablicę o rozmiarze 10x10 z wartościami zwiększającymi się o 0.01"""

import numpy as np

print(np.arange(0, 1, 0.01).reshape((10, 10)))
