"""Wyznaczyć sumę wartości elementów znajdujących się w dwóch ostatnich wierszach macierzy utworzonej w zadaniu 2"""

import numpy as np

print(np.sum(np.arange(1, 26).reshape(5, 5)[-2:, :]))