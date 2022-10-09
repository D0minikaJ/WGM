"""Wybrać 3 pierwsze elementy z ostatniej kolumny tablicy utworzonej w zadaniu 2, a następnie ułożyć z nich kolumnę"""

import numpy as np

print(np.arange(1, 26).reshape(5, 5)[-1, 0:3].reshape(3, 1))
