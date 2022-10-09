"""Wybrać podtablicę 12-elementową, z tablicy utworzonej w zadaniu 2, z wartościami w zakresie 12-25"""

import numpy as np

print(np.arange(1, 26).reshape(5, 5)[2:5, 1:5])
