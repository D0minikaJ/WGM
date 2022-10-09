"""Przygotować skrypt, który stworzy tensor zawierający losowe wartości całkowite, losowym wymiarze i losowym
rozmiarze każdego z wymiarów """

import numpy as np

dimension = np.random.randint(1, 4)
print(f'wymiar: {dimension}')

if dimension == 3:
    n = np.random.randint(1, 10)
    m = np.random.randint(1, 10)
    o = np.random.randint(1, 10)
    arr = np.random.randint(1, 100, size=(n, m, o))
    print(f'rozmiar: {n}x{m}x{o}')
    print(arr)
elif dimension == 2:
    n = np.random.randint(1, 10)
    m = np.random.randint(1, 10)
    arr = np.random.randint(1, 100, size=(n, m))
    print(f'rozmiar: {n}x{m}')
    print(arr)
else:
    n = np.random.randint(1, 10)
    arr = np.random.randint(1, 11, size=n)
    print(f'rozmiar: {n}')
    print(arr)
