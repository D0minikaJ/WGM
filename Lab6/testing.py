from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.image import imsave
import math

values = plt.imread("Lenna_(test_image).png")

R = values[:, :, 0]
G = values[:, :, 1]
B = values[:, :, 2]

print(G)

plt.hist(R.ravel(), range=(0, 257))
plt.hist(G, np.arange(1, 257))
plt.hist(B, np.arange(1, 257))

plt.show()