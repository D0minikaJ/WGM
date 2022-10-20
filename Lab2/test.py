from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_arr = mpimg.imread('4.2.06.tiff')

print(img_arr)

img_plot = plt.imshow(img_arr)
plt.show()