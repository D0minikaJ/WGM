from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img_arr = mpimg.imread('4.2.06.tiff')

print(img_arr)

def get_layer(layer_id: int) -> 'BaseImage':
    r_layer, g_layer, b_layer = np.squeeze(np.dsplit(img_arr, img_arr.shape[-1]))
    f, ax_arr = plt.subplots(1,3)
    if layer_id == 0:
        ax_arr[0].imshow(r_layer, cmap='gray')
    if layer_id == 1:
        ax_arr[1].imshow(g_layer, cmap='gray')
    if layer_id == 2:
        ax_arr[2].imshow(b_layer, cmap='gray')

img_plot = plt.imshow(img_arr)
get_layer(1)
plt.show()
