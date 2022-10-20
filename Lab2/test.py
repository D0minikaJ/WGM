from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

img_arr = mpimg.imread('4.2.06.tiff')

print(img_arr)

def get_layer(layer_id: int):
    r_layer, g_layer, b_layer = np.squeeze(np.dsplit(img_arr, img_arr.shape[-1]))
    f, ax_arr = plt.subplots(1,3)
    if layer_id == 0:
        ax_arr[0].imshow(r_layer, cmap='gray')
    if layer_id == 1:
        ax_arr[1].imshow(g_layer, cmap='gray')
    if layer_id == 2:
        ax_arr[2].imshow(b_layer, cmap='gray')

"""
def to_hsv():
    for each array in for each row ->
    [101 0 0]
    R = 101
    G = 0
    B = 0
    if G >= B:
        H = math.acos((R - 0.5*G - 0.5*B)/(math.sqrt(R*R + G*G + B*B - R*G - R*B - G*B)))
    else:
        H = 360 - math.acos((R - 0.5*G - 0.5*B)/(math.sqrt(R*R + G*G + B*B - R*G - R*B - G*B)))
    M = np.amax(a, axis=1)
    m = np.amin(a, axis=1)
    V = M / 255 -> B
    B = V
    if M > 0:
        S = 1 - m/M -> G
        G = S
    else:
        S = 0 -> G
        G = S
    pass
    """

"""
def to_hsi():
    for each array in for each row ->
    [101 0 0]
    R = 101
    G = 0
    B = 0
    if G >= B:
        H = math.acos((R - 0.5*G - 0.5*B)/(math.sqrt(R*R + G*G + B*B - R*G - R*B - G*B)))
    else:
        H = 360 - math.acos((R - 0.5*G - 0.5*B)/(math.sqrt(R*R + G*G + B*B - R*G - R*B - G*B)))
    I = (R + G + B)/3 -> B
    B = I
    if M > 0:
        S = 1 - m/M -> G
        G = S
    else:
        S = 0 -> G
        G = S
    pass
"""

"""
def to_hsl():
    for each array in for each row ->
    example -> [101 0 0]
    pixel values would be:
    R = 101
    G = 0
    B = 0
    if G >= B:
        H = math.acos((R - 0.5*G - 0.5*B)/(math.sqrt(R*R + G*G + B*B - R*G - R*B - G*B)))
    else:
        H = 360 - math.acos((R - 0.5*G - 0.5*B)/(math.sqrt(R*R + G*G + B*B - R*G - R*B - G*B)))
    M = np.amax(a, axis=1)
    m = np.amin(a, axis=1)
    d = (M - m) / 255
    L = (0.5 * (M + m))/255
    B = L
    if M > 0:
        S = 1 - m/M -> G
        G = S
    else:
        S = 0 -> G
        G = S
    pass
"""
"""
def to_rgb(self):
    if color == hsv:
        #H -> R, S -> G, V -> B
        for each array in for each row ->
        example -> [101 0 0]
        pixel values would be:
        H = 101
        S = 0
        V = 0
        M = 255 * V
        m = M(1-S)
        z = (M - m)[1 - math.abs(((H/60)%2)-1)]
        if H >= 0 && H > 60:
            
            R = M
            G = z + m
            B = m
"""

img_plot = plt.imshow(img_arr)
get_layer(0)
plt.show()
