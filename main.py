import math
from turtle import width
import matplotlib.pyplot as plt
import numpy as np


def nearest_neighbour(image, ratio):
    img_width, img_height, c = image.shape

    width = int(img_width * ratio)
    height = int(img_height * ratio)

    x_ratio = img_width / width
    y_ratio = img_height / height

    resized = np.zeros([width, height, c])

    for i in range(width):
        for j in range(height):
            resized[i, j] = image[int(i * x_ratio), int(j * y_ratio)]

    return resized


def bilinear(image, ratio):
    # def linear_interpolation(x0, x1, weight):
    #     return x0 * (1 - weight_x) + x1 * weight

    # def bilinear_interpolation(a, b, c, d, weight_x, weight_y):
    #     return linear_interpolation(linear_interpolation(a, b, weight_x),
    #                                 linear_interpolation(c, d, weight_x),
    #                                 weight_y)

    img_height, img_width, c = image.shape

    width = img_width * ratio
    height = img_height * ratio

    resized = np.empty([height, width, c])

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    for i in range(height):
        for j in range(width):

            x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
            x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            a = image[y_l, x_l]
            b = image[y_l, x_h]
            c = image[y_h, x_l]
            d = image[y_h, x_h]

            pixel = a * (1 - x_weight) * (1 - y_weight) \
                + b * x_weight * (1 - y_weight) + \
                c * y_weight * (1 - x_weight) + \
                d * x_weight * y_weight

            resized[i][j] = pixel

    return resized


def bicubic(image, ratio):
    def u(s):
        if (abs(s) >= 0) & (abs(s) <= 1):
            return 1.5 * (abs(s)**3) - 2.5 * (abs(s)**2) + 1
        elif (abs(s) > 1) & (abs(s) <= 2):
            return -.5 * (abs(s)**3) + 2.5 * (abs(s)**2) - 4 * abs(s) + 2
        return 0

    def padding(img, H, W, C):
        zimg = np.zeros((H+4, W+4, C))
        zimg[2:H+2, 2:W+2, :C] = img
        # Pad the first/last two col and row
        zimg[2:H+2, 0:2, :C] = img[:, 0:1, :C]
        zimg[H+2:H+4, 2:W+2, :] = img[H-1:H, :, :]
        zimg[2:H+2, W+2:W+4, :] = img[:, W-1:W, :]
        zimg[0:2, 2:W+2, :C] = img[0:1, :, :C]
        # Pad the missing eight points
        zimg[0:2, 0:2, :C] = img[0, 0, :C]
        zimg[H+2:H+4, 0:2, :C] = img[H-1, 0, :C]
        zimg[H+2:H+4, W+2:W+4, :C] = img[H-1, W-1, :C]
        zimg[0:2, W+2:W+4, :C] = img[0, W-1, :C]
        return zimg

    H, W, C = image.shape
    img = padding(image, H, W, C)

    width = math.floor(H * ratio)
    height = math.floor(W * ratio)
    resized = np.zeros((width, height, C))

    for c in range(C):
        for j in range(height):
            for i in range(width):
                x = i / ratio + 2
                y = j / ratio + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1), u(x2), u(x3), u(x4)]])

                mat_m = np.matrix([[img[int(y-y1), int(x-x1), c], img[int(y-y2), int(x-x1), c],
                                    img[int(y+y3), int(x-x1), c], img[int(y+y4), int(x-x1), c]],
                                   [img[int(y-y1), int(x-x2), c], img[int(y-y2), int(x-x2), c],
                                    img[int(y+y3), int(x-x2), c], img[int(y+y4), int(x-x2), c]],
                                   [img[int(y-y1), int(x+x3), c], img[int(y-y2), int(x+x3), c],
                                    img[int(y+y3), int(x+x3), c], img[int(y+y4), int(x+x3), c]],
                                   [img[int(y-y1), int(x+x4), c], img[int(y-y2), int(x+x4), c],
                                    img[int(y+y3), int(x+x4), c], img[int(y+y4), int(x+x4), c]]])

                mat_r = np.matrix([[u(y1)], [u(y2)], [u(y3)], [u(y4)]])

                resized[j, i, c] = (mat_l @ mat_m) @ mat_r

    return resized


image = plt.imread("tux.png")
ratio = 2

f, axarr = plt.subplots(2, 2)
f.suptitle(f'{ratio=}')

axarr[0, 0].imshow(image)
axarr[0, 0].set_title('image')

axarr[0, 1].imshow(nearest_neighbour(image, ratio))
axarr[0, 1].set_title('nearest_neighbour')

axarr[1, 0].imshow(bilinear(image, ratio))
axarr[1, 0].set_title('bilinear_interpolation')

axarr[1, 1].imshow(bicubic(image, ratio))
axarr[1, 1].set_title('bicup_interpolation')

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()
