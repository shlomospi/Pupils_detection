import numpy as np
import cv2 as cv


def cross_annotator(img, coord, color= (0, 250, 0), size=2):
    """
    adds cross to img
    :param img:
    :param coord:
    :param color:
    :return:
    """

    c = 1
    if np.ndim(img) == 2:
        img = np.expand_dims(img, -1)
    elif np.ndim(img) == 3:
        c = img.shape[2]

    if c == 1:
        img = np.concatenate((img, img, img), axis=2)

    x, y = coord
    if x <= 1: # Denormalize
        x *= img.shape[1]
        y *= img.shape[0]

    l1xs = int(x - size)
    l1ys = int(y)
    l1xe = int(x + size)
    l1ye = int(y)

    l2xs = int(x)
    l2ys = int(y - size)
    l2xe = int(x)
    l2ye = int(y + size)

    img = cv.line(img, (l1xs, l1ys), (l1xe, l1ye), color, 1)
    img = cv.line(img, (l2xs, l2ys), (l2xe, l2ye), color, 1)

    return img


def show_img(img):

    cv.namedWindow("image")
    cv.imshow("image", img)
    cv.waitKey(0)


