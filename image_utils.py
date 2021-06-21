import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random


def cross_annotator(img, coord, color= (0, 250, 0), size=3):
    """
    adds cross to img
    :param size:
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

    img = cv.line(img, (l1xs, l1ys), (l1xe, l1ye), color, size//4+1)
    img = cv.line(img, (l2xs, l2ys), (l2xe, l2ye), color, size//3+1)

    return img


def show_img(img):
    """

    :param img:
    :return:
    """
    cv.namedWindow("image")
    cv.imshow("image", img)
    cv.waitKey(0)


def plot_example_images(images, labels, title="example of data", examples=10, folder=None):
    """

    :param title:
    :param folder:
    :param images:
    :param labels:
    :param examples:
    :return:
    """
    fig = plt.figure(figsize = (examples*2, 4))
    fig.suptitle(title)

    for image_num in range(1, examples+1):
        plt.subplot(1, examples, image_num)
        rand = random.randint(0, len(labels)-1)
        plt.imshow(cross_annotator(images[rand, :, :, :], labels[rand]))
        plt.axis('off')

    if folder is not None:
        plt.savefig('{}/{}.png'.format(folder, title))
    plt.show()


def denormalize_img(img, high_v):
    """
    from 0-1 to o-high_v
    :param img:
    :param high_v:
    :return:
    """
    return img/high_v


def normalize_img(img, high_v):
    """
    from 0-high_v to o-1
    :param img:
    :param high_v:
    :return:
    """
    return img/high_v


def denormalize_coord(coords, img_shape):
    """
    from 0-1 to 0-max pixel
    :param coords:
    :param img_shape:
    :return:
    """
    x_coord, y_coord = coords
    x_max, y_max = img_shape

    return x_coord*x_max, y_coord*y_max


def normalize_coord(coords, img_shape):
    """
    from 0-max pixel to 0-1
    :param coords:
    :param img_shape:
    :return:
    """
    x_coord, y_coord = coords
    x_max, y_max = img_shape

    return x_coord/x_max, y_coord/y_max
