import cv2 as cv
import argparse
import os
import numpy as np
from image_utils import *
import utils
from tqdm import tqdm
import tensorflow as tf
# import tensorflow_addons as tfa
import image_utils


def video_csv_to_np_wrapper(data_path = "", txt_file = "1.txt", video_file = "1.avi",
                            tresh=(79//2, 284//2, 0, 255, 0, 107), binary=False):
    """

    :param binary:
    :param tresh:
    :param data_path:q
    :param txt_file:
    :param video_file:
    :return:
    """
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    txt_path = os.path.join(fileDir, data_path, txt_file)
    video_path = os.path.join(fileDir, data_path, video_file)
    if not os.path.isfile(txt_path):
        print(txt_path)
        raise FileNotFoundError

    if not os.path.isfile(video_path):
        print(video_path)
        raise FileNotFoundError

    return video_csv_to_np(video_path, txt_path, res=(64, 32), tresh=tresh, binary=binary)


def video_csv_to_np(videopath, csvpath, res=(64, 32), augment=False,
                    tresh=(79//2, 284//2, 0, 255, 0, 107), binary=False, verbose=0):
    """

    :param augment:
    :param binary:
    :param verbose:
    :param tresh:
    :param videopath:
    :param csvpath:
    :param res:
    :return:
    """
    cap = cv.VideoCapture(videopath)
    original_res = cap.get(3), cap.get(4)
    scale = original_res[0] / res[0], original_res[1] / res[1]
    if verbose >= 1:
        print("Original res: {}".format(original_res))
        print("Resize Scale: {}".format(scale))
        print("New Res :{}".format(res))

    label_list = []
    array_list = []
    with open(csvpath, "r") as f:
        for line in f:
            #  values: [ x, y]
            values = line.strip().split(" ")
            values = list(map(float, values))

            if not augment:
                values = normalize_coord(values, original_res)

            label_list.append([values[0], values[1]])
    label_list = np.asarray(label_list)
    if verbose >= 2:
        print("label shape:        ", np.shape(label_list), type(label_list))

    cap = cv.VideoCapture(videopath)

    while True:

        ret, frame = cap.read()

        if ret:

            array_list.append(img_preprocess(frame, res=res,
                                             tresh=tresh, binary=binary))
        else:
            break

    array_list = np.asarray(array_list)

    if verbose >= 2:
        print("images array shape: ", np.shape(array_list), type(array_list))

    return label_list, array_list


def create_ir_data(tresh=(79//2, 284//2, 0, 255, 0, 107), binary=False,
                   verbose=2):
    """

    :return:
    """
    label_mat = []
    img_mat = []
    for vid in [1, 2, 3, 4, 5, 9, 13, 21, 22, 91, 111, 121]:
        txt_file = str(vid)+".txt"
        video_file = str(vid)+".avi"

        labels_, images = video_csv_to_np_wrapper(data_path = "dsp-ip/data/images/IR videos/",
                                                  txt_file=txt_file, video_file=video_file,
                                                  tresh=tresh, binary=binary)
        label_mat.append(labels_)
        img_mat.append(images)

    label_mat = np.concatenate(np.asarray(label_mat))
    img_mat = np.concatenate(np.asarray(img_mat))
    if verbose >= 2:
        print("IR dataset:")
        print("images array shape: ", np.shape(img_mat), type(img_mat))
        print("labels array shape: ", np.shape(label_mat), type(label_mat))
    return img_mat, label_mat


def create_RGB_data(res=(64, 32), normalize=True, tresh=(79//2, 284//2, 0, 255, 0, 107),
                    binary=False, verbose=0):

    return img_txt_to_np("inferno/images_dataset", res=res, normalize=normalize,
                         binary=binary, tresh=tresh, verbose=verbose)


def create_RGB2_data(res=(64, 32), normalize=True, tresh=(79//2, 284//2, 0, 255, 0, 107),
                     binary=False, verbose=0):

    return img_txt_to_np("inferno/images_from_videos1_dataset", res=res, normalize=normalize,
                         binary=binary, tresh=tresh, verbose=verbose)


def img_preprocess(image, res=(64, 32), tresh=(79//2, 284//2, 0, 255, 0, 107), binary=False):
    low_H, high_H, low_S,  high_S, low_V, high_V = tresh
    resized_frame = cv.resize(image, res, fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    frame_HSV = cv.cvtColor(resized_frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    frame_new = cv.bitwise_and(frame_HSV, frame_HSV, mask=frame_threshold)
    if binary:
        _, frame_new = cv.threshold(frame_new, 0, 255, cv.THRESH_BINARY)
    frame_new = frame_new[:, :, 0]
    frame_array = np.asarray(frame_new)

    return frame_array


def img_txt_to_np(folderpath, res=(64, 32), tresh=(79//2, 284//2, 0, 255, 0, 107),
                  normalize=True, binary=False, verbose=0):
    """

    :param binary: convert to binary or leave in grayscale
    :param tresh: treshhold to use on HSV format
    :param folderpath:
    :param res: final resolution of images
    :param normalize: Normalize coordinates to 0-1 or not
    :param verbose: 2 - to print out dataset size in the end
    :return: array_list, label_list : lists of image tensors
    """

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(fileDir, folderpath)
    labelpath = os.path.join(path, "labels.txt")
    print("Loading data from:\n", path)
    label_list = []
    array_list = []
    if not os.path.isfile(labelpath):
        print(labelpath)
        raise FileNotFoundError

    with open(labelpath) as f:
        img_count = 0
        for _ in f:
            img_count += 1

    for image_num in tqdm(range(1, img_count+1)):

        image_path = os.path.join(path, "{}.jpg".format(image_num))
        if not os.path.isfile(image_path):
            print(image_path)
            raise FileNotFoundError
        image = cv.imread(image_path, cv.IMREAD_COLOR)

        # img handling
        """frame = cv.resize(image, res, fx=0, fy=0, interpolation=cv.INTER_CUBIC)
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        frame_new = cv.bitwise_and(frame_HSV, frame_HSV, mask=frame_threshold)
        frame_new = frame_new[:, :, 0]
        frame_array = np.asarray(frame_new)
        array_list.append(frame_array)"""
        array_list.append(img_preprocess(image, res=res, tresh=tresh, binary=binary))
        # label handling
        label_line = utils.read_selected_line(labelpath, image_num)
        # original_res = image.shape[:2] # y, x
        original_res = [image.shape[1], image.shape[0]] # x, y
        values = label_line.strip().split(" ")
        values = map(float, values)
        if normalize:
            values = normalize_coord(values, original_res)

        label_list.append([values[0], values[1]])

    label_list = np.asarray(label_list)
    if verbose >= 2:
        print("label shape:        ", np.shape(label_list), type(label_list))

    array_list = np.asarray(array_list)
    if verbose >= 2:
        print("images array shape: ", np.shape(array_list), type(array_list))

    return array_list, label_list


def view_img_dataset(folderpath):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(fileDir, folderpath)
    labelpath = os.path.join(path, "labels.txt")
    print("Loading data from folder path:\n", path)
    if not os.path.isfile(labelpath):
        print(labelpath)
        raise FileNotFoundError

    with open(labelpath) as f:
        img_count = 0
        for _ in f:
            img_count += 1

    for image_num in range(1, img_count+1):
        image_path = os.path.join(path, "{}.jpg".format(image_num))
        if not os.path.isfile(image_path):
            print(image_path)
            raise FileNotFoundError
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = np.asarray(image)
        label_line = utils.read_selected_line(labelpath, image_num)
        # original_res = image.shape[:2]
        values = label_line.strip().split(" ")
        values = map(float, values)
        image = cross_annotator(image, values, color=(0, 250, 0), size=2)
        show_img(image)


def flipLR_img_landmark(x_train, y_train):
    """
    flips and adds flipped images to dataset with one landmark
    :param x_train:
    :param y_train:
    :return:
    """
    if y_train[0, 0] <= 1 and y_train[0, 1] <= 1:
        x_flipedlr = tf.image.flip_left_right(x_train)
        y_flipedlr = np.subtract(1, y_train)
        x_train = np.concatenate((x_train, x_flipedlr), axis=0)
        y_train = np.concatenate((y_train, y_flipedlr), axis=0)

        return x_train, y_train

    else:
        print(y_train[0])
        raise SystemExit("only normalized landmarks are supported.")


def translate_img_landmark_once(x_train, y_train, t_x=3, t_y=2):
    assert x_train.ndim == 4

    img_shape = x_train.shape
    if t_x < 1 and t_y < 1:  # convert to pixels
        shiftX = t_x * img_shape[2]
        shiftY = t_y * img_shape[1]
    else:
        shiftX, shiftY = t_x, t_y
    if y_train[0, 0] < 1 and y_train[0, 1] < 1:  # convert to percent
        trans_vec = np.array([[shiftX/img_shape[2], shiftY/img_shape[1]]])
    else:
        trans_vec = np.array([[shiftX, shiftY]])
    M = np.float32([
        [1, 0, shiftX],
        [0, 1, shiftY]])
    if y_train[0, 0] <= 1 and y_train[0, 1] <= 1:
        shifted_imgs = []
        for image in x_train:
            shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
            shifted_imgs.append(shifted)

        trans_x = np.expand_dims(np.asarray(shifted_imgs), axis = -1)
        trans_y = np.add(y_train, trans_vec)
        return trans_x, trans_y

    else:
        raise SystemExit("only normalized landmarks are supported.")


def translate_img_landmark(x_train, y_train, max_x=8, max_y=2, iterations=4):

    assert x_train.ndim == 4

    if y_train[0, 0] <= 1 and y_train[0, 1] <= 1:
        randx = np.random.randint(-max_x, max_x + 1, size=iterations)
        randy = np.random.randint(-max_y, max_y + 1, size=iterations)
        x_res = []
        y_res = []
        for i in range(iterations):
            trans_x, trans_y = translate_img_landmark_once(x_train, y_train,
                                                           t_x=randx[i], t_y=randy[i])
            # testx, testy = translate_img_landmark_once(trans_x[:5], trans_y[:5])
            # image_utils.plot_example_images(testx, testy,
            #                                 title="trans " + str(i))
            x_res.append(trans_x)
            y_res.append(trans_y)
        for new_datax in x_res:
            x_train = np.concatenate((x_train, new_datax), axis=0)
        for new_datay in y_res:
            y_train = np.concatenate((y_train, new_datay), axis=0)

        return x_train, y_train

    else:
        print(y_train[0])
        cv.imshow("tmp", x_train[0])
        raise SystemExit("only normalized landmarks are supported.")


if __name__ == '__main__':

    view_img_dataset("images_dataset")
