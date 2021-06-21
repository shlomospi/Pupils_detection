import cv2 as cv
import argparse
import os
import numpy as np
from image_utils import *
import utils
from tqdm import tqdm


def parsers():
    parser = argparse.ArgumentParser(description='data prep script')
    parser.add_argument('--video', help='Camera divide number. default camera0', default=0)
    parser.add_argument('--new_res', help='new res', default=[64, 32], nargs='+', type=int)
    parser.add_argument('--threshold', help=' threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax)',
                        nargs='+', default=(79//2, 284//2, 0, 255, 0, 107))

    args = parser.parse_args()
    low_H, high_H, low_S,  high_S, low_V, high_V = args.threshold
    x, y = args.new_res


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


def video_csv_to_np(videopath, csvpath, res=(64, 32), normalize=True,
                    tresh=(79//2, 284//2, 0, 255, 0, 107), binary=False, verbose=0):
    """

    :param binary:
    :param verbose:
    :param tresh:
    :param normalize:
    :param videopath:
    :param csvpath:
    :param res:
    :return:
    """
    cap = cv.VideoCapture(videopath)
    original_res = cap.get(3), cap.get(4)
    scale = original_res[0] / res[0], original_res[1] / res[1]
    low_H, high_H, low_S, high_S, low_V, high_V = tresh
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

            values = map(float, values)
            if normalize:
                values = normalize_coord(values, original_res)

            label_list.append([values[0], values[1]])
    label_list = np.asarray(label_list)
    if verbose >=2:
        print("label shape:        ", np.shape(label_list), type(label_list))

    #fourcc = cv.VideoWriter_fourcc(*'XVID')
    cap = cv.VideoCapture(videopath)
    scale = 0
    while True:

        ret, frame = cap.read()

        if ret:
            """frame = cv.resize(frame, res, fx=0, fy=0, interpolation=cv.INTER_CUBIC)
            frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            frame_new = cv.bitwise_and(frame_HSV, frame_HSV, mask = frame_threshold)
            frame_new = frame_new[:, :, 0]
            frame_array = np.asarray(frame_new)
            array_list.append(frame_array)
            """
            array_list.append(img_preprocess(frame, res=res, tresh=tresh, binary=binary))
        else:
            break

    array_list = np.asarray(array_list)
    if verbose >= 2:
        print("images array shape: ", np.shape(array_list), type(array_list))

    return label_list, array_list


def create_ir_data(tresh=(79//2, 284//2, 0, 255, 0, 107), binary=False):
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
    print("final dataset:")
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


def view_img_dataset(folderpath, verbose=0):
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
        original_res = image.shape[:2]
        values = label_line.strip().split(" ")
        values = map(float, values)
        image = cross_annotator(image, values, color=(0, 250, 0), size=2)
        show_img(image)


if __name__ == '__main__':

    view_img_dataset("images_dataset")


