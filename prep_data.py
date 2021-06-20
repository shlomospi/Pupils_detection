import cv2 as cv
import argparse
import os
import numpy as np
from image_utils import *
import utils

print(cv.__version__)
low_H, high_H, low_S,  high_S, low_V, high_V = 79//2, 284//2, 0, 255, 0, 107
x, y = 64, 32
def parsers():
    parser = argparse.ArgumentParser(description='data prep script')
    parser.add_argument('--video', help='Camera divide number. default camera0', default=0)
    parser.add_argument('--new_res', help='new res', default=[64, 32], nargs='+', type=int)
    parser.add_argument('--threshold', help=' threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax)',
                        default=(79//2, 284//2, 0, 255, 0, 107))

    args = parser.parse_args()
    low_H, high_H, low_S,  high_S, low_V, high_V = args.threshold
    x, y = args.new_res


def video_csv_to_np_wrapper(data_path = "", txt_file = "1.txt", video_file = "1.avi"):
    """

    :param data_path:
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

    return video_csv_to_np(video_path, txt_path, res=(64, 32))


def video_csv_to_np(videopath, csvpath, res=(64, 32), normalize=True, verbose=0):
    """

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

            frame = cv.resize(frame, res, fx=0, fy=0, interpolation=cv.INTER_CUBIC)
            frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            frame_new = cv.bitwise_and(frame_HSV, frame_HSV, mask = frame_threshold)
            frame_new = frame_new[:, :, 0]
            frame_array = np.asarray(frame_new)
            array_list.append(frame_array)
        else:
            break

    array_list = np.asarray(array_list)
    if verbose >= 2:
        print("images array shape: ", np.shape(array_list), type(array_list))

    return label_list, array_list


def create_ir_data():
    """

    :return:
    """
    label_mat = []
    img_mat = []
    for vid in [1,2,3,4,5,9,13,21,22,91,111,121]:
        txt_file = str(vid)+".txt"
        video_file = str(vid)+".avi"

        labels_, images = video_csv_to_np_wrapper(data_path = "dsp-ip/data/images/IR videos/",
                                                  txt_file=txt_file, video_file=video_file)
        label_mat.append(labels_)
        img_mat.append(images)

    label_mat = np.concatenate(np.asarray(label_mat))
    img_mat = np.concatenate(np.asarray(img_mat))
    print("final dataset:")
    print("images array shape: ", np.shape(img_mat), type(img_mat))
    print("labels array shape: ", np.shape(label_mat), type(label_mat))
    return img_mat, label_mat


def create_RGB_data(res=(64, 32), normalize=True, verbose=0):

    return img_txt_to_np("inferno/images_dataset", res=res, normalize=normalize, verbose=verbose)


def img_txt_to_np(folderpath, res=(64, 32), normalize=True, verbose=0):
    """

    :param folderpath:
    :param res:
    :param normalize:
    :param verbose:
    :return:
    """

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(fileDir, folderpath)
    labelpath = os.path.join(path, "labels.txt")
    print("Loading data from folder path:\n", path)
    label_list = []
    array_list = []
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

        # img handling
        frame = cv.resize(image, res, fx=0, fy=0, interpolation=cv.INTER_CUBIC)
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        frame_new = cv.bitwise_and(frame_HSV, frame_HSV, mask=frame_threshold)
        frame_new = frame_new[:, :, 0]
        frame_array = np.asarray(frame_new)
        array_list.append(frame_array)

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


def main():
    parsers()

if __name__ == '__main__':

    view_img_dataset("images_dataset")


