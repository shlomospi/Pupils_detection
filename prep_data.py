import cv2 as cv
import argparse
import os

import numpy as np

print(cv.__version__)
parser = argparse.ArgumentParser(description='data prep script')
parser.add_argument('--video', help='Camera divide number. default camera0', default=0)
parser.add_argument('--new_res', help='new res', default=[64, 32], nargs='+', type=int)
parser.add_argument('--threshold', help=' threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax)',
                    default=(79//2, 284//2, 0, 255, 0, 107))

args = parser.parse_args()
low_H, high_H, low_S,  high_S, low_V, high_V = args.threshold
x, y = args.new_res


def normalize_img(img, high_v):
    """

    :param img:
    :param high_v:
    :return:
    """
    return img/high_v


def normalize_coord(coords, img_shape):
    """

    :param coords:
    :param img_shape:
    :return:
    """
    x_coord, y_coord = coords
    x_max, y_max = img_shape

    return x_coord/x_max, y_coord/y_max


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


def video_csv_to_np(videopath, csvpath, res=(64, 32)):
    """

    :param videopath:
    :param csvpath:
    :param res:
    :return:
    """

    label_list = []
    array_list = []
    with open(csvpath, "r") as f:
        for line in f:
            #  values: [ x, y]
            values = line.strip().split(" ")
            label_list.append([values[0], values[1]])
    label_list = np.asarray(label_list)
    print("label shape:        ", np.shape(label_list), type(label_list))

    #fourcc = cv.VideoWriter_fourcc(*'XVID')
    cap = cv.VideoCapture(videopath)

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
    print("images array shape: ", np.shape(array_list), type(array_list))

    return label_list, array_list


def create_ir_data():
    label_mat = []
    img_mat = []
    for vid in [1,2,3,4,5,9,13,21,22,91,111,121]:
        txt_file = str(vid)+".txt"
        video_file = str(vid)+".avi"

        labels, images = video_csv_to_np_wrapper(data_path = "dsp-ip/data/images/IR videos/",
                                                 txt_file=txt_file, video_file=video_file)
        label_mat.append(labels)
        img_mat.append(images)

    label_mat = np.concatenate(np.asarray(label_mat))
    img_mat = np.concatenate(np.asarray(img_mat))
    print("fin")
    print("images array shape: ", np.shape(img_mat), type(img_mat))
    print("labels array shape: ", np.shape(label_mat), type(label_mat))


create_ir_data()



