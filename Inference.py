import os

import numpy as np
import tensorflow as tf
import argparse
import cv2 as cv
import time
from image_utils import *

# ----------------------------------------GPU check---------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ----------------------------------------parsing-----------------------------------------------------------------------


def parse_args():

    parser = argparse.ArgumentParser(description="inference pupil location")
    parser.add_argument("-i", "--image_folder", type=str, default="", help='image path')
    parser.add_argument("-v", '--video', type=str, default="", help='video path')

    parser.add_argument('--model',
                        default="BatchSize_64_Epochs_200_LearningRate_001_model_small_dataRGB_phaseretrain_saved_model",
                        help="relative path to saved model")

    return check_args(parser.parse_args())


def check_args(args):

    # --phase
    try:

        assert type(args.video) is str
    except ValueError:
        print('video path must be a string, instred got:\n{}\n{}'.format(args.video, type(args.video)))

    return args


def main():
    res = (64, 32)
    args = parse_args()
    low_H, high_H, low_S, high_S, low_V, high_V = 79 // 2, 284 // 2, 0, 255, 0, 107
    mode = "video" if args.video != "" else "images"
    model = tf.keras.models.load_model('saved_model')
    model.summary()

    if mode == "video":
        if args.video == "0":
            video_path = 0
        else:
            fileDir = os.path.dirname(os.path.realpath('__file__'))
            video_path = os.path.join(fileDir, args.video)
        print("Loading video file from: \n{}".format(video_path))
        vid = cv.VideoCapture(video_path)
        width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv.CAP_PROP_FPS))
        codec = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter("output_inference.avi", codec, fps, (width, height))
        out_proccessed = cv.VideoWriter("output_inference_proccessed.avi", codec, fps, (width, height))
        cv.namedWindow("output", cv.WINDOW_NORMAL)
        cv.resizeWindow('output', 900, 600)
        # cv.namedWindow("outputp", cv.WINDOW_NORMAL)
        # cv.resizeWindow('outputp', 900, 600)
        times = []

        while True:
            _, frame = vid.read()

            if frame is None:
                print("Empty Frame")
                time.sleep(0.1)
                continue
            # original_res = (vid.get(4), vid.get(3)) # x, y

            frame_resized = cv.resize(frame, res, fx=0, fy=0, interpolation=cv.INTER_CUBIC)
            frame_HSV = cv.cvtColor(frame_resized, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            frame_new = cv.bitwise_and(frame_HSV, frame_HSV, mask=frame_threshold)
            frame_new = frame_new[:, :, 0]
            frame_for_infer = frame_new.reshape(1, 32, 64, 1)
            t1 = time.time()
            pred = model.predict(frame_for_infer)
            x, y = pred[0]
            t2 = time.time()
            times.append(t2 - t1)
            times = times[-10:]

            # coords = denormalize_coord((x, y), original_res)
            # frame_new = cv.resize(frame_new, (900, 600), fx=0, fy=0, interpolation=cv.INTER_CUBIC)  # temp
            out_img = cross_annotator(frame, (x, y), size = frame.shape[0]//50)
            # out_proccessed = cross_annotator(frame_new, (x, y), size=frame.shape[0] // 50)
            out_img = cv.putText(out_img, "FPS: {:.2f} Pred: {}".format(len(times) / sum(times), (x, y)), (0, 30),
                                 cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            out.write(out_img)

            cv.imshow('output', out_img)

            # cv.imshow('outputp', out_proccessed)
            if cv.waitKey(1) == ord('q'):
                break

        cv.destroyAllWindows()

    # if mode == "images": # TODO create image inference mode
        # folder_path = args.image_folder
        # fileDir = os.path.dirname(os.path.realpath('__file__'))
        # path = os.path.join(fileDir, folder_path)


if __name__ == '__main__':

    main()
