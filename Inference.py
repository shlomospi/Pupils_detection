import os

import numpy as np
import tensorflow as tf
import argparse
import cv2 as cv
import time
from image_utils import *
from tkinter import Tk, filedialog
from prep_data import *
from config import config

# ----------------------------------------GPU check---------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ----------------------------------------tkinter init-----------------------------------------------------------------

root = Tk()
root.withdraw()
root.attributes('-topmost', True)

# ----------------------------------------parsing-----------------------------------------------------------------------


def parse_args():

    inf_parser = argparse.ArgumentParser(description="fast inference pupil location")
    # inf_parser.add_argument("-i", "--image_folder", type=str, default="", help='image path') # TODO
    inf_parser.add_argument("-v", '--video',
                            type=str,
                            default="",
                            help='video path')
    inf_parser.add_argument("--save",
                            type=bool,
                            default=True,
                            help="save the results?")
    inf_parser.add_argument("-a", "--average",
                            type=int,
                            default=1,
                            help="How many frames back to average prediction")
    inf_parser.add_argument("-o", '--output',
                            type=str,
                            default="original",
                            help='"original" to show inference on original image, '
                                 '"preproccessed" to show on preproccessed frames')
    inf_parser.add_argument('-t', '--threshold',
                            nargs='+',
                            default=1,
                            help=' threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax) for '
                                 'image preproccessing. or an int for picked values for dictionary')
    inf_parser.add_argument('-bin', '--binary',
                            default=False,
                            type=bool,
                            help="'-bin True' for converting data to binary pixels, "
                                 "ignore for False ")
    return check_args(inf_parser.parse_args())


def check_args(args):

    # --phase
    try:

        assert type(args.video) is str
    except ValueError:
        print('video path must be a string, instred got:\n{}\n{}'.format(args.video, type(args.video)))

    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    res = config["res"]
    args = parse_args()
    binary = args.binary
    if len(args.threshold) == 6:
        thresh = args.threshold
    else:
        assert len(args.threshold) == 1
        thresh = config["thresholds"][args.threshold[0]]

    low_H, high_H, low_S, high_S, low_V, high_V = thresh # 79 // 2, 284 // 2, 0, 255, 0, 107
    mode = "video" if args.video != "" else "images"
    print("Waiting for saved model..")
    model_path = filedialog.askdirectory()  # Choose
    try:
        model = tf.keras.models.load_model(model_path)
        model.summary()
    except AttributeError:
        print("The folder does not contain a saved model.\n "
              "Choose log/<training session>/final_model")

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
        cv.namedWindow("output", cv.WINDOW_NORMAL)
        cv.resizeWindow('output', width, height)
        if args.average > 1:
            output_video_path = os.path.join(model_path,
                                             "output_{}_inference_avg{}_{}.mp4".format(args.video[:-4],
                                                                                       args.average,
                                                                                       args.output))
        else:
            output_video_path = os.path.join(model_path, "output_{}_inference_{}.mp4".format(args.video[:-4],
                                                                                             args.output))
        out = cv.VideoWriter(output_video_path, codec, fps, (width, height))

        times = []
        predictions_x = []
        predictions_y = []
        while True:
            _, frame = vid.read()

            if frame is None:
                break

            # original_res = (vid.get(4), vid.get(3)) # x, y
            frame_new = img_preproccess(frame, res=res, thresh=thresh, binary=binary)

            frame_for_infer = frame_new.reshape(1, res[1], res[0], 1)
            t1 = time.time()
            pred = model.predict(frame_for_infer)
            x, y = pred[0]
            t2 = time.time()
            times.append(t2 - t1)
            times = times[-10:]
            if args.average > 1:
                predictions_x.append(x)
                predictions_y.append(y)
                predictions_x = predictions_x[-args.average:]
                predictions_y = predictions_y[-args.average:]
                x = sum(predictions_x)/len(predictions_x)
                y = sum(predictions_y)/len(predictions_y)

            if args.output == "original":
                out_img = cross_annotator(frame, (x, y), size = frame.shape[0] // 50)
            elif args.output == "preproccessed":
                frame_for_infer = np.squeeze(frame_for_infer)
                out_img = cross_annotator(frame_for_infer, (x, y), size=2)
                out_img = cv.resize(out_img, (width, height), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
            else: # replace with typo catcher
                out_img = cross_annotator(frame, (x, y), size=frame.shape[0] // 50)

            out_img = cv.putText(out_img, "FPS: {:.2f} Pred: {}".format(len(times) / sum(times), (x, y)), (0, 30),
                                 cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            if args.save:
                out.write(out_img)

            cv.imshow('output', out_img)

            if cv.waitKey(1) == ord('q'):
                break

        if args.save:
            print("saving output at :\n", output_video_path)
        cv.destroyAllWindows()

    # if mode == "images": # TODO create image inference mode
        # folder_path = args.image_folder
        # fileDir = os.path.dirname(os.path.realpath('__file__'))
        # path = os.path.join(fileDir, folder_path)


if __name__ == '__main__':

    main()
