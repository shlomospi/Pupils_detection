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
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants
# ----------------------------------------GPU check---------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print("\n\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ----------------------------------------tkinter init-----------------------------------------------------------------

root = Tk()
root.withdraw()
root.attributes('-topmost', True)

# ----------------------------------------parsing-----------------------------------------------------------------------


def input_fn():
    Inp1 = np.random.normal(size=(1, 36, 64, 1)).astype(np.float32)
    yield (Inp1, )


def parse_args():

    inf_parser = argparse.ArgumentParser(description="fast inference pupil location")

    inf_parser.add_argument("-v", '--video',
                            type=str,
                            default="0",
                            help='video path')
    inf_parser.add_argument("--save",
                            type=bool,
                            default=True,
                            help="save the results?")
    inf_parser.add_argument("-l", "--load",
                            type=str,
                            default='saved',
                            help="Load by 'choice' or 'saved' model?")
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
                            default=["0"],
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

        assert type(args.video) is str or args.video == 0
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

    mode = "video" if args.video != "" else "images"
    if args.load == "choice":
        print("Waiting for saved model..")
        model_path = filedialog.askdirectory()  # Choose
    else:
        model_path = "TF_saved_model_TFTRT_FP32"
    try:
        saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
        # signature_keys = list(saved_model_loaded.signatures.keys())
        # print(signature_keys)
        infer = saved_model_loaded.signatures['serving_default']
        # print(infer.structured_outputs)
        # graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        # frozen_func = convert_to_constants.convert_variables_to_constants_v2(
        #     graph_func)

    except AttributeError:
        print("Failed to load the model.")

    if mode == "video":
        if args.video == "0":
            video_path = 0
        else:
            fileDir = os.path.dirname(os.path.realpath('__file__'))
            video_path = os.path.join(fileDir, args.video)
            print("Loading video file from: \n{}".format(video_path))
        vid = cv.VideoCapture(video_path)
        width = 244 # int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
        height = 172 # int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv.CAP_PROP_FPS))
        codec = cv.VideoWriter_fourcc(*'mp4v')
        cv.namedWindow("output", cv.WINDOW_NORMAL)
        cv.resizeWindow('output', 244, 172) # width, height)
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
            frame_new = img_preproccess(frame, res=res, thresh=thresh, binary=binary).astype("float32")
            frame_for_infer = frame_new.reshape(1, res[1], res[0], 1)
            t1 = time.time()
            frame_for_infer = tf.constant(frame_for_infer)
            pred = infer(frame_for_infer)

            x, y = pred['dense_1'][0]
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

            out_img = cv.putText(out_img, "FPS: {:.2f}".format(len(times) / sum(times)), (0, 30),
                                 cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            if args.save:
                out.write(out_img)

            cv.imshow('output', out_img)

            if cv.waitKey(1) == ord('q'):
                break

        if args.save:
            print("saving output at :\n", output_video_path)
        cv.destroyAllWindows()


if __name__ == '__main__':

    main()
