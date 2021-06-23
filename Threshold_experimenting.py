from __future__ import print_function
import cv2 as cv
import argparse
import os

max_value = 255
max_value_H = 360 // 2
# low_H = 0
# low_S = 0
# low_V = 0
# high_H = max_value_H
# high_S = max_value
# high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)


parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('-v', '--video', help='video or 0 for camera', default=0)
parser.add_argument('-r', '--change_res', nargs='+', help='change the res?', default=[64, 32])
parser.add_argument('-t', '--threshold', help=' threshold (Hmin, Hmax, Smin, Smax, Vmin, Vmax)',
                    nargs='+',
                    default=(83, 135, 0, 255, 0, 162)) # (90//2, 270//2, 0, 255, 0, 100))

parser.add_argument('-bin', '--binary',
                    default=False,
                    type=bool,
                    help="'-bin True' for converting data to binary pixels, ignore for False ")

args = parser.parse_args()
low_H, high_H, low_S,  high_S, low_V, high_V = args.threshold
print("thresh:\n", low_H, high_H, low_S,  high_S, low_V, high_V)
video = 0 if args.video == 0 else os.path.join(os.path.dirname(os.path.realpath('__file__')), args.video)
cap = cv.VideoCapture(args.video)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)

cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
while True:

    ret, frame = cap.read()
    if frame is None:
        print("Reloading video, press 'q' to quit")
        cap = cv.VideoCapture(args.video)
        ret, frame = cap.read()
    if args.change_res is not None:
        res = list(map(int, args.change_res))
    #     frame = cv.resize(frame, (res[0], res[1]), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    frame_new = cv.bitwise_and(frame_HSV, frame_HSV, mask=frame_threshold)
    if args.change_res is not None:
        res = list(map(int, args.change_res))
        frame_new = cv.resize(frame_new, (res[0], res[1]), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    if args.binary:
        _, frame_new = cv.threshold(frame_new, 0, 255, cv.THRESH_BINARY)
    frame_new = frame_new[:, :, 0]
    if args.change_res is not None:
        res = list(map(int, args.change_res))
        frame = cv.resize(frame, (640, 320), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
        frame_new = cv.resize(frame_new, (640, 320), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_new)

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
