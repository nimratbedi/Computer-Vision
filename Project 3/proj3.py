import cv2
import numpy as np
import argparse
import sys
import os
from os import listdir
from os.path import isfile, join

from mtcnn_cv2 import MTCNN

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-webcam', help="no path", action='store_true')
parser.add_argument('-video', help="requires path")
parser.add_argument('-image', help="requires  path")
parser.add_argument('-folder', help="requires path")
args = parser.parse_args()


###########################################################################
def draw_boxes_eyes(frame, rectangles, color):
    for rect in rectangles:
        print(rect)
        draw_box_eyes(frame, rect, color)


def draw_box_eyes(frame, rect, color):
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def drawbox(frame, box, boxcolor, confidence):
    # print(f)
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    cv2.rectangle(frame, (x, y), (x + w, y + h), boxcolor, 2)
    textcolor = (255, 155, 0)
    text = "{:.4f}".format(confidence)
    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 2)
    return (frame)


def draw_boxes(frame, detections, detector):
    number_detections = 0
    detection_color = (0, 155, 255)
    non_detection_color = (155, 255, 0)

    if len(detections) > 0:
        for f in detections:
            box = f['box']
            confidence = f['confidence']
            if (wink(frame, f, len(detections))):
                number_detections += 1

                drawbox(frame, box, detection_color, confidence)
            
    return (number_detections)


def detect_eyes(cascade, gray_frame):
    scaleFactor = 1.3  # range is from 1 to infinity
    minNeigh =5  # range is from 0 to infinity
    flag = 0 | cv2.CASCADE_SCALE_IMAGE  # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (2, 2)  # range is from (0,0) to ..
    eyes = cascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeigh,
        flag,
        minSize)
    return (eyes)


def wink(frame, f, detection):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                        + 'haarcascade_eye.xml')

    eye_detections = detect_eyes(eye_cascade, gray_frame)
    eye_color = (100, 255, 100)
    print(len(eye_detections))

    box = f['box']
    if (len(eye_detections) == 1 ):
        return (True)
    else:
        return (False)


def detection_frame(detector, frame):
    detections = detector.detect_faces(frame)
    number_detections = draw_boxes(frame, detections, detector)
    return (number_detections)


def detect_video(detector, video_source):
    windowName = "Video"
    showlive = True
    while (showlive):
        ret, frame = video_source.read()
        if not ret:
            showlive = False;
        else:
            detection_frame(detector, frame)
            cv2.imshow(windowName, frame)
            if cv2.waitKey(30) >= 0:
                showlive = False
    # outside the while loop
    video_source.release()
    cv2.destroyAllWindows()
    return


###########################################################################

def runon_image(detector, path):
    frame = cv2.imread(path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections_in_frame = detection_frame(detector, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("one image", frame)
    cv2.waitKey(0)
    return detections_in_frame


def runon_webcam(detector):
    video_source = cv2.VideoCapture(0)
    if not video_source.isOpened():
        print("Can't open default video camera!")
        exit()

    detect_video(detector, video_source)
    return


def runon_video(detector, path):
    video_source = cv2.VideoCapture(path)
    if not video_source.isOpened():
        print("Can't open video ", path)
        exit()
    detect_video(detector, video_source)
    return


def runon_folder(detector, path):
    if (path[-1] != "/"):
        path = path + "/"
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    all_detections = 0
    for f in files:
        f_detections = runon_image(detector, f)
        all_detections += f_detections
        print(all_detections)
    return all_detections


if __name__ == '__main__':
    webcam = args.webcam
    video = args.video
    image = args.image
    folder = args.folder
    if not webcam and video is None and image is None and folder is None:
        print(
            "one argument from webcam,video,image,folder must be given.",
            "\n",
            "Example)",
            "-webcam or -video clip.avi or -image image.jpg or -folder images")
        sys.exit()

    detector = MTCNN()

    if webcam:
        runon_webcam(detector)
    elif video is not None:
        runon_video(detector, video)
    elif image is not None:
        runon_image(detector, image)
    elif folder is not None:
        all_detections = runon_folder(detector, folder)
        print("total of ", all_detections, " detections")
    else:
        print("impossible")
        sys.exit()

    cv2.destroyAllWindows()
