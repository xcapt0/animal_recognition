import torch
import cv2
import numpy as np
from argparse import ArgumentParser


def detect(args):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=True)
    cap = cv2.VideoCapture(args.source)

    while cap.isOpened():
        ret, frame = cap.read()

        # Make detections
        results = model(frame)
        cv2.imshow('YOLO', np.squeeze(results.render()))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', type=str, required=True,
                        help='Source to load the video from webcam or mp4/avi file')
    user_args = parser.parse_args()

    if user_args.source.isdigit():
        user_args.source = int(user_args.source)

    detect(user_args)