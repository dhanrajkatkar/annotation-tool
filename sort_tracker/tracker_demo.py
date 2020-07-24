# import the necessary packages
import sys
import time
from collections import deque
import numpy as np
from imutils.object_detection import non_max_suppression
import cv2
from tracker import Tracker

WORKING_LINE_COLOR = (0, 0, 255)


def draw_label(img, text, position, bg_color):
    """
    Draws a label at the specified position.

    :param img: the to be drawn into
    :param text: the text to be displayed
    :param position: the x, y position where the label will be drawn
    :param bg_color: the background color for the label. best choose the color of your bounding box.
    """

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    bg_end_x = position[0] + txt_size[0][0] + margin
    bg_end_y = position[1] + txt_size[0][1] + margin
    bg_end_pos = (bg_end_x, bg_end_y)

    text_start_pos = (position[0], position[1] + txt_size[0][1])

    cv2.rectangle(img, position, bg_end_pos, bg_color, thickness)
    cv2.putText(img, text, text_start_pos, font_face, scale, color, 1, cv2.LINE_AA)


def update_fps(prev_time, fps, actual_fps):
    curr_time = time.time()
    if abs(curr_time - prev_time) <= 1:
        fps = fps + 1
    else:
        actual_fps = fps
        prev_time = curr_time
        fps = 0
    return prev_time, fps, actual_fps


count = 0
prev_pos = None

m_coordinates = None

prev_ls = []

tracker = Tracker(100, 20, 9, 70)

# Variables initialization
# skip_frame_count = 0
track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127)]


# start looping over all the frames
def l_cross(net, vs):
    # horizon = horizon
    # print('here', on_mouse2)
    buffer = 20
    pts = deque(maxlen=buffer)
    # Counts the minimum no. of frames to be detected where direction change occurs
    counter = 0
    # Change in direction is stored in dX, dY
    (dX, dY) = (0, 0)
    # Variable to store direction string
    direction = ''

    prev_time = time.time()
    fps = 0
    actual_fps = 0
    cv2.namedWindow('Line Cross Tracker', )
    while True:
        prev_time, fps, actual_fps = update_fps(prev_time, fps, actual_fps)
        # time.sleep(0.2)
        # receive RPi name and frame from the RPi and acknowledge
        # the receipt
        _, frame = vs.read()
        frame = cv2.resize(frame, (600, 480))
        if frame is None:
            continue
        clone = frame.copy()

        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1.1, size=(300, 300), swapRB=True, crop=False)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        rows, cols, channels = frame.shape
        # reset the object count for each object in the CONSIDER set
        rects = []

        for i, detection in enumerate(detections[0, 0, :, :]):

            score = float(detection[2])
            if score > 0.4:
                # print(np.arange(0, detection.shape[2]))
                # idx = int(detection[0, 0, , 1])
                idx = int(detections[0, 0, i, 1])
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)

                if idx == 1:
                    # confidences.append(float(score))
                    rects.append([int(left), int(top), int(right), int(bottom)])

        rects = np.array([[x, y, w, h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        centers = []
        for (xA, yA, xB, yB) in pick:

            r_width = xA - xB
            r_height = yA - yB

            p_area = ((r_width * r_height) / (frame.shape[0] * frame.shape[1])) * 100
            if p_area > 20:
                print('greater')
                continue
            cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

            pos = (int(xA), int(yA))
            center = (int((pos[0] + int(xB)) / 2), int((pos[1] + int(yB)) / 2))
            # centers.append([center[0], center[1]])
            b = np.array([[center[0]], [center[1]]])
            centers.append(np.round(b))
            counter += 1

            cv2.circle(clone, center, 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

            draw_label(clone, 'person', (xA, yA), (0, 255, 0))

        # Track object using Kalman Filter
        tracker.Update(centers)

        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id,
        # tracking algorithm here functions more effecient in terms of speed performance tradeoff, the overall logarithmic time complexity using kalman filter is Olog(n)
        for i in range(len(tracker.tracks)):
            if len(tracker.tracks[i].trace) > 1:

                x = int(tracker.tracks[i].trace[-1][0, 0])
                y = int(tracker.tracks[i].trace[-1][1, 0])
                # tl = (x - 10, y - 10)
                # br = (x + 10, y + 10)
                clr = tracker.tracks[i].track_id % 9
                cv2.putText(clone, str(clr), (int(x) - 10, int(y) - 20), 0, 0.5, track_colors[clr], 2)

                trace_tail_x = int(tracker.tracks[i].trace[0][0][0])
                trace_tail_y = int(tracker.tracks[i].trace[0][1][0])

                for j in range(len(tracker.tracks[i].trace) - 1):
                    # Draw trace line
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j + 1][0][0]
                    y2 = tracker.tracks[i].trace[j + 1][1][0]

                    cv2.line(clone, (int(x1), int(y1)), (int(x2), int(y2)),
                             track_colors[clr], 2)

                cv2.circle(clone, (trace_tail_x, trace_tail_y), 3, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(clone, (x, y), 3, (255, 255, 255), thickness=-1, lineType=cv2.FILLED)

        # display the montage(s) on the screen
        cv2.putText(clone, "FPS:" + str(actual_fps), (60, 70), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Line Cross Tracker", clone)
        # detect any kepresses
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cam = sys.argv[1]
    net = cv2.dnn.readNetFromTensorflow(
        './model/frozen_inference_graph.pb',
        './model/ssd_mobilenet.pbtxt.txt')
    vs = cv2.VideoCapture(cam)
    l_cross(net, vs)
