import cv2
import time
import numpy as np


def find_eye(model, frame):
    eyes = model.detectMultiScale(frame, 1.1, 15, cv2.CASCADE_SCALE_IMAGE, (220, 220))
    maxWidth = 0
    c1, c2, c3, c4 = 0, 0, 0, 0
    # Find the biggest "eye" from cascade results
    for (ex, ey, ew, eh) in eyes:
        if maxWidth < ew:
            maxWidth = ew
            c1, c2, c3, c4 = ex, ey, ew, eh
    # Cut off 15% all around eye region to reduce pixels to process and avoid false detections
    x_c = c1 + 0.5 * c3  # get center of box
    y_c = c2 + 0.5 * c4  # get center of box
    w = 0.70 * c3  # cut size by 30%
    h = 0.70 * c4  # cut size by 30%
    # return param of bounding box
    if len(eyes) <= 0:
        return None
    else:
        return np.array([[x_c], [y_c], [w], [h]], np.float32)


def process_roi(frame, eye, kernel):
    # set region of interest
    roi = frame[int(eye[1] - eye[5] / 2): int(eye[1] + eye[5] / 2), int(eye[0] - eye[4] / 2): int(eye[0] + eye[4] / 2)]
    blurred = cv2.GaussianBlur(roi, (15, 15), 10)  # blur to reduce noise
    highcon = cv2.addWeighted(blurred, 2.5, blurred, 0, 0)
    # (Not very useful) use adaptive thresholds to overcome some lighting condition differences
    # threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 3)
    # (Not very useful)  enhance features, connect small gaps
    # dilated = cv2.dilate(threshold, kernel, iterations=1)
    return highcon


def find_pupil(frame, prev, eye):
    p2 = 150
    minR = 80
    maxR = 200
    if prev > 5:  # limit circle size to be +- 3 from previously found. Reduce jitter
        p2 = 100
        minR = int(0.94 * prev)
        maxR = int(1.06 * prev)

    # attempt to find 1 circle by giving large minDist, 1080
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 3, 1080, param1=20, param2=p2,
                               minRadius=minR, maxRadius=maxR)
    if circles is not None:  # return first circle found
        x = circles[0][0][0] + eye[0] - eye[4] / 2  # shift to distance from origin
        y = circles[0][0][1] + eye[1] - eye[5] / 2  # shift to distance from origin
        return np.array([x, y, circles[0][0][2]], int)
    else:
        return None


def get_centroid(path, smooth=False, display=False, show=True):
    """ This function takes in 4 parameters
    :param path: A string indicating the video file path
    :param smooth: A Bool to indicate "smooth" or "snappy" behaviour. It is snappy by default
    :param display: A bool to indicate whether to show layer view
    :param show: A bool to indicate whether to show video with centroid marking

    This function will show 2 optional window:
    1. Original frame with centroid mark
    2. Processed frame with bounding box, pupil marking and other filters

    Use "spacebar" to pause/play. While paused, use "k" to go to next frame

    :return: A list of (x, y) tuple for centroids in each frame (-1, -1) indicate not found
    """

    cap = cv2.VideoCapture(path)  # input video stream

    # region Rescale input video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    scale_x = 1280 / w
    scale_y = 960 / h
    scale = int(max(scale_x, scale_y))
    w = w * scale
    h = h * scale
    # endregion

    # region Start of variables
    kernel = np.ones((5, 5), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    eye_cascade = cv2.CascadeClassifier('src/haarcascae_eye.xml')  # eye classifier
    paused = False
    fr_counter = 0
    output = []
    ave_num = 0.4 * fps
    if not smooth:
        ave_num = 0.8 * fps
    pupil_ls = np.empty((0, 3), int)
    prev_size = 0
    empty_frame_cnt = fps  # count of number of no eye to trigger reset
    no_pupil_cnt = fps  # count of number of no pupil to trigger reset

    tp = np.zeros((6, 1), np.float32)  # tracked/prediction array [x, y, dx, dy, w, h]
    kf = cv2.KalmanFilter(6, 4)
    kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 1]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0, 0, 0],  # [x,0,dt,0,0,0]
                                    [0, 1, 0, 1, 0, 0],  # [0,y,0,dt,0,0]
                                    [0, 0, 1, 0, 0, 0],  # [0,0,vx,0,0,0]
                                    [0, 0, 0, 1, 0, 0],  # [0,0,0,vy,0,0]
                                    [0, 0, 0, 0, 1, 0],  # [0,0,0,0,w,0] chose not to keep delta size
                                    [0, 0, 0, 0, 0, 1],  # [0,0,0,0,0,h] chose not to keep delta size
                                    ], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1], ], np.float32) * 0.03
    # endregion

    start = time.process_time()
    while True:
        ret, frame = cap.read()
        fr_counter += 1
        pupil_found = False
        if frame is None:
            break

        # Step 0: Resize
        frame = cv2.resize(frame, (w, h))

        # Step 1: Find eye in image
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_bound = find_eye(eye_cascade, grey_frame)

        eye_flag = False
        if empty_frame_cnt < fps / 2:  # check that eye exist within the past 0.5 seconds
            tp = kf.predict()
            if tp[0] - tp[4] / 2 > 0 and tp[1] - tp[5] / 2 > 0:
                eye_flag = True

        # Update filter
        if eye_bound is not None:
            kf.correct(eye_bound)
            empty_frame_cnt = 0
        else:
            empty_frame_cnt += 1

        # Step 2: find pupil in eye region
        if eye_flag:
            focused = process_roi(grey_frame, tp, kernel)
            grey_frame[int(tp[1] - tp[5] / 2): int(tp[1] + tp[5] / 2),
            int(tp[0] - tp[4] / 2): int(tp[0] + tp[4] / 2)] = focused
            pupil = find_pupil(focused, prev_size, tp)

            if pupil is not None:
                no_pupil_cnt = 0
                prev_size = int(pupil[2])

                # Step 3: Averaging of position for smoothing
                pupil_ls = np.vstack([pupil_ls, pupil])  # add to list for averaging
                if np.shape(pupil_ls)[0] > ave_num:
                    pupil_ls = np.delete(pupil_ls, 0, 0)  # remove earliest in list
                cur_ave = np.mean(pupil_ls, axis=0)

                # Step 3A: Fix large error caused by quick movement. This is a quick snap
                if abs(pupil[0] - cur_ave[0]) + abs(pupil[1] - cur_ave[1]) > 32 and not smooth:
                    cur_ave = pupil_ls[-1]
                    pupil_ls = np.empty((0, 3), int)  # clear pupil list, fast travelling

                # Step 4: Outputs
                cx = int(cur_ave[0])
                cy = int(cur_ave[1])
                cr = int(cur_ave[2])
                output.append((cx, cy))
                pupil_found = True
                cv2.circle(frame, (cx, cy), 3, (255, 255, 0), 3)
                cv2.putText(frame, "Centroid", (cx - 65, cy - 20), font, 1, (255, 255, 0), 2)
                cv2.putText(frame, '(' + str(cx) + ',' + str(cy) + ')', (cx - 80, cy + 40), font, 1, (255, 255, 0), 2)

                cv2.circle(grey_frame, (cx, cy), cr, (0, 255, 0), 2)
                cv2.putText(grey_frame, "Pupil", (cx - 35, cy - cr - 10), font, 1, (0, 255, 0), 2)

            else:
                no_pupil_cnt += 1

            if no_pupil_cnt > fps / 2:
                prev_size = 0

            cv2.rectangle(grey_frame, (int(tp[0] - tp[4] / 2), int(tp[1] - tp[5] / 2)),
                          (int(tp[0] + tp[4] / 2), int(tp[1] + tp[5] / 2)), (0, 255, 0), 2)
            cv2.putText(grey_frame, "eye_box", (int(tp[0]), int(tp[1] - tp[5] / 2 - 20)), font, 1, (0, 0, 255), 2)

        if display:
            cv2.imshow("Processing", grey_frame)
        cv2.putText(frame, 'fr no: ' + str(fr_counter), (10, 50), font, 1, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)

        if not pupil_found:
            output.append((-1, -1))

        key = cv2.waitKey(30)
        if key == 32:  # spacebar to pause and play
            paused = True

        if paused:
            while key not in [27, 32, ord('k')]:  # 'k' to next, spacebar to unpause
                key = cv2.waitKey(0)
                if key == 32:
                    paused = False

        if key == 27:  # 'esc' to exit
            break

    cv2.destroyAllWindows()
    return output
