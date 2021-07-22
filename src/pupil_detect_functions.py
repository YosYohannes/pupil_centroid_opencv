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
    x1 = c1 + int(0.15 * c3)
    y1 = c2 + int(0.15 * c4)
    x2 = c1 + int(0.85 * c3)
    y2 = c2 + int(0.85 * c4)
    # return param of bounding box
    if len(eyes) <= 0:
        return None
    else:
        return [x1, y1, x2, y2]


def process_roi(frame, eye, kernel):
    roi = frame[eye[1]:eye[3], eye[0]:eye[2]]  # set region of interest
    blurred = cv2.GaussianBlur(roi, (15, 15), 10)  # blur to reduce noise
    highcon = cv2.addWeighted(blurred, 2, blurred, 0, 0)
    # (Not very useful) use adaptive thresholds to overcome some lighting condition differences
    # threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 3)
    # (Not very useful)  enhance features, connect small gaps
    # dilated = cv2.dilate(threshold, kernel, iterations=1)
    return highcon


def find_pupil(frame, prev, eye):
    p2 = 150
    minR = 80
    maxR = 200
    if prev > 3:  # limit circle size to be +- 3 from previously found. Reduce jitter
        p2 = 100
        minR = prev - 5
        maxR = prev + 5

    # attempt to find 1 circle by giving large minDist, 1080
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 3, 1080, param1=20, param2=p2,
                               minRadius=minR, maxRadius=maxR)
    if circles is not None:  # return first circle found
        x = circles[0][0][0] + eye[0]  # shift to distance from origin
        y = circles[0][0][1] + eye[1]  # shift to distance from origin
        return np.array([x, y, circles[0][0][2]], int)
    else:
        return None


def find_centroids(inp, mode="smooth"):
    """ This function takes in 2 parameters
    :param inp: A string indicating the video file path
    :param mode: A string to indicate "smooth" or "snappy" behaviour. It is smooth by default

    This function will show 2 window:
    1. Original frame with centroid mark
    2. Processed frame with bounding box, pupil marking and other filters

    Use "spacebar" to pause/play. While paused, use "k" to go to next frame

    :return: A list of (x, y) tuple for centroids in each frame (-1, -1) indicate not found
    """

    cap = cv2.VideoCapture(inp)  # input video stream

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
    paused = False
    output = []
    prev_size = 0
    kernel = np.ones((5, 5), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ave_num = 10  # averaging number of frames
    if mode == "snap":
        ave_num = 20
    empty_frame_cnt = 0  # count of number of no eye to trigger reset
    prev_ok_bound = None  # keep previously known eye location to try when model detection fail
    pupil_ls = np.empty((0, 3), int)
    eye_cascade = cv2.CascadeClassifier('src/haarcascae_eye.xml')  # eye classifier
    fr_counter = 0
    # endregion

    # End of variables
    while True:
        ret, frame = cap.read()
        fr_counter += 1
        if frame is None:
            break

        # Step 0: Resize
        frame = cv2.resize(frame, (w, h))

        # Step 1: Find eye in image
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_bound = find_eye(eye_cascade, grey_frame)

        # Step 1.5: Prepare image for pupil finding
        focused_flag = True  # flag used to proceed with pupil finding
        if eye_bound is not None:
            focused = process_roi(grey_frame, eye_bound, kernel)
            prev_ok_bound = eye_bound
        elif prev_ok_bound is not None:  # if model fails, attempt to find pupil anyway unless gone for long
            eye_bound = prev_ok_bound  # use previously know area to search
            focused = process_roi(grey_frame, prev_ok_bound, kernel)
        else:
            focused_flag = False

        # Step 2: find pupil in eye region
        if focused_flag:
            pupil = find_pupil(focused, prev_size, eye_bound)
            if pupil is not None:

                # Step 3: Averaging of position for smoothing
                pupil_ls = np.vstack([pupil_ls, pupil])  # add to list for averaging
                if np.shape(pupil_ls)[0] > ave_num:
                    pupil_ls = np.delete(pupil_ls, 0, 0)  # remove earliest in list
                cur_ave = np.mean(pupil_ls, axis=0)

                # Step 3A: Fix large error caused by quick movement. This is a quick snap
                if abs(pupil[0] - cur_ave[0]) + abs(pupil[1] - cur_ave[1]) > 32 and mode == "snap":
                    cur_ave = pupil_ls[-1]
                    pupil_ls = np.empty((0, 3), int)  # clear pupil list, fast travelling

                prev_size = int(pupil[2])

                # Step 4: Outputs
                cx = int(cur_ave[0])
                cy = int(cur_ave[1])
                cr = int(cur_ave[2])
                output.append((cx, cy))
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), 3)
                cv2.putText(frame, "Centroid", (cx - 65, cy - 20), font, 1, (0, 0, 255), 2)
                cv2.putText(frame, '(' + str(cx) + ',' + str(cy) + ')', (cx - 80, cy + 40), font, 1, (0, 0, 255), 2)

                grey_frame[eye_bound[1]:eye_bound[3], eye_bound[0]:eye_bound[2]] = focused
                cv2.circle(grey_frame, (cx, cy), cr, (0, 255, 0), 2)
                cv2.putText(grey_frame, "Pupil", (cx - 35, cy - cr - 10), font, 1, (0, 0, 255), 2)

            else:
                empty_frame_cnt += 1

            cv2.rectangle(grey_frame, (eye_bound[0], eye_bound[1]), (eye_bound[2], eye_bound[3]), (0, 255, 0), 2)
            cv2.putText(grey_frame, "eye_box", (eye_bound[0] + 10, eye_bound[1] - 20), font, 1, (0, 0, 255), 2)
            cv2.imshow("Processing", grey_frame)

        # Step 5: Reset smoothing data if no pupil was found for x time
        if empty_frame_cnt > 12:  # gone for 0.5 seconds to trigger reset
            pupil_ls = np.empty((0, 3), int)  # clear pupil list, when pupil or eyes can't be found
            prev_size = 0
            empty_frame_cnt = 0
            prev_ok_bound = None

        cv2.putText(frame, 'fr no: ' + str(fr_counter), (10, 50), font, 1, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)

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
