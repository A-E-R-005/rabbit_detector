import cv2
import torch
import numpy as np
import time

MOTION_THRESHOLD_PCT = 0.10
POINT_RADIUS = 1
POINT_DURATION = 0.25
LINE_THICKNESS = 1
BOX_PADDING = 10

def cameraToTensor():
    currentFrameTensor = None
    cap = cv2.VideoCapture(0)
    firstLoopPassed = False
    threshold = 255 * 3 * MOTION_THRESHOLD_PCT
    activePoints = []

    if not cap.isOpened():
        raise IOError("Cannot open camera")

    cv2.namedWindow('Motion Only', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Motion Only', 1280, 720)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original', 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame")
            break

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chwTensor = torch.from_numpy(rgbFrame).permute(2, 0, 1).float()

        if firstLoopPassed:
            oldFrameTensor = currentFrameTensor
            currentFrameTensor = chwTensor

            differenceTensor = torch.abs(currentFrameTensor - oldFrameTensor).sum(dim=0)
            changeMask = differenceTensor > threshold
            maskNumpy = changeMask.numpy()

            changedPixels = np.argwhere(maskNumpy)
            if len(changedPixels) > 0:
                now = time.time()
                step = max(1, int(POINT_RADIUS * 2))
                sampled = changedPixels[::step]
                for (y, x) in sampled:
                    activePoints.append((int(x), int(y), now))

        else:
            currentFrameTensor = chwTensor

        now = time.time()
        activePoints = [(x, y, t) for (x, y, t) in activePoints if now - t < POINT_DURATION]

        outputFrame = np.full_like(frame, 255)

        for (x, y, t) in activePoints:
            age = now - t
            fade = age / POINT_DURATION
            grey = int(200 * fade)
            colour = (grey, grey, grey)
            cv2.circle(outputFrame, (x, y), POINT_RADIUS, colour, -1)

        if activePoints:
            xs = [x for (x, y, _) in activePoints]
            ys = [y for (x, y, _) in activePoints]
            x_min = max(0, min(xs) - POINT_RADIUS - BOX_PADDING)
            y_min = max(0, min(ys) - POINT_RADIUS - BOX_PADDING)
            x_max = min(frame.shape[1], max(xs) + POINT_RADIUS + BOX_PADDING)
            y_max = min(frame.shape[0], max(ys) + POINT_RADIUS + BOX_PADDING)
            cv2.rectangle(outputFrame, (x_min, y_min), (x_max, y_max), (0, 0, 255), LINE_THICKNESS)

        cv2.imshow('Motion Only', outputFrame)
        cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        firstLoopPassed = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cameraToTensor()