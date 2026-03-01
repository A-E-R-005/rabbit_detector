import numpy as np
import cv2
import pickle
import sys
import torch
import time
import os
from ml_genn import Network, Connection, Population
from ml_genn.neurons import LeakyIntegrateFire, SpikeInput
from ml_genn.connectivity import Dense
from ml_genn.compilers import EPropCompiler, InferenceCompiler
from ml_genn.compilers.eprop_compiler import default_params
from ml_genn.losses import SparseCategoricalCrossentropy
from ml_genn.serialisers import Numpy
from ml_genn.callbacks import Checkpoint

FRAME_DIM = (32, 32)
NUM_PIXELS = 1024
NUM_HIDDEN = 128
NUM_CLASSES = 2
TIMESTEPS = 20
PIXEL_THRESH = 20
WEIGHTS_FILE = "rabbit_weights"
NUM_EPOCHS = 50

MOTION_THRESHOLD_PCT = 0.10
POINT_RADIUS = 1
POINT_DURATION = 0.25
LINE_THICKNESS = 1
BOX_PADDING = 10


def get_spike_array(changedPixels, frame_height, frame_width):
    ys = np.clip((changedPixels[:, 0] * 32 / frame_height).astype(int), 0, 31)
    xs = np.clip((changedPixels[:, 1] * 32 / frame_width).astype(int), 0, 31)
    indices = ys * 32 + xs
    spike_array = np.zeros(NUM_PIXELS, dtype=np.float32)
    np.put(spike_array, indices, 1.0)
    return spike_array


def torch_motion_frame(frame, currentFrameTensor, firstLoopPassed, threshold):
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    chwTensor = torch.from_numpy(rgbFrame).permute(2, 0, 1).float()
    changedPixels = np.empty((0, 2), dtype=np.int32)

    if firstLoopPassed:
        differenceTensor = torch.abs(chwTensor - currentFrameTensor).sum(dim=0)
        changeMask = differenceTensor > threshold
        maskNumpy = changeMask.numpy()
        changedPixels = np.argwhere(maskNumpy)

    return chwTensor, changedPixels


def collect_training_data():
    samples = []
    labels = []

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

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        currentFrameTensor, changedPixels = torch_motion_frame(
            frame, currentFrameTensor, firstLoopPassed, threshold
        )

        spike_array = np.zeros(NUM_PIXELS, dtype=np.float32)
        if firstLoopPassed and len(changedPixels) > 0:
            spike_array = get_spike_array(changedPixels, frame_height, frame_width)

            now = time.time()
            step = max(1, int(POINT_RADIUS * 2))
            sampled = changedPixels[::step]
            for (y, x) in sampled:
                activePoints.append((int(x), int(y), now))

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

        rabbit_count = labels.count(1)
        not_count = labels.count(0)
        cv2.putText(frame, "r=rabbit  n=not rabbit  q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Rabbit: {rabbit_count}  Not: {not_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)

        cv2.imshow('Motion Only', outputFrame)
        cv2.imshow('Original', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            samples.append(spike_array)
            labels.append(1)
        elif key == ord('n'):
            samples.append(spike_array)
            labels.append(0)
        elif key == ord('q'):
            break

        firstLoopPassed = True

    samples = np.array(samples, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    cap.release()
    cv2.destroyAllWindows()
    with open("training_data.pkl", "wb") as f:
        pickle.dump((samples, labels), f)

    return samples, labels


def prepare_batch(samples, labels):
    samples = np.tile(np.expand_dims(samples, axis=1), (1, TIMESTEPS, 1))
    labels = np.reshape(labels, (len(labels), 1))
    return samples, labels


def build_network():
    net = Network(default_params)
    with net:
        inp = Population(SpikeInput(max_spikes=NUM_PIXELS), NUM_PIXELS)
        hidden = Population(LeakyIntegrateFire(v_thresh=0.5, tau_mem=20.0, tau_refrac=5.0), NUM_HIDDEN)
        output = Population(LeakyIntegrateFire(v_thresh=0.5, tau_mem=20.0, tau_refrac=5.0), NUM_CLASSES)

        w1 = np.random.normal(0.0, 0.01, (NUM_PIXELS, NUM_HIDDEN))
        w2 = np.random.normal(0.0, 0.01, (NUM_HIDDEN, NUM_CLASSES))

        Connection(inp, hidden, Dense(w1))
        Connection(hidden, output, Dense(w2))

    return net, inp, output


def train(samples, labels):
    net, inp, output = build_network()
    compiler = EPropCompiler(
        example_timesteps=TIMESTEPS,
        losses={output: SparseCategoricalCrossentropy()},
        dt=1.0,
        batch_size=8
    )

    compiled = compiler.compile(net)
    samples, labels = prepare_batch(samples, labels)
    serialiser = Numpy(WEIGHTS_FILE)

    with compiled:
        compiled.train(
            {inp: samples},
            {output: labels},
            num_epochs=NUM_EPOCHS,
            shuffle=True,
            callbacks=[Checkpoint(serialiser)]
        )


def inference():
    if not os.path.exists(WEIGHTS_FILE):
        print("No saved weights found. Run --train first.")
        sys.exit(1)

    network, inp, out = build_network()
    serialiser = Numpy(WEIGHTS_FILE)
    network.load((NUM_EPOCHS - 1,), serialiser)

    compiler = InferenceCompiler(dt=1.0, batch_size=1, evaluate_timesteps=TIMESTEPS)
    compiled = compiler.compile(network)

    cap = cv2.VideoCapture(0)
    currentFrameTensor = None
    firstLoopPassed = False
    threshold = 255 * 3 * MOTION_THRESHOLD_PCT

    with compiled:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_height = frame.shape[0]
            frame_width = frame.shape[1]

            currentFrameTensor, changedPixels = torch_motion_frame(
                frame, currentFrameTensor, firstLoopPassed, threshold
            )

            spike_array = np.zeros(NUM_PIXELS, dtype=np.float32)
            if firstLoopPassed and len(changedPixels) > 0:
                spike_array = get_spike_array(changedPixels, frame_height, frame_width)

            spikes = np.tile(spike_array[np.newaxis, np.newaxis, :], (1, TIMESTEPS, 1))
            compiled.evaluate({inp: spikes}, {})

            spike_data = out.record_spikes
            cv2.imshow("Detector", frame)
            if spike_data:
                cv2.putText(frame, "DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            firstLoopPassed = True

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--collect" in sys.argv:
        collect_training_data()
    elif "--train" in sys.argv:
        if not os.path.exists("training_data.pkl"):
            print("No training data found. Run --collect first.")
            sys.exit(1)
        with open("training_data.pkl", "rb") as f:
            samples, labels = pickle.load(f)
        train(samples, labels)
    else:
        inference()