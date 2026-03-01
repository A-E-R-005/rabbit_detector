import numpy as np
import cv2
import os
import pickle
import sys
import serial
from ml_genn import Network, Connection, Population
from ml_genn.neurons import LeakyIntegrateFire, SpikeInput
from ml_genn.connectivity import Dense
from pygenn import genn_model
from ml_genn.compilers import EPropCompiler, InferenceCompiler
from ml_genn.compilers.eprop_compiler import default_params
from ml_genn.losses import SparseCategoricalCrossentropy
from ml_genn.serialisers import Numpy
from ml_genn.callbacks import Checkpoint

FRAME_DIM = (32, 32)  # Reframing the webcam to 32x32
NUM_PIXELS = 1024  # Number of pixels in the image, one neuron per pixel
NUM_HIDDEN = 128  # This is the middle layer of the network, 128 neurons that
# sit between the pixels and output find patterns, pixels that fire together
NUM_CLASSES = 2  # 1 Represents Found 0 Represents Not found
TIMESTEPS = 20  # Each fram runs for 20ms of simulated brain time, giving the chance to fire, 20 simulation steps
PIXEL_THRESH = 20  # Threshold needed to reach to hit a spike
WEIGHTS_FILE = "rabbit_weights"  # Folder where all weights live


class SpikeRetina:
    def __init__(self):
        self.prev_grey = None


    def process(self, frame):
        reduce = cv2.resize(frame, FRAME_DIM)
        grey = cv2.cvtColor(reduce, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.prev_grey is None:
            self.prev_grey = grey
            return np.zeros(NUM_PIXELS, dtype=np.float32)

        diff = np.abs(grey - self.prev_grey)
        spikes = (diff > PIXEL_THRESH).astype(np.float32)
        self.prev_grey = grey
        return spikes.flatten()


def collect_training_data():
    cap = cv2.VideoCapture(0)
    retina = SpikeRetina()
    samples = []
    labels = []

    while True:
        success, frame = cap.read()
        spikes = retina.process(frame)
        cv2.putText(frame, "Press (r) to record as yes      Press (n) to record it as no         Press (q) to break",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Collect Training Data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            samples.append(spikes)
            labels.append(1)
        elif key == ord('n'):
            samples.append(spikes)
            labels.append(0)
        elif key == ord('q'):
            break
    samples = np.array(samples, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    cap.release()
    cv2.destroyAllWindows()
    with open("training_data.pkl", "wb") as f:
        pickle.dump((samples, labels), f)

    return samples, labels


def prepare_batch(samples, labels):
    samples = np.tile(samples[:, np.newaxis, :], (1, 20, 1))
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
    compiler = EPropCompiler(example_timesteps=TIMESTEPS, losses={output: SparseCategoricalCrossentropy()},
                             dt=1.0, batch_size=8)

    compiled = compiler.compile(net)
    samples, labels = prepare_batch(samples, labels)

    serialiser = Numpy(WEIGHTS_FILE)

    with compiled:
        compiled.train({inp: samples}, {output: labels}, num_epochs=50, shuffle=True, callbacks=[Checkpoint(serialiser)])


def inference():
    network, inp, out = build_network()
    serialiser = Numpy(WEIGHTS_FILE)
    weights = network.load((49,), serialiser)

    compiler = InferenceCompiler(dt=1.0, batch_size=1,evaluate_timesteps=TIMESTEPS)
    compiled = compiler.compile(network)

    cap = cv2.VideoCapture(0)
    retina = SpikeRetina()
    with compiled:
        while True:
            success, frame = cap.read()
            spikes = retina.process(frame)
            spikes = np.tile(spikes[np.newaxis, np.newaxis, :], (1, 20, 1))
            result = compiled.evaluate({inp: spikes}, {})
            spike_data = out.record_spikes
            cv2.imshow("Detector", frame)
            if spike_data > 0:
                cv2.putText(frame, "Detected",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if"--collect" in sys.argv:
        collect_training_data()
    elif"--train" in sys.argv:
        with open("training_data.pkl", "rb") as f:
            samples, labels = pickle.load(f)
        train(samples, labels)
    else:
        inference()
