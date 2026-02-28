
import numpy as np
import time
from dataclasses import dataclass

threshold = 0
ultrasonic_threshold_metres = 5
pir_hold_seconds = 1 #ultrasonic is awake for this long

#stage 1 - PIR gate
def pir_spike(distance):
    return 1 if distance < threshold else 0

#stage 2 - sonic sensor
def ultrasonic_spike(distance):
    #read from hardware
    return 10

#stage 3 - lidar sensor
def lidar_spike(distance):
    #classify


class GateState:
    float open_until = 0.0

def update_gate(state: GateState, pir_value:int, now: float, hold_s: float) -> bool:
    #activated by pir
    if pir_value == 1:
        state.open_until = max(state.open_until, now + hold_s )
    return now < state.open_until


def main():
    gate_state = GateState()

    while True:
        now = time.time
        pir = pir_spike()
        stage2_on = update_gate(gate_state, pir, now, pir_hold_seconds)

        if stage2_on:
            distance = ultrasonic_spike()

            if distance < ultrasonic_threshold_metres:
                lidar_spike()

