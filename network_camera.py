import socket
import pickle
import struct
import cv2
import numpy as np


class NetworkCamera:
    """Drop-in replacement for cv2.VideoCapture that reads from a network stream."""

    def __init__(self, host, port=9999):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Connecting to camera at {host}:{port} ...")
        self.client.connect((host, port))
        print("Connected — receiving frames.")
        self.payload_size = struct.calcsize("L")
        self.buffer = b""

    def isOpened(self):
        return True

    def read(self):
        try:
            # Keep reading until we have enough bytes for the size header
            while len(self.buffer) < self.payload_size:
                self.buffer += self.client.recv(4096)

            packed_size = self.buffer[:self.payload_size]
            self.buffer = self.buffer[self.payload_size:]
            msg_size = struct.unpack("L", packed_size)[0]

            # Keep reading until we have the full frame
            while len(self.buffer) < msg_size:
                self.buffer += self.client.recv(4096)

            frame_data = self.buffer[:msg_size]
            self.buffer = self.buffer[msg_size:]

            frame = pickle.loads(frame_data)
            return True, frame

        except Exception as e:
            print(f"Stream error: {e}")
            return False, None

    def release(self):
        self.client.close()
