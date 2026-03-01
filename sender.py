import cv2
import socket
import pickle
import struct

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 9999))
server.listen(1)
print("Waiting for connection from your friend's desktop...")

conn, addr = server.accept()
print(f"Connected to {addr} - streaming now. Press Ctrl+C to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    data = pickle.dumps(frame)
    size = struct.pack("L", len(data))
    try:
        conn.sendall(size + data)
    except (BrokenPipeError, ConnectionResetError):
        print("Connection lost.")
        break

cap.release()
conn.close()
server.close()
