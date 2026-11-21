import cv2
import numpy as np
import time

from ai_edge_litert.interpreter import Interpreter

# Load YOLO model
MODEL_PATH = "best.tflite"

interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=4
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Camera pipeline
pipeline = (
    "qtiqmmfsrc camera=0 ! "
    "video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera opening failed!")
    exit()

print("Camera opened successfully!")

# Capture frame
ret, frame = cap.read()

if not ret:
    print("Frame capture failed!")
    cap.release()
    exit()

print("Camera frame:", frame.shape)

# Preprocessing
image = cv2.resize(frame, (640, 640))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = image.astype(np.float32) / 255.0

# HWC -> CHW
image = np.transpose(image, (2, 0, 1))

# Add batch dimension
image = np.expand_dims(image, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]["index"], image)

start = time.time()
interpreter.invoke()
end = time.time()

output = interpreter.get_tensor(output_details[0]["index"])

print("\nInference successful!")
print("Output shape:", output.shape)
print("Inference time:", round((end - start) * 1000, 2), "ms")
print("Maximum output value:", np.max(output))
print("Minimum output value:", np.min(output))

cap.release()
