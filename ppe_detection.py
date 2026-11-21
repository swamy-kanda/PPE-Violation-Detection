import cv2
import numpy as np
import time

from ai_edge_litert.interpreter import Interpreter

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "best.tflite"

CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

CLASS_NAMES = [
    "Person",
    "Hardhat",
    "Safety Vest"
]

# ---------------- LOAD MODEL ----------------

interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=4
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("Model loaded successfully!")

# ---------------- CAMERA ----------------

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

ret, frame = cap.read()

if not ret:
    print("Frame capture failed!")
    cap.release()
    exit()

print("Frame:", frame.shape)

# ---------------- PREPROCESSING ----------------

original_height, original_width = frame.shape[:2]

image = cv2.resize(frame, (640, 640))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = image.astype(np.float32) / 255.0

# NCHW format: [1, 3, 640, 640]

image = np.transpose(image, (2, 0, 1))

image = np.expand_dims(image, axis=0)

# ---------------- INFERENCE ----------------

interpreter.set_tensor(input_index, image)

start_time = time.time()

interpreter.invoke()

inference_time = (time.time() - start_time) * 1000

output = interpreter.get_tensor(output_index)

print("\nInference successful!")
print("Inference time:", round(inference_time, 2), "ms")
print("Output shape:", output.shape)

# ---------------- OUTPUT DECODING ----------------

predictions = output[0].T

boxes = predictions[:, :4]

class_scores = predictions[:, 4:]

class_ids = np.argmax(class_scores, axis=1)

confidences = np.max(class_scores, axis=1)

# Filter detections

mask = confidences > CONF_THRESHOLD

boxes = boxes[mask]

confidences = confidences[mask]

class_ids = class_ids[mask]

print("\nDetections before NMS:", len(boxes))

# ---------------- BOX CONVERSION ----------------

# YOLO output: center_x, center_y, width, height

xyxy_boxes = []

for box in boxes:

    cx, cy, w, h = box

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)

    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    xyxy_boxes.append([x1, y1, x2, y2])

xyxy_boxes = np.array(xyxy_boxes)

# ---------------- NMS ----------------

final_indices = []

for class_id in np.unique(class_ids):

    class_indices = np.where(class_ids == class_id)[0]

    class_boxes = xyxy_boxes[class_indices]

    class_scores = confidences[class_indices]

    # Convert xyxy to xywh for NMS

    nms_boxes = []

    for box in class_boxes:

        x1, y1, x2, y2 = box

        nms_boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

    selected = cv2.dnn.NMSBoxes(
        nms_boxes,
        class_scores.tolist(),
        CONF_THRESHOLD,
        IOU_THRESHOLD
    )

    if len(selected) > 0:

        selected = np.array(selected).flatten()

        final_indices.extend(class_indices[selected])

# ---------------- DRAW DETECTIONS ----------------

for index in final_indices:

    x1, y1, x2, y2 = xyxy_boxes[index]

    confidence = confidences[index]

    class_id = class_ids[index]

    label = CLASS_NAMES[class_id]

    label_text = f"{label}: {confidence:.2f}"

    # Map coordinates to original frame

    x1 = int(x1 * original_width / 640)

    y1 = int(y1 * original_height / 640)

    x2 = int(x2 * original_width / 640)

    y2 = int(y2 * original_height / 640)

    # Draw bounding box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        label_text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    print(label_text, "Box:", x1, y1, x2, y2)

# ---------------- SAVE IMAGE ----------------

cv2.imwrite("ppe_detection_result.jpg", frame)

print("\nFinal detections:", len(final_indices))

print("Saved image: ppe_detection_result.jpg")

cap.release()
