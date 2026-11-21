import cv2
import numpy as np
import time

from ai_edge_litert.interpreter import Interpreter

# ================= CONFIGURATION =================

MODEL_PATH = "best.tflite"
IMAGE_PATH = "construction-worker-site.jpg"

CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

CLASS_NAMES = ["Person", "Hardhat", "Safety Vest"]

INPUT_SIZE = 640

# ================= LOAD MODEL =================

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

# ================= LOAD IMAGE =================

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("Image loading failed!")
    exit()

original_height, original_width = frame.shape[:2]

print("Original image:", frame.shape)

# ================= PREPROCESSING =================

# Calculate letterbox scale
scale = min(
    INPUT_SIZE / original_width,
    INPUT_SIZE / original_height
)

new_width = int(original_width * scale)
new_height = int(original_height * scale)

# Resize image
resized = cv2.resize(
    frame,
    (new_width, new_height)
)

# Calculate padding
pad_x = (INPUT_SIZE - new_width) // 2
pad_y = (INPUT_SIZE - new_height) // 2

# Create letterbox image
image = np.full(
    (INPUT_SIZE, INPUT_SIZE, 3),
    114,
    dtype=np.uint8
)

image[
    pad_y:pad_y + new_height,
    pad_x:pad_x + new_width
] = resized

# BGR -> RGB
image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# Normalize
image = image.astype(np.float32) / 255.0

# NHWC -> NCHW
image = np.transpose(
    image,
    (2, 0, 1)
)

# Add batch dimension
image = np.expand_dims(
    image,
    axis=0
)

# ================= INFERENCE =================

interpreter.set_tensor(
    input_index,
    image
)

start_time = time.time()

interpreter.invoke()

inference_time = (
    time.time() - start_time
) * 1000

output = interpreter.get_tensor(
    output_index
)

print("\nInference successful!")

print(
    "Inference time:",
    round(inference_time, 2),
    "ms"
)

print(
    "Output shape:",
    output.shape
)

# ================= OUTPUT DECODING =================

predictions = output[0].T

# First 4 values = bounding box
# Remaining values = class scores

boxes = predictions[:, :4]

scores = predictions[:, 4:]

class_ids = np.argmax(
    scores,
    axis=1
)

confidences = np.max(
    scores,
    axis=1
)

# Confidence filtering
mask = confidences > CONF_THRESHOLD

boxes = boxes[mask]
confidences = confidences[mask]
class_ids = class_ids[mask]

print(
    "Candidates after confidence filtering:",
    len(boxes)
)

# ================= COORDINATE CONVERSION =================

# Convert normalized coordinates to 640x640 pixels

boxes = boxes * INPUT_SIZE

xyxy_boxes = []

for box in boxes:

    cx, cy, w, h = box

    # XYWH -> XYXY

    x1 = cx - w / 2
    y1 = cy - h / 2

    x2 = cx + w / 2
    y2 = cy + h / 2

    # Remove letterbox padding

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale

    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    # Clip coordinates to image boundaries

    x1 = np.clip(
        x1,
        0,
        original_width
    )

    y1 = np.clip(
        y1,
        0,
        original_height
    )

    x2 = np.clip(
        x2,
        0,
        original_width
    )

    y2 = np.clip(
        y2,
        0,
        original_height
    )

    xyxy_boxes.append(
        [x1, y1, x2, y2]
    )

xyxy_boxes = np.array(
    xyxy_boxes,
    dtype=np.float32
)

# ================= IoU CALCULATION =================

def calculate_iou(box, boxes):

    x1 = np.maximum(
        box[0],
        boxes[:, 0]
    )

    y1 = np.maximum(
        box[1],
        boxes[:, 1]
    )

    x2 = np.minimum(
        box[2],
        boxes[:, 2]
    )

    y2 = np.minimum(
        box[3],
        boxes[:, 3]
    )

    intersection = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    area1 = (
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    area2 = (
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


# ================= NON-MAXIMUM SUPPRESSION =================

def nms(boxes, scores, threshold):

    order = np.argsort(scores)[::-1]

    keep = []

    while len(order) > 0:

        current = order[0]

        keep.append(current)

        if len(order) == 1:
            break

        remaining = order[1:]

        ious = calculate_iou(
            boxes[current],
            boxes[remaining]
        )

        order = remaining[
            ious < threshold
        ]

    return keep


# Class-wise NMS
final_indices = []

for class_id in np.unique(class_ids):

    class_indices = np.where(
        class_ids == class_id
    )[0]

    class_boxes = xyxy_boxes[
        class_indices
    ]

    class_scores = confidences[
        class_indices
    ]

    selected = nms(
        class_boxes,
        class_scores,
        IOU_THRESHOLD
    )

    final_indices.extend(
        class_indices[selected]
    )

# ================= DRAW DETECTIONS =================

for index in final_indices:

    x1, y1, x2, y2 = xyxy_boxes[index]

    confidence = confidences[index]

    class_id = class_ids[index]

    label = CLASS_NAMES[class_id]

    label_text = (
        f"{label}: {confidence:.2f}"
    )

    # Bounding box
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        3
    )

    # Label
    cv2.putText(
        frame,
        label_text,
        (int(x1), max(int(y1) - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    print(
        label_text,
        "Box:",
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    )

# ================= SAVE RESULT =================

cv2.imwrite(
    "ppe_result.jpg",
    frame
)

print("\nTotal detections:", len(final_indices))

print(
    "Output saved:",
    "ppe_result.jpg"
)
