import os
import cv2
import numpy as np
import time

from ai_edge_litert.interpreter import Interpreter

# ============================================================
# WAYLAND DISPLAY CONFIGURATION
# ============================================================

os.environ["XDG_RUNTIME_DIR"] = "/dev/socket/weston"
os.environ["WAYLAND_DISPLAY"] = "wayland-1"

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best.tflite"

CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

INPUT_SIZE = 640
NUM_THREADS = 4

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

CLASS_NAMES = [
    "Person",
    "Hardhat",
    "Safety Vest"
]

OUTPUT_IMAGE = "ppe_realtime.jpg"

WINDOW_NAME = "RB3 Gen2 - Live PPE Detection"

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading PPE detection model...")

interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=NUM_THREADS
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("Model loaded successfully!")

# ============================================================
# CAMERA CONFIGURATION
# ============================================================

pipeline = (
    "qtiqmmfsrc camera=0 ! "
    "video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1"
)

cap = cv2.VideoCapture(
    pipeline,
    cv2.CAP_GSTREAMER
)

if not cap.isOpened():

    print("ERROR: Camera opening failed!")

    exit()

print("Camera opened successfully!")

# ============================================================
# FULL-SCREEN DISPLAY CONFIGURATION
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

print("Full-screen display initialized!")
print("Starting live PPE detection...")
print("Press 'q' to stop.")
print("============================================================")

# ============================================================
# NMS
# ============================================================

def calculate_iou(box, boxes):

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])

    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

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


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(frame):

    original_height, original_width = frame.shape[:2]

    scale = min(
        INPUT_SIZE / original_width,
        INPUT_SIZE / original_height
    )

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized = cv2.resize(
        frame,
        (new_width, new_height)
    )

    pad_x = (INPUT_SIZE - new_width) // 2
    pad_y = (INPUT_SIZE - new_height) // 2

    image = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3),
        114,
        dtype=np.uint8
    )

    image[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image, scale, pad_x, pad_y


# ============================================================
# POSTPROCESSING
# ============================================================

def postprocess(output, frame, scale, pad_x, pad_y):

    original_height, original_width = frame.shape[:2]

    predictions = output[0].T

    boxes = predictions[:, :4] * INPUT_SIZE

    scores = predictions[:, 4:]

    class_ids = np.argmax(
        scores,
        axis=1
    )

    confidences = np.max(
        scores,
        axis=1
    )

    mask = confidences > CONF_THRESHOLD

    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    xyxy_boxes = []

    for box in boxes:

        cx, cy, w, h = box

        x1 = cx - w / 2
        y1 = cy - h / 2

        x2 = cx + w / 2
        y2 = cy + h / 2

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale

        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = np.clip(x1, 0, original_width)
        y1 = np.clip(y1, 0, original_height)

        x2 = np.clip(x2, 0, original_width)
        y2 = np.clip(y2, 0, original_height)

        xyxy_boxes.append(
            [x1, y1, x2, y2]
        )

    if len(xyxy_boxes) == 0:

        return []

    xyxy_boxes = np.array(
        xyxy_boxes,
        dtype=np.float32
    )

    # Class-wise NMS

    final_indices = []

    for class_id in np.unique(class_ids):

        indices = np.where(
            class_ids == class_id
        )[0]

        selected = nms(
            xyxy_boxes[indices],
            confidences[indices],
            IOU_THRESHOLD
        )

        final_indices.extend(
            indices[selected]
        )

    detections = []

    for index in final_indices:

        detections.append({

            "box": xyxy_boxes[index],

            "confidence": float(confidences[index]),

            "class_id": int(class_ids[index]),

            "class_name": CLASS_NAMES[class_ids[index]]

        })

    return detections


# ============================================================
# PPE ASSOCIATION
# ============================================================

def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def associate_ppe(person_box, ppe_box, ppe_class):

    px1, py1, px2, py2 = person_box

    cx, cy = box_center(ppe_box)

    person_width = px2 - px1
    person_height = py2 - py1

    if person_width <= 0 or person_height <= 0:

        return False

    relative_x = (cx - px1) / person_width
    relative_y = (cy - py1) / person_height

    if relative_x < -0.10 or relative_x > 1.10:

        return False

    if relative_y < -0.10 or relative_y > 1.10:

        return False

    # Hardhat: Upper region

    if ppe_class == 1:

        return relative_y <= 0.50

    # Safety Vest: Torso region

    elif ppe_class == 2:

        return 0.20 <= relative_y <= 0.85

    return False


# ============================================================
# COMPLIANCE ANALYSIS
# ============================================================

def analyze_compliance(detections):

    persons = [
        d for d in detections
        if d["class_id"] == 0
    ]

    hardhats = [
        d for d in detections
        if d["class_id"] == 1
    ]

    vests = [
        d for d in detections
        if d["class_id"] == 2
    ]

    workers = []

    for person_index, person in enumerate(persons):

        person_box = person["box"]

        has_hardhat = False
        has_vest = False

        # Hardhat association

        for hardhat in hardhats:

            if associate_ppe(
                person_box,
                hardhat["box"],
                1
            ):

                has_hardhat = True

        # Vest association

        for vest in vests:

            if associate_ppe(
                person_box,
                vest["box"],
                2
            ):

                has_vest = True

        # Compliance status

        if has_hardhat and has_vest:

            status = "COMPLIANT"

            color = (0, 255, 0)

        elif not has_hardhat and not has_vest:

            status = "PPE VIOLATION"

            color = (0, 0, 255)

        elif not has_hardhat:

            status = "NO HARDHAT"

            color = (0, 0, 255)

        else:

            status = "NO VEST"

            color = (0, 0, 255)

        workers.append({

            "person": person,

            "worker_id": person_index + 1,

            "has_hardhat": has_hardhat,

            "has_vest": has_vest,

            "status": status,

            "color": color

        })

    return workers


# ============================================================
# DRAW RESULTS
# ============================================================

def draw_results(frame, detections, workers):

    # PPE object boxes

    for detection in detections:

        class_id = detection["class_id"]

        if class_id == 0:

            continue

        x1, y1, x2, y2 = detection["box"]

        confidence = detection["confidence"]

        label = (
            f"{detection['class_name']}: "
            f"{confidence:.2f}"
        )

        color = (255, 255, 0)

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (int(x1), max(int(y1) - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2
        )

    # Worker compliance boxes

    for worker in workers:

        person = worker["person"]

        x1, y1, x2, y2 = person["box"]

        color = worker["color"]

        status = worker["status"]

        worker_id = worker["worker_id"]

        label = f"Worker {worker_id}: {status}"

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            3
        )

        cv2.putText(
            frame,
            label,
            (int(x1), max(int(y1) - 25, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # Statistics

    compliant_count = sum(
        1 for worker in workers
        if worker["status"] == "COMPLIANT"
    )

    violation_count = len(workers) - compliant_count

    return compliant_count, violation_count


# ============================================================
# MAIN REAL-TIME LOOP
# ============================================================

frame_count = 0

total_inference_time = 0

start_time = time.time()

last_report_time = time.time()

last_save_time = time.time()

try:

    while True:

        # ----------------------------------------------------
        # CAPTURE FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print("Frame capture failed!")

            time.sleep(0.1)

            continue

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        image, scale, pad_x, pad_y = preprocess(frame)

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        inference_start = time.time()

        interpreter.set_tensor(
            input_index,
            image
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_index
        )

        inference_time = (
            time.time() - inference_start
        ) * 1000

        total_inference_time += inference_time

        # ----------------------------------------------------
        # POSTPROCESS
        # ----------------------------------------------------

        detections = postprocess(
            output,
            frame,
            scale,
            pad_x,
            pad_y
        )

        # ----------------------------------------------------
        # PPE COMPLIANCE
        # ----------------------------------------------------

        workers = analyze_compliance(
            detections
        )

        # ----------------------------------------------------
        # DRAW RESULTS
        # ----------------------------------------------------

        compliant_count, violation_count = draw_results(
            frame,
            detections,
            workers
        )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        frame_count += 1

        elapsed = time.time() - start_time

        fps = frame_count / elapsed if elapsed > 0 else 0

        # ----------------------------------------------------
        # DISPLAY INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Inference: {inference_time:.1f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Workers: {len(workers)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Compliant: {compliant_count}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Violations: {violation_count}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # ----------------------------------------------------
        # SAVE IMAGE EVERY SECOND
        # ----------------------------------------------------

        if time.time() - last_save_time >= 1:

            cv2.imwrite(
                OUTPUT_IMAGE,
                frame
            )

            last_save_time = time.time()

        # ----------------------------------------------------
        # TERMINAL REPORT EVERY 5 SECONDS
        # ----------------------------------------------------

        if time.time() - last_report_time >= 5:

            avg_inference = (
                total_inference_time / frame_count
            )

            print(
                f"FPS: {fps:.2f} | "
                f"Inference: {inference_time:.2f} ms | "
                f"Avg Inference: {avg_inference:.2f} ms | "
                f"Workers: {len(workers)} | "
                f"Compliant: {compliant_count} | "
                f"Violations: {violation_count}"
            )

            last_report_time = time.time()

        # ----------------------------------------------------
        # FULL-SCREEN HDMI DISPLAY
        # ----------------------------------------------------

        display_frame = cv2.resize(
            frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("\n'q' pressed. Stopping detection...")

            break

# ============================================================
# CLEANUP
# ============================================================

except KeyboardInterrupt:

    print("\nDetection stopped by user.")

except Exception as e:

    print("\nERROR:", e)

finally:

    cap.release()

    cv2.destroyAllWindows()

    print("\nPPE detection stopped.")

    print(
        "Total frames processed:",
        frame_count
    )

    if frame_count > 0:

        print(
            "Average inference time:",
            round(
                total_inference_time / frame_count,
                2
            ),
            "ms"
        )

    print(
        "Latest output image:",
        OUTPUT_IMAGE
    )

    print("Camera released.")
