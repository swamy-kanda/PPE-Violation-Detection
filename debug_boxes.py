import cv2
import numpy as np

from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(
    model_path="best.tflite",
    num_threads=4
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

frame = cv2.imread("construction-worker-site.jpg")

h, w = frame.shape[:2]

scale = min(640 / w, 640 / h)

nw = int(w * scale)
nh = int(h * scale)

resized = cv2.resize(frame, (nw, nh))

pad_x = (640 - nw) // 2
pad_y = (640 - nh) // 2

image = np.zeros((640, 640, 3), dtype=np.uint8)

image[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = image.astype(np.float32) / 255.0

image = np.transpose(image, (2, 0, 1))

image = np.expand_dims(image, axis=0)

interpreter.set_tensor(
    input_details[0]["index"],
    image
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]["index"]
)

predictions = output[0].T

scores = predictions[:, 4:]

confidences = np.max(scores, axis=1)

class_ids = np.argmax(scores, axis=1)

indices = np.argsort(confidences)[::-1][:10]

print("\nTop 10 predictions:\n")

for i in indices:

    box = predictions[i, :4]

    confidence = confidences[i]

    class_id = class_ids[i]

    print(
        f"Class: {class_id}, "
        f"Confidence: {confidence:.3f}, "
        f"Raw box: {box}"
    )
