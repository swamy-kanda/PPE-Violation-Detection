from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = "best.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nModel loaded successfully!")

print("\nInput Details:")
for detail in input_details:
    print("Name:", detail["name"])
    print("Shape:", detail["shape"])
    print("Data Type:", detail["dtype"])

print("\nOutput Details:")
for detail in output_details:
    print("Name:", detail["name"])
    print("Shape:", detail["shape"])
    print("Data Type:", detail["dtype"])
