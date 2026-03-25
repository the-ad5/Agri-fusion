# infer.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json
import os

# --- CONFIG ---
ONNX_MODEL_PATH = "model/final_model.onnx"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_PATH = "test.jpg"  # Replace with any image path
IMAGE_SIZE = 224

# --- LOAD CLASS NAMES ---
with open(CLASS_NAMES_PATH,"r") as f:
    class_names = json.load(f)

# --- ONNX SESSION ---
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- LOAD IMAGE ---
img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0).numpy()

# --- INFERENCE ---
outputs = session.run([output_name], {input_name: img})[0]
pred_class_idx = np.argmax(outputs, axis=1)[0]
probs = np.exp(outputs)/np.sum(np.exp(outputs), axis=1, keepdims=True)
confidence = np.max(probs)

print(f"Predicted class index: {pred_class_idx}")
print(f"Predicted label: {class_names[pred_class_idx]}")
print(f"Confidence: {confidence:.4f}")
