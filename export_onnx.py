# export_onnx.py
import torch
import os
import json
from train import setup_model
from preprocess import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DATA_DIR = r"Soyabean_UAV-Based_Image_Dataset_Split"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "Soyabean_UAV-Based_Image_Dataset_Split")
IMAGE_SIZE = 224
BATCH_SIZE = 32

# --- LOAD DATA ---
dataloaders, dataset_sizes, NUM_CLASSES, idx_to_class = get_dataloaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
class_names = list(idx_to_class.values())

# --- LOAD MODEL ---
model_ft, _, _ = setup_model(NUM_CLASSES, DEVICE)
model_ft.load_state_dict(torch.load("model/best_model.pth"))
model_ft.eval()

# --- EXPORT TO ONNX ---
os.makedirs("model", exist_ok=True)
onnx_path = "model/final_model.onnx"
dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

torch.onnx.export(
    model_ft,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"ONNX model exported to: {onnx_path}")

with open("class_names.json","w") as f:
    json.dump(class_names, f)
print("Class names saved to class_names.json")
