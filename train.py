# train.py
# Training script for soybean crop classification
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from preprocess import get_dataloaders   

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..","..", "Soyabean_UAV-Based_Image_Dataset_Split")

# DATA_DIR = r"..\..\Soyabean_UAV-Based_Image_Dataset_Split"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10


PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "training_logs.txt")

os.makedirs(LOG_DIR, exist_ok=True)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# ---------------- MODEL SETUP ----------------
def setup_model(num_classes, device, lr=LEARNING_RATE):
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"MobileNetV2 initialized with {num_classes} classes")
    return model, criterion, optimizer


# ---------------- TRAINING LOOP ----------------
def train_model(model, dataloaders, dataset_sizes, criterion,
                optimizer, num_epochs=NUM_EPOCHS,
                log_file_path=LOG_FILE):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(log_file_path, "w") as log_file:

        def log(msg):
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush() 

        log(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Epochs: {num_epochs}\n")

        for epoch in range(num_epochs):
            log(f"Epoch {epoch+1}/{num_epochs}")
            log("-" * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    if phase not in dataloaders:
                        log("Validation loader not found. Skipping val phase.")
                        continue
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                log(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            log("")

        time_elapsed = time.time() - since
        log(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        log(f"Best Val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Best model saved to: {MODEL_PATH}")
    return model


# ---------------- MAIN ENTRY ----------------
if __name__ == "__main__":

    dataloaders, dataset_sizes, NUM_CLASSES, idx_to_class = get_dataloaders(
        DATA_DIR, IMAGE_SIZE, BATCH_SIZE
    )

    model_ft, criterion, optimizer_ft = setup_model(NUM_CLASSES, DEVICE)

    model_ft = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        NUM_EPOCHS
    )
