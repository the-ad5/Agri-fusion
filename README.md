# Agri-fusion
AI-powered crop intelligence system that predicts soybean crop health using UAV (drone) imagery. Helps in early disease detection and supports smart farming decisions.
AgriFusion-AI: UAV-Based Soybean Crop Health Prediction

Overview
AgriFusion-AI is an AI-powered crop intelligence system designed to predict soybean crop health and disease conditions using UAV (drone) imagery. By analyzing large-scale field images, the system enables early detection of crop stress and diseases, helping farmers take timely action and reduce yield loss. This project combines computer vision with real-world agricultural needs to create a meaningful and practical solution.

Problem Statement
Farmers often struggle to detect crop diseases early, especially across large agricultural fields. Manual inspection is time-consuming, inconsistent, and prone to human error. This project addresses that issue by automatically analyzing UAV images to detect crop health conditions and provide fast, data-driven insights.

Why This Project Matters
Early detection of diseases helps prevent crop loss and improves productivity. This system reduces dependency on manual monitoring and supports sustainable farming practices. It is scalable and can be applied in real-world agricultural environments.

Dataset Description
Dataset Name: Soybean UAV Image Dataset
Source: Mendeley Data
Link: https://data.mendeley.com/datasets/hkbgh5s3b7/1

The dataset consists of RGB drone images of soybean fields. Unlike close-up leaf datasets, this dataset captures real farming conditions, making the model more reliable in practical scenarios.
Compliance: Public dataset, no personal or sensitive data used.

Dataset Structure
The dataset follows a standard image classification format with separate folders for training and validation data. Each folder contains subfolders for different classes such as Healthy Soybean, Pest Attack, Mosaic, and Rust.

Dataset Summary
Total Classes: 4
Healthy Soybean: 224 images
Pest Attack: 632 images
Mosaic: 617 images
Rust: 800 images
Total Training Samples: 2273
Validation Samples: 569
The dataset reflects real-world class imbalance, making the problem more realistic and challenging.

Model Details
Model Used: MobileNetV2
Type: Convolutional Neural Network (CNN)
MobileNetV2 is a lightweight and efficient model that performs well on image classification tasks while requiring less computational power. It is suitable for real-time and edge-based agricultural applications.

Project Workflow
The system follows a structured pipeline: data collection, preprocessing, model training, validation, model export to ONNX format, and final inference on test images.

Project Structure
train.py handles model training
preprocess.py handles data loading and transformations
export_onnx.py converts the trained model to ONNX format
infer.py performs predictions using the ONNX model
model folder stores trained models
logs folder stores training logs
class_names.json stores label mappings

How to Run the Project
Step 1: Clone the repository
git clone <your-repo-link>
cd agri-fusion

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Train the model
python train.py

Step 4: Export model to ONNX
python export_onnx.py

Step 5: Run inference
python infer.py

Inference Process
The ONNX model is loaded, class labels are mapped, the input image is preprocessed, and predictions are generated along with confidence scores.

Requirements
python >= 3.8
torch
torchvision
opencv-python
Pillow
numpy
scikit-learn
tqdm
onnx
onnxruntime
matplotlib

Key Learnings
Transfer learning improves performance with limited data. Data augmentation enhances generalization. Lightweight models are more practical for deployment. Converting models to ONNX makes them portable and scalable.

Challenges Faced
Handling class imbalance in the dataset
Avoiding overfitting during training
Maintaining consistency between training and inference preprocessing
Selecting a model that balances accuracy and efficiency

Development Process
The project started with a basic approach and was improved using a pretrained MobileNetV2 model. Data augmentation techniques were added to improve performance. Validation tracking was implemented to save the best model. Finally, ONNX export was integrated to enable deployment.

Limitations
Limited dataset size may affect generalization
Performance may vary under extreme field conditions
No real-time deployment implemented yet

Future Improvements
Integration with real-time drone systems
Expanding the dataset with more crop types
Using advanced architectures for improved accuracy
Developing a user-friendly interface for farmers

Compliance
Public dataset used
Single AI model implemented
CNN-based approach followed
No rule-based logic used
ONNX export completed
Inference successfully tested


This project demonstrates that a good solution is not defined by complexity but by its ability to solve a real-world problem effectively. By combining AI with agriculture, this system aims to make farming smarter, faster, and more efficient.
Note: Model files are not uploaded due to size limitations. They can be generated using train.py.
