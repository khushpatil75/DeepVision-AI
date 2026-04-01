# 🧠 DeepVision AI: Multi-Model Handwriting Recognition

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)
![Vanilla JS](https://img.shields.io/badge/Vanilla_JS-Frontend-F7DF1E.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview
**DeepVision AI** is a full-stack Machine Learning application designed to recognize and classify handwritten characters across multiple scripts, including **Devanagari**, **Sanskrit**, and **English (EMNIST)**. 

Built as a comprehensive B.Tech project, this application features a custom-built, lightweight browser frontend where users can draw characters in real-time. The backend routes these drawings through three distinct neural network architectures simultaneously, allowing for a real-time comparison of model accuracy, generalization, and performance.

## ✨ Key Features
* **Interactive Digital Canvas:** A Vanilla JavaScript frontend featuring a responsive drawing board with dynamic brush strokes.
* **Smart Pre-processing Pipeline:** Features an auto-cropping algorithm and dynamic color-inversion (handling white-canvas to black-background mapping) to ensure user input perfectly matches the training data distribution.
* **Real-Time Inference:** Fast, asynchronous API calls powered by FastAPI to deliver instant predictions.
* **Multi-Model Comparison:** Evaluates user input against three distinct architectures:
  1. **Baseline CNN:** A lightweight Convolutional Neural Network for fast, shape-based geometric feature extraction.
  2. **Vision Transformer (ViT):** Utilizing `vit_tiny_patch16_224` to capture global context and long-range dependencies within the character strokes.
  3. **Custom Hybrid Architecture:** A state-of-the-art fusion model leveraging a **ResNet18 backbone** for localized feature extraction, fed into a **Transformer Encoder** layer for global spatial awareness.

## 🛠️ Technology Stack
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Canvas API)
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** PyTorch, Torchvision, Timm (Torch Image Models)

## 📂 Project Structure

```text
MINOR_FINAL/
│
├── frontend/
│   ├── index.html       # Main UI structure
│   ├── style.css        # Responsive styling and canvas layout
│   └── script.js        # Canvas drawing logic and API fetching
│
├── Models/              # Pre-trained .pth weight files
│   ├── Devanagari/
│   ├── Sanskrit/
│   └── EMNIST/
│
├── main.py              # FastAPI application and PyTorch inference logic
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── LICENSE.txt          # Custom usage rights
```
## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/khushpatil75/DeepVision-AI.git](https://github.com/khushpatil75/DeepVision-AI.git)
cd DeepVision-AI
```
2. Setup the Python Environment
It is highly recommended to use a virtual environment to prevent dependency conflicts.
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. Install Dependencies
Install all required machine learning and backend libraries.
```Bash
pip install -r requirements.txt
```
4. Run the Backend Server
Start the FastAPI server using Uvicorn.
```Bash
uvicorn main:app --reload
```
The API will be available at http://localhost:8000.
5. Launch the Frontend
```bash
Because the frontend is built with Vanilla web technologies, simply open frontend/index.html in your preferred web browser, or use a tool like VS Code's "Live Server" extension.
```

## 📊 Model Training & Pre-Processing
The models were trained using PyTorch in a Google Colab environment. The data pre-processing pipeline focuses on standardizing inputs for the neural networks, ensuring consistency between the dataset images and the digital canvas inputs. The core transformations include:
* **Grayscale to RGB Conversion:** Padding single-channel images to 3 channels to satisfy ResNet and ViT architecture requirements.
* **Resizing:** Scaling all inputs to `224x224` pixels.
* **Normalization:** Standardizing pixel values using ImageNet standard means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.

## ⚖️ License & Copyright
**Copyright (c) 2026 Khush Patil. All Rights Reserved.**
This software is provided for personal, educational, and non-commercial viewing purposes only. Unauthorized modification, distribution, or presentation of this codebase as your own academic or professional work is strictly prohibited. Please see the `LICENSE.txt` file for full details.