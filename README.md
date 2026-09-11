# Clothing Classifier

# 👕 Project Title: Clothing Classifier

## 📝 Overview
This is a Deep Learning project using a computer-vision classification model where I predict the category of a clothing item from a single photo. The model recognizes **15 types of clothing** (e.g. Blazer, Jeans, Hoodie, Kemeja, Rok). Instead of training from scratch, it uses **transfer learning** — starting from a network already trained on millions of images and adapting it to clothing. This can be useful for e-commerce catalog tagging, inventory sorting, or any app that needs to auto-label garment photos.

## 🚀 Deployment Link:
https://clothing-classifier-bram.netlify.app/

## 📂 Dataset
Source: https://www.kaggle.com/datasets/ryanbadai/clothes-dataset/data

Size: 7,500 images (15 classes × 500 each)

Description: A balanced clothing image dataset with one folder per class. All 15 categories have exactly 500 images, so no over/undersampling was needed.

## 🔧 Technologies Used
Programming Language: Python, JavaScript

Libraries: PyTorch, torchvision, timm, Pillow, NumPy, pandas, scikit-learn, Matplotlib, FastAPI, Uvicorn, Pydantic, slowapi

Tools: Jupyter Notebook, Docker, Google Cloud Run, Netlify

## 🚀 Workflow

### 1️ Environment Setup
Install the dependencies from `requirements.txt` into a virtual environment. This project uses the **CPU build** of PyTorch (`torch==...+cpu`) so it runs on any machine for local development; the code auto-detects a GPU and uses it unchanged if one is available.

### 2️ Training the Model
The core of the project — a transfer-learning pipeline built around `timm`:

- **Data loading & inspection:** images are loaded with `ImageFolder` (one folder per class) and checked for balance — 15 classes, 500 images each.
- **Model & transforms:** the backbone is **ConvNeXt-Tiny**, pre-trained on ImageNet. Its ImageNet head is swapped for a fresh 15-class head, and the image transforms (resize, crop, normalization) are pulled **from the model's own config** so preprocessing always matches what the backbone expects.
- **Train / validation split:** a stratified 80/20 split (seed = 42) keeps every class balanced across the 6,000 train / 1,500 validation images.
- **Phase 1 — Freeze & warm up:** the backbone is frozen and only the small classifier head is trained (learning rate 1e-3). This is fast on CPU and gets the random head into a sensible state before touching the backbone.
- **Phase 2 — Unfreeze & fine-tune:** the whole network is unfrozen and fine-tuned with a lower learning rate (1e-4) and a **cosine-annealing schedule** that eases the rate toward zero for a clean finish. A save-best-on-validation checkpoint keeps the strongest model.
- **Evaluation:** measured with a per-class classification report and a confusion matrix.
  - ResNet-18 baseline: **~0.71** validation accuracy.
  - **ConvNeXt-Tiny: ~0.82** validation accuracy.
  - Remaining errors cluster in visually similar groups — the jacket family (`Jaket` / `Jaket_Olahraga` / `Jaket_Denim` / `Hoodie`), tops (`Polo` ↔ `Kemeja`), and `Blazer` ↔ `Mantel`.
- **Model saving:** the best model is saved as `best.pt` with `torch.save`, storing the weights **plus** metadata (`model_name`, `classes`, `class_to_idx`, `input_size`) so it can be reloaded for inference on its own.

### 3️ Exploratory Data Analysis (after training)
A separate notebook explores what the image data actually looks like by "tabularizing" it three ways:
1. **Metadata table** — one row per image (class, dimensions, path) for familiar pandas-style inspection.
2. **Raw pixels** — viewing a single image as its underlying grid of numbers.
3. **Feature embeddings** — running images through the model to turn each into a fixed-length feature vector, then visualizing them in 2D with **t-SNE**. Visually similar classes cluster together, which explains the confusions seen in the evaluation.

### 4️ Deployment
- **API (`api/`):** a **FastAPI** service exposing a `/predict` endpoint that accepts an image upload, validates it (JPEG/PNG/WEBP only), runs inference, and returns the predicted label + confidence. It includes rate limiting (slowapi) and CORS for the browser frontend.
- **Docker:** the API is containerized with a slim Python image (CPU torch) for reproducible deployment.
- **Google Cloud Run:** the container is deployed to GCP Cloud Run (region `asia-southeast2`), auto-scaling and publicly reachable.
- **Frontend (`frontend/`):** a simple **HTML / CSS / JS** page hosted on **Netlify** that lets users upload a photo and see the prediction, calling the Cloud Run API.

## 🧾 Conclusion & Suggestion
The model reaches ~82% accuracy on 15 fairly fine-grained clothing categories — a solid result given how visually similar some classes are. Most errors happen exactly where a human might hesitate too (a polo vs a button shirt, a jacket vs a sports jacket).

Suggestions to improve further:
- Use a larger backbone (e.g. ConvNeXt-Base) on a GPU.
- Try a coarse-to-fine (hierarchical) approach for the confusable groups.
- Add more varied images for the hardest classes (the jacket family).
- Proper MLOps deployment architecture that tracks how the model performs in production.

## Thank You For Visiting!!

📬 Connect with me and give me feedback

💼 LinkedIn: https://www.linkedin.com/in/bramantyo-anandaru-suyadi-0b9729208/