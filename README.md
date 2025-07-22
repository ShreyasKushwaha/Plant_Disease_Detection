# 🌿 Plant Disease Recognition System

Welcome to the **Plant Disease Recognition System**, a deep learning-based solution to identify plant diseases from leaf images. This tool empowers farmers, researchers, and agritech developers to detect diseases early, ensuring healthier crops and improved yield.

---

## 🔍 Overview

This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify leaf images into **38 different plant disease categories**. A **Streamlit-based web application** allows users to upload images and instantly get predictions with model confidence scores. 

**🧠 Model Accuracy:** ~96% on the validation set  
**📊 Evaluation Metrics:** Precision, Recall, F1-Score, Confusion Matrix  
**📁 Dataset Size:** 87,900+ images (train, validation, and test)

📌 **Kaggle Notebook/Project:** [View on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

🗂️ Project Structure
dataset/
└── plant_dataset/
    ├── main.py
    ├── Train_plant_disease.ipynb
    ├── Test_Plant_Disease.ipynb
    ├── trained_model.keras
    ├── training_hist.json
    ├── train/
    ├── valid/
    ├── test/
    └── README.md

 🏋️‍♂️ Training the Model
 
Run the Jupyter notebook:

Train_plant_disease.ipynb

This will train the CNN model and save the weights in .keras format.


## ✨ Features

- ✅ **Image Classification:** Detects 38 plant diseases using RGB leaf images  
- 🧠 **Deep Learning Model:** Custom CNN built with TensorFlow and Keras  
- 🌐 **Web Interface:** User-friendly Streamlit app for uploading and predicting  
- 📈 **Visualizations:** Accuracy/loss curves and confusion matrix  
- 🔄 **Data Augmentation:** Enhances training with offline augmentation techniques  

---

## 📂 Dataset

- **Source:** [Original GitHub Dataset – PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
- **Structure:**
  - `train/` – 70,295 images  
  - `valid/` – 17,572 images  
  - `test/` – 33 images  
- **Classes:** 38 plant disease categories

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.10+
- TensorFlow
- Streamlit
- Matplotlib, Seaborn, Pandas, scikit-learn, Pillow

### 📦 Installation

```bash
pip install tensorflow streamlit matplotlib seaborn pandas scikit-learn pillow
