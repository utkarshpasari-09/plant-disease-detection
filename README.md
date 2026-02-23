# 🌿 Plant Disease Detection using Deep Learning

An AI-powered web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN) and provides instant predictions through a Streamlit interface.

---

## 🚀 Project Overview

Plant diseases significantly reduce crop yield and quality.  
This project uses **Deep Learning (TensorFlow/Keras)** to automatically classify plant leaf images into **38 disease categories**.

The system allows users to upload an image and instantly receive:
- Disease classification
- Confidence score
- Healthy vs Diseased detection

---

## 🧠 Model Details

- Architecture: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Input Size: `128 x 128 RGB`
- Classes: **38 plant disease categories**
- Dataset Size:
  - Training: **70,295 images**
  - Validation: **17,572 images**
- Dataset Source: PlantVillage (augmented version)

---

## 📂 Project Structure
plant-disease-detection/
│
├── app/
│ └── main.py # Streamlit web app
│
├── model/
│ └── trained_model.keras # Saved trained CNN model
│
├── notebooks/
│ ├── train_plant_disease.ipynb
│ └── Test_Plant_Disease.ipynb
│
├── assets/
│ └── home_page.jpeg
│
├── requirements.txt
├── .gitignore
└── README.md


> ⚠ Dataset is excluded due to size.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection

Create environment (recommended):

conda create -n plantenv python=3.10
conda activate plantenv

Install dependencies:

pip install -r requirements.txt
🧪 Training the Model

Open the training notebook:

notebooks/train_plant_disease.ipynb

Steps:

Load dataset

Build CNN model

Train model

Save model:

model.save("trained_model.keras")
🔍 Testing the Model

Use:

notebooks/Test_Plant_Disease.ipynb

For:

Model evaluation

Prediction visualization

Accuracy analysis

🌐 Run the Web Application

Start the Streamlit app:

streamlit run app/main.py

Then open in browser:

http://localhost:8501
🎯 Features

✔ Upload leaf image for instant prediction
✔ Detects 38 different plant diseases
✔ Confidence score visualization
✔ Clean, responsive UI
✔ Fast inference using saved model
✔ Easily deployable

🛠 Technologies Used

Python

TensorFlow / Keras

Streamlit

NumPy

Pillow

Matplotlib (for training visualization)

📈 Future Improvements

Add treatment recommendations

Mobile camera integration

Deploy on cloud (Streamlit Cloud / AWS)

Grad-CAM visualization for explainability

👨‍💻 Author

Developed as part of an AI/ML project for automated agricultural disease detection.

📜 License

This project is for educational and research purposes.


---

## ✅ After Creating README

Run:

```bash
git add README.md
git commit -m "Added README"
git push
