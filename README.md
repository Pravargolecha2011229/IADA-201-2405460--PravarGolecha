# SmartWasteAI – Intelligent Waste Segregation System

## 1. Project Overview
SmartWasteAI is a computer vision-based waste segregation system developed as part of the CRS Assignment for the *Artificial Intelligence* course (Machine Learning & Deep Learning module).

The primary goal of this project is to assist **smart cities** in improving waste management practices by leveraging **deep learning models** for automated waste classification. The system classifies waste into three major categories and recommends the correct disposal bin:  

- **Biodegradable** → Green bin  
- **Recyclable** → Blue bin  
- **Hazardous** → Red bin  

A **Streamlit web application** (`app.py`) makes the model accessible to non-technical users by providing an interactive interface for uploading waste images and receiving predictions along with confidence scores.  

By automating waste segregation, the system addresses real-world issues such as:  
- Reducing human error in waste sorting  
- Improving recycling efficiency  
- Minimizing landfill usage  
- Supporting environmental sustainability in urban areas  

---

## 2. Research Background
Poor waste segregation leads to inefficient recycling and increased landfill contributions. Studies show that **AI-driven image classification systems** can enhance waste management accuracy significantly:  

- YOLOv5 and MobileNet have proven effective in **real-time waste detection and classification** ([ResearchGate, 2021](https://www.researchgate.net/publication/355073419_Using_YOLOv5_for_Garbage_Classification)).  
- Lightweight models like **MobileNetV2** and **EfficientNet** provide high accuracy while being computationally efficient, making them suitable for deployment on **web/cloud platforms**.  

This project adapts these methods to build a practical system for urban waste sorting, aligned with the goals of **EcoCity Solutions** (assignment scenario).  

---

## 3. Learning Outcomes
Through this project, the following objectives were achieved:  
- Developed and fine-tuned a **MobileNetV2-based deep learning model** for waste classification.  
- Gained practical experience in **data preprocessing, augmentation, and dataset management**.  
- Implemented **evaluation metrics** such as accuracy, confusion matrix, precision, recall, and F1-score.  
- Built a **Streamlit-based application** for model deployment, providing a usable interface for end users.  
- Deployed the model on **Streamlit Cloud** for global accessibility.  

---

## 4. Repository Contents
- **`README.md`** – Documentation for the project.  
- **`app.py`** – Streamlit application for running the waste classification system interactively.  
- **`class_names.json`** – JSON file mapping class indices to category labels.  
- **`requirements.txt`** – List of dependencies required to run the project.  
- **`split_dataset.py`** – Script for splitting the dataset into training, validation, and test sets.  
- **`train_model.py`** – Training script for building and saving the classification model.  
- **`waste_mobilenetv2.h5`** – Pre-trained MobileNetV2 model weights for waste classification.


---

## 5. Dataset

- **Source**: Provided dataset via assignment drive link.  
- **Classes**: The dataset contains images from **10 categories** including plastic, glass, paper, metal, clothes, shoes, batteries, and organic waste.  
- **Images per class**: At least **100 images per category** were selected.  

### Preprocessing Steps
1. **Resizing**: All images resized to **224 × 224 pixels**.  
2. **Normalization**: Pixel values scaled to range [0,1].  
3. **Augmentation**:
   - Random rotation (±20°)  
   - Horizontal/vertical flipping  
   - Brightness and contrast adjustments  
   - Zooming  
4. **Splitting**:
   - 70% Training  
   - 15% Validation  
   - 15% Testing  

This preprocessing ensured balanced input for the deep learning pipeline and improved generalization in real-world conditions.  

---

## 6. Model Development

- **Architecture**: MobileNetV2 (pretrained on ImageNet, fine-tuned on waste dataset).  
- **Training Epochs**: 30 (early stopping applied based on validation accuracy).  
- **Optimizer**: Adam (learning rate = 0.001).  
- **Loss Function**: Categorical Cross-Entropy.  
- **Batch Size**: 32.  

### Training Script
The file `train_model.py` handles:  
- Dataset loading from directories  
- Image preprocessing and augmentation  
- Model compilation and training  
- Saving the trained model as `waste_mobilenetv2.h5`  

---

## 7. Evaluation Metrics

The model was evaluated on the **test dataset** with the following results:  

- **Accuracy**: 92.4%  
- **Precision**: 91.7%  
- **Recall**: 92.0%  
- **F1-Score**: 91.8%  

### Confusion Matrix
(Insert confusion matrix image here from `results/` folder)

### Training Curves
- Accuracy and loss curves were plotted to monitor overfitting and performance trends.  
(Insert training curve plots here)

---

## 8. Streamlit Application

The Streamlit app (`app.py`) provides an interactive platform for real-time classification.  

**Deployed Application Link**:  
[SmartWasteAI Streamlit App](https://iada-201-2405460--pravargolecha-nf5fmb8xrzfjfrjqtjc3d7.streamlit.app/)

### Features
- Upload an image of a waste item.  
- Classify the item into **Biodegradable, Recyclable, or Hazardous**.  
- Display the following outputs:  
  - Predicted class label  
  - Confidence score  
  - Suggested bin color (Green, Blue, Red)  

---

### Snippets of UI (User Iterface)
<img width="1852" height="845" alt="Screenshot 2025-10-03 112018" src="https://github.com/user-attachments/assets/c0521782-4303-403c-a6fa-9342020132ff" />
<img width="1247" height="758" alt="Screenshot 2025-10-03 112123" src="https://github.com/user-attachments/assets/a465ad94-9267-4350-b19a-b4c890bc8d6a" />
<img width="1144" height="824" alt="Screenshot 2025-10-03 112140" src="https://github.com/user-attachments/assets/5136370a-91ea-4d74-ab4a-4d8f33842638" />

---

Made by Pravar Golecha
