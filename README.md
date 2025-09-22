# SmartWasteAI – Intelligent Waste Segregation System

## Project Overview
SmartWasteAI is a computer vision-based waste segregation system developed as part of the CRS Assignment for the *Artificial Intelligence* course (Machine Learning & Deep Learning module).

The system leverages **MobileNetV2** for image classification and organizes waste into three categories:
- **Biodegradable** → Green bin  
- **Recyclable** → Blue bin  
- **Hazardous** → Red bin  

A **Streamlit web application** (`app.py`) makes the system accessible for end-users, allowing them to upload an image of waste and receive both a classification result and a recommended disposal bin.


## Repository Contents

- **`README.md`** – Documentation for the project.  
- **`app.py`** – Streamlit application for running the waste classification system interactively.  
- **`class_names.json`** – JSON file mapping class indices to category labels.  
- **`requirements.txt`** – List of dependencies required to run the project.  
- **`split_dataset.py`** – Script for splitting the dataset into training, validation, and test sets.  
- **`train_model.py`** – Training script for building and saving the classification model.  
- **`waste_mobilenetv2.h5`** – Pre-trained MobileNetV2 model weights for waste classification.  


## Dataset

- **Source**: Dataset provided via the course assignment (Drive link).  
- **Classes**: Includes categories such as glass, plastic, paper, metal, clothes, shoes, battery, and organic waste.  

### Preprocessing
- Images resized to **224 × 224 pixels**.  
- Augmentation applied: rotation, flipping, zoom, brightness adjustments.  
- Split performed using `split_dataset.py`:  
  - 70% Training  
  - 15% Validation  
  - 15% Testing  



## Model Development

- **Architecture**: MobileNetV2 (pretrained on ImageNet, fine-tuned for waste classification).  
- **Saved Model**: Stored as `waste_mobilenetv2.h5`.  
- **Training Script**: `train_model.py` handles dataset loading, preprocessing, training, and saving the model.  

### Evaluation Metrics
- **Accuracy**: Percentage of correct predictions.  
- **Confusion Matrix**: Distribution of correct vs. incorrect classifications.  
- **Precision, Recall, and F1-Score**: Evaluated across classes for balanced performance.  


## Streamlit Application

Link to the model: https://iada-201-2405460--pravargolecha-nf5fmb8xrzfjfrjqtjc3d7.streamlit.app/

The Streamlit app (`app.py`) allows interactive classification.

### Features
- Upload an image of waste.  
- Run classification using the trained MobileNetV2 model.  
- Display:  
  - **Predicted class** (Biodegradable, Recyclable, Hazardous)  
  - **Confidence score**  
  - **Bin recommendation** (Green, Blue, or Red)  


