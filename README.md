# SmartWasteAI – Intelligent Waste Segregation System

## Project Overview
This repository contains the implementation of **SmartWasteAI**, a computer vision-based waste segregation system developed as part of the CRS Assignment for the *Artificial Intelligence* course (Machine Learning & Deep Learning module).

The project leverages **YOLOv5** for object detection and **MobileNet/EfficientNet** for image classification. The model classifies detected waste into three primary categories:

- **Biodegradable** → Green bin  
- **Recyclable** → Blue bin  
- **Hazardous** → Red bin  

The system is deployed via **Streamlit Cloud**, making it accessible as an interactive web application. By assisting users in identifying the correct disposal method for different types of waste, the project supports the development of smart cities and promotes environmentally responsible behavior.


## Learning Outcomes
Through this project, the following learning objectives were achieved:

- Developed proficiency in applying deep learning architectures (YOLOv5, MobileNet, EfficientNet) to real-world problems.  
- Gained practical insights into the importance of computer vision in sustainable urban development.  
- Implemented a complete end-to-end pipeline including data preprocessing, model training, evaluation, deployment, and user interface design.  
- Designed a Streamlit-based interactive application that allows non-technical users to benefit from AI-powered decision-making.  



## Repository Contents
The repository is organized as follows:

- `train_model.ipynb` – Jupyter Notebook containing dataset preprocessing, model training, and evaluation.  
- `app.py` – Streamlit web application for interactive waste detection and bin recommendation.  
- `models/waste_classifier.pth` – Trained model weights (classification model).  
- `data/` – Dataset, organized into class folders (e.g., glass, plastic, paper, metal, clothes, shoes, battery, organic).  
- `results/` – Contains performance metrics, plots, and evaluation results (e.g., confusion matrix, accuracy plots).  
- `screenshots/` – Screenshots demonstrating application functionality and user interface.  

## Dataset
- **Source**: Dataset provided via course assignment (Drive link).  
- **Classes**: The dataset includes images of waste items belonging to categories such as glass, plastic, paper, metal, clothes, shoes, and batteries.  

### Preprocessing
- All images were resized to a fixed resolution of **224×224 pixels**.  
- Data augmentation techniques such as random rotation, horizontal and vertical flipping, zooming, and brightness adjustments were applied.  
- Dataset split:  
  - 70% Training – for model learning.  
  - 15% Validation – for monitoring model performance during training.  
  - 15% Testing – for final evaluation on unseen data.  


## Model Development
The system is designed as a two-step pipeline:

### 1. Object Detection (YOLOv5)
- YOLOv5 was used to detect waste items in the input image.  
- The model is lightweight and optimized for real-time detection.  

### 2. Classification (MobileNet/EfficientNet)
- Detected waste items were classified into one of the three primary categories: Biodegradable, Recyclable, or Hazardous.  
- MobileNet and EfficientNet were selected due to their balance of accuracy and computational efficiency, making them suitable for deployment on cloud platforms.  

### Evaluation Metrics
- **Accuracy**: Overall percentage of correct predictions.  
- **Confusion Matrix**: Visual representation of correct and incorrect predictions by class.  
- **Precision, Recall, and F1-Score**: Metrics used to evaluate classification reliability and balance across categories.  



## Streamlit Application

link to the model: https://iada-201-2405460--pravargolecha-nf5fmb8xrzfjfrjqtjc3d7.streamlit.app/


The web application, built with **Streamlit**, provides an intuitive interface for end users.

### Features
- Upload an image of a waste item from local storage.  
- Detect the waste object using YOLOv5.  
- Classify the waste into one of the three main categories.  
- Provide a **bin recommendation** based on classification results:  
  - Biodegradable → Green bin  
  - Recyclable → Blue bin  
  - Hazardous → Red bin  
- Display the **confidence score** of the prediction, allowing users to assess the certainty of the model.  


