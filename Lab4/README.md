# Face Detection & Recognition on Pins FR Dataset using CNNs & Transfer Learning

This project focuses on developing a real-time face detection and recognition system using deep learning and transfer learning techniques. The system is designed to detect and recognize faces in real-time using the Pins Face Recognition dataset from Kaggle. This project is structured to leverage CNN models with transfer learning to achieve high accuracy and robust performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Project Overview
This project is divided into two main phases: face detection and face recognition. The system leverages OpenCV’s deep learning detector, TensorFlow, and transfer learning techniques to accurately detect and recognize faces in real time.

## Problem Statement
### Objective:
Develop a face recognition system using deep learning and transfer learning on the Pins Face Recognition dataset from Kaggle. The system will involve real-time face detection and recognition capabilities, ensuring efficient and accurate performance.

### Dataset
- Dataset Link: [Pins Face Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition)

### Steps:
1. **Face Detection**: Implement face detection using OpenCV’s deep learning-based detector, utilizing the UINT8 model and previous architecture and weights from prior sessions.
2. **Dataset Preparation**: Crop faces using the OpenCV DNN module and create a modified dataset of only cropped faces.
3. **Face Recognition**: Use any pre-trained model and apply transfer learning using TensorFlow with VGG16, ResNet50, or Inception.
4. **Model Optimization**: Improve accuracy with additional dense and softmax layers and regularization techniques to attain over 85% accuracy on validation data.
5. **Unidentified Face Detection**: Enable the model to recognize unclassified faces using either:
   - Cosine similarity for a "not identified" class, or
   - A dataset of 10,000-70,000 non-defined images.
6. **Real-Time Functionality**: Ensure compatibility with a live camera feed.
7. **Bonus Task**: Develop a multi-input, multi-output (MIMO) model to perform face detection and recognition simultaneously on a live stream.

### Expected Output:
The system should detect and recognize faces in real-time, achieving at least 85% accuracy on the validation dataset.

### Deliverables:
- Comprehensive repository detailing each phase of the process.
- Complete code documentation and README file.
- Guide on running the system in real-time with all evaluation metrics and graphs for both validation and test datasets.

## Features
- **Face Detection**: Detects faces using OpenCV and CNNs.
- **Face Recognition**: Identifies faces with high accuracy using transfer learning.
- **Data Augmentation**: Enhances dataset with data augmentation for improved model generalization.
- **Real-Time Compatibility**: Designed to function in real-time with live camera feeds.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Alsafy01/Deep-Learning/edit/main/Lab4
    ```
2. Navigate to the project directory:
    ```bash
    cd face-detection-recognition
    ```

## Usage
1. **Data Preparation**: Load and preprocess the dataset.
2. **Training**: Train the model on the cropped dataset.
3. **Inference**: Run the model for real-time face recognition on test images or live feed.

Run the script as follows:
```bash
python Face_Detection_&_Recognition_on_Pins_FR_Dataset_using_CNNs_&_Transfer_Learning.py
