
# Face Detection & Recognition using CNNs and Transfer Learning

This project implements a face detection and recognition system using Convolutional Neural Networks (CNNs) with transfer learning. The system is trained on the **Pins Face Recognition (FR) Dataset** to identify and recognize individual faces. The project covers the complete pipeline from data preprocessing to model training, validation, and real-time face recognition.

## Project Overview

This face recognition system leverages CNNs, which are effective in handling image data, along with transfer learning to achieve high accuracy with limited data. By using a pre-trained model (such as ResNet50 or VGG16) as a base, we fine-tune it for face recognition, thus saving training time and improving performance. This system can be used in various applications such as secure access control, social media tagging, and more.

## Installation

To set up and run this project locally, please follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Face-Recognition-CNN.git
   cd Face-Recognition-CNN
   ```

2. **Install Dependencies**:
   Make sure you have Python installed, then install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project uses the **Pins FR Dataset**, which contains labeled images of individual faces. The dataset should be organized into `train` and `test` directories within the `data` folder as follows:

   - `data/train`: Contains subfolders, each representing a class (person), with images for each person.
   - `data/test`: Similar structure to `train`, used for testing and evaluation.

**Note**: Please download the dataset and structure it accordingly before running the project.

## Project Steps

### 1. Data Preprocessing

Data preprocessing is essential to ensure model accuracy and performance. The `data_preprocessing.py` script handles these tasks:

   - **Image Resizing**: All images are resized to a uniform dimension suitable for the model.
   - **Image Normalization**: Each image is normalized to a [0, 1] range to improve convergence.
   - **Data Augmentation**: To increase variability, we apply random rotations, flips, and scaling.

Run the preprocessing step with:
```bash
python src/data_preprocessing.py
```

### 2. Model Training

We use a pre-trained CNN (ResNet50 or VGG16) as a feature extractor, adding custom layers to adapt it to our face recognition task. The model is then trained on the processed dataset.

   - **Optimizer**: Adam or SGD optimizer for efficient convergence.
   - **Loss Function**: Cross-entropy loss to handle multi-class classification.
   - **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-score are tracked during training.

To train the model, run:
```bash
python src/model.py
```

Model checkpoints are saved periodically in the `models/` directory to prevent loss of progress in case of interruptions.

### 3. Model Evaluation

The trained model is evaluated on both validation and test datasets to assess its performance. The `evaluate.py` script generates detailed metrics:

   - **Confusion Matrix**: Visualizes the model's classification accuracy across classes.
   - **Precision, Recall, F1-score**: Provides a comprehensive performance assessment.
   - **ROC and Precision-Recall Curves**: Offers insights into model thresholds and performance.

Run the evaluation script:
```bash
python src/evaluate.py
```

Evaluation graphs and metrics are saved in the `reports` directory.

### 4. Real-time Prediction

For real-time face recognition, we use the trained model to make predictions on new images or live video input. The `predict.py` script loads the saved model and processes input from images or video frames to identify individuals.

To run real-time predictions on a video stream or images:
```bash
python src/predict.py
```

This script supports webcam input or can be modified for other sources. Predictions are displayed on the screen with bounding boxes around recognized faces.

## Evaluation Metrics

The following metrics are computed and visualized to evaluate the model:

   - **Accuracy**: The proportion of correctly identified faces across all classes.
   - **Precision and Recall**: Measures of model performance in identifying individual faces without errors.
   - **F1-Score**: The harmonic mean of precision and recall, providing a balanced measure of the model's accuracy.
   - **Confusion Matrix**: A matrix showing true vs. predicted labels to analyze misclassifications.
   - **ROC and Precision-Recall Curves**: These plots provide insights into the model's threshold sensitivity and performance in various conditions.

All metrics and plots are saved in the `reports` directory for analysis.

## Usage in Real-Time Scenarios

This model can be integrated into real-time systems where facial recognition is required. For instance, to deploy it as a secure login system, the model can be connected to a webcam, recognize faces in real time, and authorize individuals based on their identification.

Deployment steps:

1. Ensure you have a compatible webcam or video feed.
2. Run `predict.py` and follow on-screen instructions for input.
3. The script will display real-time predictions along with labels on identified faces.

---

**Note**: The project requires GPU support for real-time performance. Ensure a CUDA-compatible GPU is available for efficient model inference.

## Future Improvements

While the model performs well, here are potential enhancements:

   - **Data Augmentation**: Further augment the data with lighting adjustments and background variations.
   - **Model Optimization**: Experiment with more advanced architectures like EfficientNet for even better accuracy.
   - **Real-Time Enhancements**: Integrate a face-tracking algorithm for smoother real-time recognition.

## Credits

This project was developed using TensorFlow/Keras and relies on a pre-trained model from ImageNet. Special thanks to the contributors of the Pins FR Dataset and the open-source community for their libraries and tools.
