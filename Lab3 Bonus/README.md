# Multiclass Classification with Pins Face Dataset

## Objective

The objective of this project is to build a multiclass classification model using Keras to classify facial images of different celebrities from the Pins Face Dataset. The goal is to achieve an accuracy of over 85% in identifying different individuals based on their facial features.

## Dataset Description

The Pins Face Dataset contains multiple facial images for each person, where each class corresponds to a different individual. The dataset includes images of **105 celebrities** and consists of a total of **17,533 images**. Each person has a varying number of images, allowing the model to learn robust features for classification.

- **Link to Dataset:** [Pins Face Dataset](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition)

### Dataset Summary
- **Total number of images:** 17,533
- **Number of classes:** 105
- **Images per class:** Varies from 86 to 237 images

## Instructions for Running the Code

1. **Clone the Repository:**
   Clone this repository to your local machine.

   ```bash
   git clone https://github.com/your_username/pins_face_recognition.git
   cd pins_face_recognition
   ```

2. **Set Up the Environment:**
   Ensure you have Python 3.x installed. You can create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Install the required packages listed in `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required Files:**
   You will need the following files for face detection. Download them and place them in the project root directory:
   - [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel?raw=true)
   - [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt?raw=true)

5. **Run the Jupyter Notebook:**
   Open the Jupyter notebook for model training.

   ```bash
   jupyter notebook notebooks/model_training.ipynb
   ```

6. **Execute the Cells:**
   Follow the instructions in the notebook to execute each cell and train the model.

## Dependencies and Installation Instructions

### Required Libraries
To set up the project environment, you need the following libraries:

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Expected Results and Accuracy Target

The model aims to achieve at least **85% accuracy** on the validation dataset. Performance metrics such as precision, recall, and F1-score will be provided in the classification report generated after model evaluation.

## Repository Structure

```plaintext
├── data/
│   ├── processed_data/          # Directory for processed image data
│   └── pins_face_dataset/       # Original dataset directory (if downloaded)
├── notebooks/
│   └── Multiclass Classification with Pins Face Dataset Using Keras (ANN Only).ipynb     # Jupyter notebook for model training
├── requirements.txt              # Required Python packages
├── res10_300x300_ssd_iter_140000.caffemodel  # Face detection model file
└── deploy.prototxt               # Face detection model configuration file
└── README.md                    # Project documentation
```

## Conclusion

This project demonstrates the use of neural networks for image classification tasks. With proper tuning and validation, this model can be enhanced for better performance. 

