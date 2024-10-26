
# CNN Classifier for Fashion MNIST

## Objective

The objective of this project is to develop a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The model aims to accurately predict the category of fashion items, ranging from T-shirts to shoes, by learning to recognize patterns and features in the images.

## Dataset Description

The Fashion MNIST dataset is a collection of grayscale images of fashion products. It serves as a more challenging alternative to the classic MNIST dataset of handwritten digits. The dataset consists of:
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

Each image is 28x28 pixels, and there are 10 categories:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Model Architecture

The model is composed of several convolutional and pooling layers followed by fully connected layers. The key components include:
- **Convolutional Layers**: Extract features from the input images.
- **Max Pooling Layers**: Reduce the spatial dimensions of the features.
- **Dropout Layers**: Prevent overfitting by randomly dropping units during training.
- **Dense Layers**: Fully connected layers for classification.

### Architecture Details
1. **Input Layer**: 28x28x1 (grayscale image)
2. **Conv2D + MaxPooling** layers (with increasing filter size)
3. **Dropout**
4. **Flatten**
5. **Dense + Dropout**
6. **Output Layer**: Softmax activation

## Training

The model was trained with the following parameters:
- **Batch size**: 32
- **Epochs**: 50 (early stopping applied)
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy

### Learning Rate Scheduler
A learning rate reduction strategy was applied when validation accuracy plateaued, ensuring a more refined training process.

### Early Stopping
To avoid overfitting, the training stopped automatically when there was no improvement in validation loss for 10 epochs.

## Results

### Training and Validation Accuracy

![Training and Validation Accuracy](./path/to/accuracy_plot.png)

### Training and Validation Loss

![Training and Validation Loss](./path/to/loss_plot.png)

### Test Accuracy
- **Final Test Accuracy**: 88.25%

### Classification Report

| Class         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| T-shirt/top   | 0.84      | 0.84   | 0.84     |
| Trouser       | 0.99      | 0.97   | 0.98     |
| Pullover      | 0.79      | 0.80   | 0.80     |
| Dress         | 0.89      | 0.89   | 0.89     |
| Coat          | 0.77      | 0.83   | 0.80     |
| Sandal        | 0.97      | 0.95   | 0.96     |
| Shirt         | 0.70      | 0.65   | 0.68     |
| Sneaker       | 0.93      | 0.96   | 0.95     |
| Bag           | 0.98      | 0.97   | 0.98     |
| Ankle boot    | 0.96      | 0.96   | 0.96     |

- **Overall Accuracy**: 88%
- **Macro Average**: 88%
- **Weighted Average**: 88%

## Steps to Run the Code in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the project notebook (`cnn_fashion_mnist.ipynb`) to Colab.
3. Ensure the following files are available:
   - `train.py`
   - `evaluate.py`
   - `requirements.txt`
4. If you are running the code for the first time, run the following command to install dependencies:
    ```python
    !pip install -r requirements.txt
    ```
5. Execute each cell in the notebook step by step to train and evaluate the model.

## Dependencies and Installation Instructions

Make sure you have Python 3.x installed. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### Requirements
- `TensorFlow`
- `NumPy`
- `Matplotlib`
- `scikit-learn`

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/username/cnn-fashion-mnist.git
    cd cnn-fashion-mnist
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. To train the model, run:
    ```bash
    python train.py
    ```

4. To evaluate the model on the test set, run:
    ```bash
    python evaluate.py
    ```

## Conclusion

The CNN model performed well on the Fashion MNIST dataset, achieving a final test accuracy of 88.25%. While the model shows strong performance across most categories, there is room for improvement, especially in classes like "Shirt". Future work could involve experimenting with different architectures, such as ResNet or EfficientNet, to further improve classification accuracy.

## Future Improvements

1. Experiment with more complex models like ResNet or EfficientNet.
2. Implement transfer learning for faster training.
3. Tune hyperparameters such as learning rate, batch size, and dropout rates.
