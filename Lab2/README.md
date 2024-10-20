
# Binary Classification of Heart Disease using a Multi-Layer Perceptron (MLP) with Keras

## Objective
The goal of this project is to build a binary classification model using a Multi-Layer Perceptron (MLP) in Keras to predict heart disease based on various health indicators from the provided dataset. The performance of the model is evaluated using accuracy, confusion matrix, precision, recall, F1-score, and ROC-AUC curve. The model is also visualized using loss curves over epochs and tensorboard.

## Dataset Description
The dataset contains health indicators related to heart disease, such as blood pressure, cholesterol levels, and lifestyle choices. The target variable is `HeartDiseaseorAttack`, which indicates whether an individual has heart disease or not. The dataset includes 22 columns.

### Sample Columns:
- **HeartDiseaseorAttack**: 0 (No heart disease), 1 (Heart disease present) - [Target variable]
- **HighBP**: 0 or 1 - High blood pressure
- **BMI**: Body Mass Index
- **Smoker**: 0 or 1 - Whether the person is a smoker
- **Diabetes**: 0 or 1 - Whether the person has diabetes
- **PhysActivity**: 0 or 1 - Physical activity status
- **Age**: Age of the person
- **Income**: Income level (categorical)
- And more...

## Steps to Run the Code

### 1. Installation of Dependencies

Before running the code, ensure that the following dependencies are installed. You can install the required libraries using `pip`.

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib keras-tuner
```

Alternatively, you can create a `requirements.txt` file and install dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
tensorflow==2.9.0
pandas==1.3.3
numpy==1.21.2
scikit-learn==1.0.1
matplotlib==3.4.3
keras-tuner==1.1.0
```

### 2. Prepare the Dataset

- Load the dataset (e.g., `heart_disease_data.csv`) into a Pandas DataFrame.
- Preprocess the dataset:
  - Split the dataset into features `X` and target `y`.
  - Scale the features using `StandardScaler`.
  - Split the dataset into training and testing sets (e.g., 80% training, 20% testing).

### 3. Build and Train the MLP Model

- Define an MLP model using Keras' `Sequential` API.
- Add hidden layers with regularization (L1 and L2) and batch normalization.
- Use dropout layers to prevent overfitting.
- Compile the model with binary cross-entropy loss and `Adam` optimizer.
- Use callbacks for early stopping and learning rate reduction on plateau.
- Train the model, using validation data to monitor progress.

### 4. Evaluate the Model

- Evaluate the model using metrics like accuracy, confusion matrix, precision, recall, and F1-score.
- Plot the ROC-AUC curve to measure performance.
- Visualize the loss curves over training epochs.

### 5. Visualize with TensorBoard

- To use TensorBoard for visualizing model training, include the following steps:

  - Install TensorBoard:
    ```bash
    pip install tensorboard
    ```

  - Start TensorBoard while running the notebook:
    ```python
    from tensorflow.keras.callbacks import TensorBoard
    import time

    log_dir = f'logs/fit/{time.strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Add the callback to model fitting
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32,
                        callbacks=[early_stopping, reduce_lr, tensorboard_callback])
    ```

  - Run the following command in the terminal to start TensorBoard:
    ```bash
    tensorboard --logdir=logs/fit
    ```

  - Open TensorBoard in your browser at the URL displayed in the terminal (typically http://localhost:6006).

### 6. Generate and Plot ROC-AUC Curve

You can evaluate your model's performance by generating a ROC-AUC curve as follows:

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Predict probabilities for the test set
y_pred_proba = model.predict(X_test)

# Generate ROC-AUC values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC-AUC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
```

### 7. Saving and Loading the Model

After training, you can save your model for later use:

```python
# Save the model
model.save('heart_disease_mlp_model.h5')

# Load the saved model
from tensorflow.keras.models import load_model
model = load_model('heart_disease_mlp_model.h5')
```


## Outputs

1. **Confusion Matrix**:
   - A matrix showing true positives, true negatives, false positives, and false negatives.
   
2. **ROC-AUC Curve**:
   - A curve plotting the true positive rate (TPR) vs false positive rate (FPR).
   - AUC value close to 1 indicates a better-performing model.
   
3. **Loss Curves**:
   - Graphs showing the training and validation loss over the epochs.

---

### Notes:
- The target accuracy for this project is 93%.
- Hyperparameter tuning can be performed using Keras Tuner to optimize the model's number of layers, neurons, and learning rate.
- Early stopping ensures that the model does not overfit, and learning rate reduction helps the model converge smoothly.

--- 
