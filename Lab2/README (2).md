
# Multi-class Classification of Iris Dataset using PyTorch

## Objective
The goal of this project is to build a multi-class classification model using PyTorch to classify Iris species based on various features from the Iris dataset. The performance of the model is evaluated using accuracy, confusion matrix, precision, recall, F1-score, and ROC-AUC curve. Additionally, the model's performance is visualized using loss curves over epochs and ROC-AUC plots for each class.

## Dataset Description
The Iris dataset is a classic dataset containing 150 samples of iris flowers. Each sample has 4 features:
- **Sepal length**
- **Sepal width**
- **Petal length**
- **Petal width**

The target variable is the species of the iris flower, which is classified into three categories:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Steps to Run the Code

### 1. Installation of Dependencies

Before running the code, make sure you have the necessary dependencies installed. You can install them using `pip`.

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

Alternatively, use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
torch==2.0.0
pandas==1.3.3
numpy==1.21.2
scikit-learn==1.0.1
matplotlib==3.4.3
```

### 2. Prepare the Dataset

- Load the Iris dataset (`iris.csv`) into a Pandas DataFrame.
- Preprocess the dataset:
  - Split the data into features `X` and target `y`.
  - Normalize the feature values using `StandardScaler`.
  - Split the dataset into training and testing sets (80% training, 20% testing).

### 3. Build and Train the Neural Network Model

- Define a simple feed-forward neural network (Multi-Layer Perceptron) using PyTorch.
- Use hidden layers with ReLU activation.
- Use `CrossEntropyLoss` as the loss function and `Adam` as the optimizer.
- Train the model over multiple epochs, tracking training and validation loss.

### 4. Evaluate the Model

- Evaluate the trained model on the test set using metrics like accuracy, confusion matrix, precision, recall, and F1-score.
- Plot the ROC-AUC curve to measure the model's performance for each class.
- Visualize the loss curves over the epochs for training and validation.

### 5. Visualize Results

#### Training and Validation Loss
- The training and validation loss curves help you assess whether the model is underfitting or overfitting.
  
![Training Loss Plot](![image](https://github.com/user-attachments/assets/9eb48854-6174-446d-a552-ec6b5582c085)
)

#### ROC-AUC Curves
- ROC-AUC curves are plotted for each class, which helps visualize how well the model distinguishes between classes.

![ROC-AUC Plot](![image](https://github.com/user-attachments/assets/db5eeaf0-e27d-4e73-8cb1-ede8c4598782)
)

### 6. Run the Code in Jupyter Notebook

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Open `iris_classification.ipynb` and run all the cells to train, evaluate, and visualize the model.

## File Structure

```
├── iris_classification.ipynb   # Jupyter Notebook for building and training the model
├── iris.csv                    # Iris dataset
├── requirements.txt            # Dependencies required for the project
├── plots/                      # Contains the generated plots (loss curves, ROC-AUC curves)
└── README.md                   # Project description and instructions
```

## Outputs

1. **Accuracy**:
   - The overall accuracy of the model on the test set is **100%**.
   
2. **Confusion Matrix**:
   ```
   [[10  0  0]
    [ 0  9  0]
    [ 0  0 11]]
   ```
   - The confusion matrix indicates perfect classification for all classes.
   
3. **Classification Report**:
   ```
                    precision    recall  f1-score   support

        Iris-setosa       1.00      1.00      1.00        10
    Iris-versicolor       1.00      1.00      1.00         9
     Iris-virginica       1.00      1.00      1.00        11

           accuracy                           1.00        30
          macro avg       1.00      1.00      1.00        30
       weighted avg       1.00      1.00      1.00        30
   ```

4. **ROC-AUC Curve**:
   - ROC curves are plotted for each class (Iris-setosa, Iris-versicolor, Iris-virginica).
   - The AUC (Area Under the Curve) for all classes is **1.00**, indicating perfect classification.

5. **Loss Curves**:
   - Graphs showing the training and validation loss over each epoch.

## Model Evaluation

The model's performance is evaluated using the following metrics:
- **Accuracy**: The percentage of correctly classified samples.
- **Confusion Matrix**: Displays the true positives, true negatives, false positives, and false negatives for each class.
- **Precision, Recall, and F1-Score**: Evaluated per class.
- **ROC-AUC Curve**: Provides a visualization of the model's performance per class.

---

### Notes:
- The target accuracy for this project is 95% or higher, and the current model achieved 100%.
- Hyperparameter tuning (e.g., the number of layers, neurons, learning rate) can be performed to optimize the model's performance.

---

### Additional Resources:
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Iris Dataset Information](https://archive.ics.uci.edu/ml/datasets/iris)

---
