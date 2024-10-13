
# Orbital Position Prediction Using Neural Networks

## Objective:
This task aims to study the relationship between orbital positions and time by performing polynomial regression using a neural network implemented in **Keras**. We will utilize an orbital dataset containing time and corresponding position data to analyze and visualize the patterns.

The goal is to build a neural network model that predicts orbital positions based on time and evaluate its performance using **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R-squared (R²)** metrics. Visualizations comparing the predicted vs. actual positions and model loss over time are also included.

## Dataset Description:
The dataset `orbit.csv` contains information about the time and corresponding orbital positions of an object. The columns in the dataset are:

- **time_steps**: Time steps representing specific points in time.
- **y**: Orbital positions corresponding to each time step.

The dataset is used to study the relationship between time and position, and predict future positions based on the time steps.

## Steps to Run the Code in Google Colab:

### 1. Set Up Your Environment:
- Open [Google Colab](https://colab.research.google.com/) in your browser.
- Create a new notebook.

### 2. Upload Dataset:
- Upload the dataset file (`orbit.csv`) to the Colab environment by clicking on the file icon on the left panel and then clicking on the upload icon.

### 3. Install Dependencies:
- Colab comes with most libraries pre-installed, but to ensure everything is set up properly, install the necessary libraries with the following command:

```bash
!pip install tensorflow matplotlib pandas scikit-learn
```

### 4. Load and Preprocess Data:
- Load the dataset using `pandas`, check for any missing values, and remove them.
- Normalize the time steps using `MinMaxScaler` to scale the values between [-1, 1].
- Create polynomial features (i.e., add squared time steps to the input features).

### 5. Define and Compile the Model:
- Define a neural network with hidden layers using the **Keras** Sequential API.
- The model includes hidden layers with ReLU activations and an output layer to predict orbital positions.

### 6. Train the Model:
- Train the model using the **Adam** optimizer and monitor the performance with **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** metrics.
- Early stopping is used to prevent overfitting.

### 7. Evaluate the Model:
- After training, evaluate the model on the validation set and calculate the **MSE**, **MAE**, and **R²** scores.

### 8. Visualizations:
- Plot the predicted vs actual orbital positions over time.
- Plot the training and validation loss over the epochs to see the model's learning curve.

## Dependencies and Installation Instructions:

The following Python packages are required to run the code:

- **TensorFlow/Keras**: The deep learning framework used to build and train the neural network.
- **pandas**: For loading and processing the dataset.
- **scikit-learn**: For data preprocessing and evaluation metrics.
- **matplotlib**: For visualizing the predictions and loss curves.

To install the dependencies, run:

```bash
!pip install tensorflow matplotlib pandas scikit-learn
```

Alternatively, if running in a local environment, use:

```bash
pip install tensorflow matplotlib pandas scikit-learn
```

## Graphs/Plots:

- **Predicted vs Actual Positions**: A graph comparing the predicted positions by the model with the actual orbital positions over time.
- **Training and Validation Loss**: A plot showing how the model's loss decreased over the epochs during training.

## Deliverables:
1. A **.ipynb** file with fully commented code.
2. A **README.md** file including the task objective, dataset description, and instructions.
3. **Graphs/Plots** in PNG/JPEG format comparing actual vs predicted positions.
4. A GitHub repository containing the **.ipynb** file, **README.md**, and visualization plots.

