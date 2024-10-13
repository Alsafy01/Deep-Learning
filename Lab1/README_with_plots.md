
# Housing Price Prediction Task

## Objective:
The goal of this task is to perform linear regression using an Artificial Neural Network (ANN) implemented in **PyTorch** to predict house prices based on a set of features (such as area, number of bedrooms, bathrooms, etc.). The ANN is trained and evaluated on a dataset, and the final performance is measured using **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R-squared (R²)** metrics. Visualizations comparing the predicted vs. actual prices are also generated.

---

## Dataset Description:
The dataset used for this task contains information about housing properties. Each record represents a house, with the following columns:

1. **price**: The price of the house (Target variable).
2. **area**: The size of the house in square feet.
3. **bedrooms**: Number of bedrooms in the house.
4. **bathrooms**: Number of bathrooms in the house.
5. **stories**: Number of floors in the house.
6. **mainroad**: Whether the house is adjacent to the main road (categorical).
7. **guestroom**: Whether the house has a guest room (categorical).
8. **basement**: Whether the house has a basement (categorical).
9. **hotwaterheating**: Whether the house has hot water heating (categorical).
10. **airconditioning**: Whether the house has air conditioning (categorical).
11. **parking**: Number of parking spaces.
12. **prefarea**: Whether the house is in a preferred area (categorical).
13. **furnishingstatus**: The furnishing status of the house (furnished, semi-furnished, unfurnished) (categorical).

The objective is to predict the **price** of the house based on these features.

---

## Steps to Run the Code in Google Colab:

1. **Set Up Your Environment**:
   - Open [Google Colab](https://colab.research.google.com/) in your browser.
   - Create a new notebook.

2. **Upload Dataset**:
   - Upload the dataset file (`Housing-1.csv`) to the Colab environment by clicking on the file icon on the left panel and then clicking on the upload icon.

3. **Install Dependencies**:
   - Google Colab already comes with most dependencies (like `PyTorch` and `sklearn`) pre-installed. However, to ensure everything is installed, run the following code to install any missing dependencies:
   
   ```bash
   !pip install torch matplotlib pandas scikit-learn
   ```

4. **Load and Preprocess the Data**:
   - Upload the dataset to Colab and ensure you load it correctly using `pandas`. The preprocessing steps include encoding categorical variables and scaling the features using `StandardScaler`.

5. **Train the Model**:
   - Use the provided code to define the neural network using **PyTorch**. Set up the optimizer, loss function, and data loader.
   - Train the model using the `train_model()` function for 100 epochs.

6. **Evaluate the Model**:
   - After training, evaluate the model using the provided `mean_squared_error`, `mean_absolute_error`, and `r2_score` metrics from `scikit-learn`.
   - Use the trained model to make predictions and compare them with the actual house prices.

7. **Visualizations**:
   - Plot the **Predicted vs. Actual Prices** using `matplotlib` to visualize how well the model performed.
   - Plot the **Training Loss Curve** to see how the model's performance improves over time.

8. **Running the Full Code**:
   - You can copy and paste the provided full code into a single Colab notebook cell and run it sequentially. Ensure that the dataset is uploaded and properly referenced.

---

## Visualizations

### 1. Predicted vs. Actual Prices:

![Predicted vs. Actual Prices](https://i.imgur.com/4zcgXfY.png)

### 2. Training Loss Over Time:

![Training Loss Over Time](https://i.imgur.com/WWF8vKF.png)

---

## Dependencies and Installation Instructions:

To run the code, the following Python packages are required:

- **PyTorch**: The framework used for building and training the ANN.
- **pandas**: For loading and preprocessing the dataset.
- **scikit-learn**: For preprocessing and evaluating model performance (e.g., scaling, MSE, MAE, and R² metrics).
- **matplotlib**: For visualizing the predicted vs. actual prices and the training loss curve.

In Colab, these can be installed using:

```bash
!pip install torch matplotlib pandas scikit-learn
```

If you are using a local machine or another environment, ensure you install these packages by running the above command in your terminal or using `conda`:

```bash
conda install pytorch torchvision torchaudio -c pytorch
conda install pandas matplotlib scikit-learn
```

---

## Conclusion:
By following the steps outlined in this document, you can train an Artificial Neural Network model to predict house prices, evaluate its performance using common regression metrics, and visualize the results using Python libraries in Colab.
