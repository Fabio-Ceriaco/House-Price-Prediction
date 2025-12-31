# XG Boost Model
# The XG Boost model is an efficient and scalable implementation of gradient boosting framework by Tianqi Chen,
# designed for speed and performance. It is widely used for regression, classification, and ranking tasks.
# It's Like a decision tree-based ensemble Machine Learning algorithm
# In this script, we will implement an XG Boost model to predict house prices based on various features.
# This is a Supervised learning, to solve a regression problem where the target variable is continuous.

# This dataset contains information about housing prices in Boston and is often used for regression analysis and predictive modeling.
# The dataset is based on the classic Boston Housing dataset.
#
# Attributes:

# 1. CRIM (Per Capita Crime Rate): The per capita crime rate in the neighborhood.
# 2. ZN (Proportion of Residential Land Zoned for Large Lots): The proportion of residential land zoned for lots over 25,000 sq. ft.
# 3. INDUS (Proportion of Non-Retail Business Acres): The proportion of non-retail business acres per town.
# 4. CHAS (Charles River Dummy Variable): A binary variable indicating whether the Charles River bounds the tract (1 if bounded, 0 otherwise).
# 5. NOX (Nitric Oxides Concentration): Nitric oxides concentration (parts per 10 million).
# 6. RM (Average Number of Rooms per Dwelling): The average number of rooms per dwelling.
# 7. AGE (Proportion of Owner-Occupied Units Built Prior to 1940): The proportion of owner-occupied units built prior to 1940.
# 8. DIS (Weighted Distances to Employment Centers): Weighted distances to five Boston employment centers.
# 9. RAD (Index of Accessibility to Radial Highways): An index representing accessibility to radial highways.
# 10. TAX (Full-Value Property Tax Rate per $10,000): The full-value property tax rate per $10,000.
# 11. PTRATIO (Pupil-Teacher Ratio): The pupil-teacher ratio by town.
# 12. B (1000(Bk - 0.63)^2 where Bk is the Proportion of Black Residents): A measure of the proportion of Black residents adjusted for an offset.
# 13. LSTAT (Percentage of Lower Status of the Population): The percentage of lower-status residents in the population.
# 14. MEDV (Median Value of Owner-Occupied Homes): The median value of owner-occupied homes in $1000s (Target Variable)


# WORK FLOW
# 1. Import Libraries
# 2. Load Dataset
# 3. Data Preprocessing
# 4. Data Analysis
# 5. Split Dataset into Training and Testing Sets
# 6. Train the XG Boost Model
# 7. Make Predictions
# 8. Evaluate the Model
# 9. Visualize Results

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import (
    XGBRegressor,
)  # Import the XGBoost Regressor to build the model this is a regression problem


# 2. Load Dataset
# Load the Boston Housing dataset from a CSV file

data_df = pd.read_csv("boston.csv")

# 3. Data Preprocessing

# Check for missing values and handle them if necessary

print(data_df.head())  # Print the first 5 rows of the dataset

print(data_df.shape)  # Print the shape of the dataset

print(data_df.info())  # Print the info of the dataset

print(data_df.isnull().sum())  # Check for missing values in the dataset

print(data_df.describe())  # Print the statistical summary of the dataset


# 4. Data Analysis

# Print the correlation matrix of the dataset
# values near 1 indicates strong positive correlation
# values near -1 indicates strong negative correlation
# values near 0 indicates no correlation
# Positive correlation means that as one variable increases, the other variable also increases.
# Negative correlation means that as one variable increases, the other variable decreases.

print(data_df.corr())

# Visualize the correlation matrix using a heatmap
# annot=True to display the correlation values on the heatmap
# cmap='coolwarm' to use a color scheme that ranges from cool colors (blue) to warm colors (red)

plt.figure(figsize=(10, 8))
sns.heatmap(data_df.corr(), annot=True, cmap="coolwarm")
plt.show()


# Split Dataset into Features and Target Variable
X = data_df.drop("MEDV", axis=1)  # Features axis=1 indicates we are dropping a column
y = data_df["MEDV"]  # Target Variable

print(X)  # Print the features
print(y)  # Print the target variable

# 5. Split Dataset into Training and Testing Sets
# Test size is 20% of the total dataset
# random_state is set to 2 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

print("X shape:", X.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y shape:", y.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 6. Train the XG Boost Model

xgb_model = XGBRegressor()

xgb_model.fit(
    X_train, y_train
)  # We need to fit the model to the training data and the target variable

# 7. Make Predictions

# Prediction on Training Data

train_data_prediction = xgb_model.predict(
    X_train
)  # Predict the target variable for the training data

print("Training Data Prediction:", train_data_prediction)

# Prediction on Testing Data
test_data_prediction = xgb_model.predict(
    X_test
)  # Predict the target variable for the testing data

print("Testing Data Prediction:", test_data_prediction)


# 8. Evaluate the Model
# we can not use accuracy score for regression problems, the accuracy score is used for classification problems
# accuracy score will check or count the number of correct predictions made by the model
# and will subtract it from the total number of predictions made by the model
# to find the difference between the actual and predicted values
# In regression we can use accuracy because the target variable is continuous and is only numerical values.

# In this regression model we will use the following evaluation metrics:
# Mean Absolute Error (MAE) - Mean Absolute Percentage Error (MAPE) represents the average absolute percentage difference between the predicted and actual values.
# Mean Squared Error (MSE) - Mean Squared Error (MSE) represents the average squared difference between the predicted and actual values.
# Root Mean Squared Error (RMSE) - Root Mean Squared Error (RMSE) represents the square root of the average squared difference between the predicted and actual values.
# R squared Error (R2) - R squared Error (R2) represents the proportion of variance in the dependent variable that is predictable from the independent variables.
# This are commonly used metrics to evaluate the performance of regression models.
# Because they provide insights into the magnitude of errors made by the model in its predictions.

# Evaluate on Training Data

# R squared Error

r2_train = metrics.r2_score(
    y_train, train_data_prediction
)  # Calculate R squared error for training data using the actual and predicted values
print("R squared Error on Training Data:", r2_train)
# R2 Error 0.9999980039471451
# Conclusion: The model is performing exceptionally well on the training data, explaining nearly all the variance in house prices.
# This indicates that the model has learned the training data very effectively.
# The metrics of this test are:
# R squared value of 0.9999980039471451, indicating an almost perfect fit to the training data, in percentage terms, the model explains approximately 99.9998% of the variance in house prices.

# Mean Absolute Error
mea_train = metrics.mean_absolute_error(y_train, train_data_prediction)
print("Mean Absolute Error on Training Data:", mea_train)
# MEA 0.0091330346494618
# Conclusion: On average, the model's predictions for house prices deviate from the actual prices by approximately $9.13, indicating high accuracy.
# This low error suggests that the model is very precise in estimating house prices.
# The metrics of this test are:
# Mean Absolute Error (MEA) of 0.0091330346494618, indicating that, on average, the model's predictions deviate from the actual house prices by approximately $9.13.
# 0.0091330346494618 * 1000 = 9.13  - 1000 is because the target variable is in $1000s.

# Mean Squared Error
mse_train = metrics.mean_squared_error(y_train, train_data_prediction)
print("Mean Squared Error on Training Data:", mse_train)
# MSE 0.00016880599071692853
# Conclusion: The model's predictions for house prices have a very low average squared deviation from the actual prices, indicating high accuracy.
# This low error suggests that the model is very precise in estimating house prices.
# The metrics of this test are:
# Mean Squared Error (MSE) of 0.00016880599071692853, indicating that the average squared deviation between the model's predictions and the actual house prices is very low.

# Root Mean Squared Error
rmse_train = np.sqrt(mse_train)  # Calculate RMSE by taking the square root of MSE
print("Root Mean Squared Error on Training Data:", rmse_train)
# RMSE 0.012992535961733127
# Conclusion: The model's predictions for house prices have a very low average deviation from the actual prices, indicating high accuracy.
# This low error suggests that the model is very precise in estimating house prices.
# The metrics of this test are:
# Root Mean Squared Error (RMSE) of 0.012992535961733127, indicating that the average deviation between the model's predictions and the actual house prices is very low.

# Evaluate on Testing Data

# R squared Error
r2_test = metrics.r2_score(y_test, test_data_prediction)
print("R squared Error on Testing Data:", r2_test)
# R2 Error 0.9051721149855378
# Conclusion: The model performs well on unseen data, explaining a significant portion of the variance in house prices.
# However, there is a noticeable drop in performance compared to the training data, suggesting potential overfitting.
# The metrics of this test are:
# R squared value of 0.9051721149855378, indicating that the model explains approximately 90.52% of the variance in house prices on unseen data.

# Mean Absolute Error
mea_test = metrics.mean_absolute_error(y_test, test_data_prediction)
print("Mean Absolute Error on Testing Data:", mea_test)
# MEA 2.0748727686264927
# Conclusion: On average, the model's predictions for house prices deviate from the actual prices by approximately $2074.87.
# This indicates that while the model is reasonably accurate, there is still room for improvement in its predictions.
# The metrics of this test are:
# Mean Absolute Error (MEA) of 2.0748727686264927, indicating that, on average, the model's predictions deviate from the actual house prices by approximately $2074.87.
# 2.0748727686264927 * 1000 = 2074.87  - 1000 is because the target variable is in $1000s.

# Mean Squared Error
mse_test = metrics.mean_squared_error(y_test, test_data_prediction)
print("Mean Squared Error on Testing Data:", mse_test)
# MSE 7.9332706911154185
# Conclusion: The model's predictions for house prices have a moderate average squared deviation from the actual prices.
# This indicates that while the model is reasonably accurate, there is still room for improvement in its predictions.
# The metrics of this test are:
# Mean Squared Error (MSE) of 7.9332706911154185, indicating that the average squared deviation between the model's predictions and the actual house prices is moderate.

# Root Mean Squared Error
rmse_test = np.sqrt(mse_test)  # Calculate RMSE by taking the square root of MSE
print("Root Mean Squared Error on Testing Data:", rmse_test)
# RMSE 2.8166062364333815
# Conclusion: The model's predictions for house prices have a moderate average deviation from the actual prices.
# This indicates that while the model is reasonably accurate, there is still room for improvement in its predictions.
# The metrics of this test are:
# Root Mean Squared Error (RMSE) of 2.8166062364333815, indicating that the average deviation between the model's predictions and the actual house prices is moderate.

# 9. Visualize Results

# Plotting the Actual vs Predicted values for the training data
plt.scatter(y_train, train_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices (Training Data)")
plt.show()

# The plot show a linear relationship along the diagonal line y=x, indicating that the predicted prices closely match the actual prices.
# This suggests that the model is performing well on the training data.

# Plotting the Actual vs Predicted values for the testing data
plt.scatter(y_test, test_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices (Testing Data)")
plt.show()

# In this plot, we can see that the predicted prices generally follow the trend of the actual prices, but there is more scatter compared to the training data plot.
# This indicates that while the model performs well on unseen data, there are some discrepancies between the predicted and actual prices.
