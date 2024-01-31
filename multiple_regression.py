import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from the source file
try:
    data = np.genfromtxt("final_data_regression.csv", delimiter=',', dtype=float, skip_header=1)
except ValueError as e:
    print(f"Error: {e}")

# Extract views, sentiments, comments, and target (box office collection)
X = data[:, :3]  # Views, Sentiments, and Comments
y = data[:, 3]    # Box Office Collection

plt.scatter(np.log1p(data[:,1]),(data[:,3]))
plt.show()
# # Apply logarithmic transformation to the "Views" column
# X[:, 0] = np.log1p(X[:, 0])
# X[:, 1] = np.log1p(X[:, 1])
# X[:, 2] = np.log1p(X[:, 2])
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Create a multiple linear regression model
# model = LinearRegression()
#
# # Train the model
# model.fit(X_train, y_train)
#
# # Print the coefficients
# coefficients = model.coef_
# intercept = model.intercept_
#
# equation = f'BO Collection = {intercept:.2f} + {coefficients[0]:.2f} * log(Views) + {coefficients[1]:.2f} * Sentiments + {coefficients[2]:.2f} * Comments'
#
# print(f'Intercept: {intercept:.2f}')
# print(f'Coefficients: log(Views)={coefficients[0]:.2f}, Sentiments={coefficients[1]:.2f}, Comments={coefficients[2]:.2f}')
# print(f'Equation: {equation}')
#
# # Make predictions on the test set
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')
#
# # Visualize the data
# fig = plt.figure(figsize=(12, 6))
#
# # Visualize the data for Views
# # plt.figure(figsize=(8, 6))
# plt.scatter(X_test[:, 0], y_test, label='Actual Data')
# plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted Data')
# plt.xlabel('log(Views)')
# plt.ylabel('Box Office Collection')
# plt.title('log(Views) vs. Box Office Collection')
# plt.legend()
#
#
# # Visualize the data for Sentiments
# plt.figure(figsize=(8, 6))
# plt.scatter(X_test[:, 1], y_test, label='Actual Data')
# plt.scatter(X_test[:, 1], y_pred, color='red', label='Predicted Data')
# plt.xlabel('Sentiments')
# plt.ylabel('Box Office Collection')
# plt.title('Sentiments vs. Box Office Collection')
# plt.legend()
#
#
# # Visualize the data for Comments
# plt.figure(figsize=(8, 6))
# plt.scatter(X_test[:, 2], y_test, label='Actual Data')
# plt.scatter(X_test[:, 2], y_pred, color='red', label='Predicted Data')
# plt.xlabel('Comments')
# plt.ylabel('Box Office Collection')
# plt.title('Comments vs. Box Office Collection')
# plt.legend()
#
# plt.show()

