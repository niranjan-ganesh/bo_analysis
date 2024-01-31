import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from the CSV file
try:
    data = np.genfromtxt("final_data_regression.csv", delimiter=',', dtype=str, skip_header=1)
except ValueError as e:
    print(f"Error: {e}")

# Extract features (views and sentiment) and target (box office collection)
X = data[:, :2].astype(float)  # Views and Sentiment
y = data[:, 2].astype(float)    # Box Office Collection

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Polynomial Regression
degree = 2  # You can adjust the degree based on your data
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Create a polynomial regression model
model = LinearRegression()

# Train the model
model.fit(X_train_poly, y_train)

# Print the equation
coefficients = model.coef_
intercept = model.intercept_

print(f'BO Collection = {intercept:.2f} + {coefficients[1]:.2f} * Views + {coefficients[2]:.2f} * Sentiment + {coefficients[3]:.2f} * Views^2 + {coefficients[4]:.2f} * Views * Sentiment + {coefficients[5]:.2f} * Sentiment^2')
equation = np.array(f'BO Collection (₹) = {intercept:.2f} + {coefficients[1]:.2f} * Views + {coefficients[2]:.2f} * Sentiment + {coefficients[3]:.2f} * Views^2 + {coefficients[4]:.2f} * Views * Sentiment + {coefficients[5]:.2f} * Sentiment^2')

# Make predictions on the test set
y_pred = model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the predicted vs actual values and regression curve
plt.scatter(X_test[:, 0], y_test, label='Actual Data')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted Data')
plt.xlabel('Views (in millions)')
plt.ylabel('Box Office Collection (in ₹ 10 billion)')
plt.title('Polynomial Regression: Actual vs Predicted Box Office Collection')
plt.text(0.1, 0.6, equation, transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom')

plt.legend()
plt.show()
