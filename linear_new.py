import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from the source file
try:
    data = np.genfromtxt("final_data_regression.csv", delimiter=',', dtype=str, skip_header=1)
except ValueError as e:
    print(f"Error: {e}")

# Extract views, sentiment, comments, and target (box office collection)
x = data[:, :3].astype(float)  # Views, Sentiments, and Total Comments
y = data[:, 3].astype(float)   # Box Office Collection

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Print the equation
coefficients = model.coef_
intercept = model.intercept_

print(f'Equation: BO Collection = {intercept:.2f} + {coefficients[0]:.2f} * Views + {coefficients[1]:.2f} * Sentiments + {coefficients[2]:.2f} * Total Comments')
equation = np.array([f'BO Collection (₹) = {intercept:.2f} + {coefficients[0]:.2f} * Views + {coefficients[1]:.2f} * Sentiments + {coefficients[2]:.2f} * Total Comments'])

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the data
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.scatter(x_test[:, 0], y_test, label='Actual Data')
plt.plot(x_test[:, 0], y_pred, color='red', linewidth=3, label='Regression Line')
plt.xlabel('Views (in millions)')
plt.ylabel('Box Office Collection (in ₹ 10 billion)')
plt.title('Linear Regression: Views vs. Box Office Collection')
plt.text(0.25, 0.4, *equation, transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom')
plt.legend()
plt.show()
