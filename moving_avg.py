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

# Extract views, comments, and target (box office collection)
x = data[:, [0, 2]].astype(float)  # Views and Comments
y = data[:, 3].astype(float)        # Box Office Collection

# Calculate the moving average for 'Views' and 'Comments'
window_size = 3  # You can adjust this parameter
padded_views = np.pad(x[:, 0], (window_size // 2, window_size // 2), mode='edge')
x[:, 0] = np.convolve(padded_views, np.ones(window_size)/window_size, mode='valid')

padded_comments = np.pad(x[:, 1], (window_size // 2, window_size // 2), mode='edge')
x[:, 1] = np.convolve(padded_comments, np.ones(window_size)/window_size, mode='valid')

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the data
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.scatter(x_test[:, 0], y_test, label='Actual Data', s=10, alpha=.6)
plt.scatter(x_test[:, 0], y_pred, color='red', linewidth=3, label='Predicted Data', s=10, alpha=.5)
plt.xlabel('Moving Average of Views')
plt.ylabel('Box Office Collection (in ₹ 10 billion)')
plt.title(f'Linear Regression with Moving Average: Views vs. Box Office Collection (Window Size={window_size})')

# Display equation in one line
coefficients = model.coef_
intercept = model.intercept_
equation = f'BO Collection = {intercept:.2f} + {coefficients[0]:.2f} * Moving Average(Views) + {coefficients[1]:.2f} * Moving Average(Comments)'
plt.text(0.5, 0.9, equation, transform=plt.gca().transAxes, fontsize=10, ha='center', va='center')

plt.legend()
plt.show()
