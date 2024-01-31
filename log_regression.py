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

# Extract views, sentiment, comments, and box office collection
x = data[:, :3].astype(float)  # Views, Sentiments, and Comments
y = data[:, 3].astype(float)    # Box Office Collection

# Apply logarithmic transformation to 'Views'
x[:, 0] = np.log(x[:, 0] + 1)  # Adding 1 to avoid log(0)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Print the equation
coefficients = model.coef_
intercept = model.intercept_

# Logarithmic transformation equation for 'Views'
log_equation = f'log(Views)'

# Overall equation
equation = f'BO Collection = {intercept:.2f} + {coefficients[0]:.2f} * {log_equation} + {coefficients[1]:.2f} * Sentiments + {coefficients[2]:.2f} * Comments'

print(f'Intercept: {intercept:.2f}')
print(f'Coefficients: {log_equation}={coefficients[0]:.2f}, Sentiments={coefficients[1]:.2f}, Comments={coefficients[2]:.2f}')
print(f'Equation: {equation}')

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the data
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.scatter(x_test[:, 0], y_test, label='Actual Data',s=10,alpha=.6)
plt.scatter(x_test[:, 0], y_pred, color='red', linewidth=3, label='Predicted Data',s=10,alpha=.5)
plt.xlabel('log(Views)')
plt.ylabel('Box Office Collection (in ₹ 10 billion)')
plt.title('Logarithmic Regression: log(Views) vs. Box Office Collection')

# Display the equation
equation_lines = equation.split(' + ')
equation_str = ', '.join(equation_lines)
plt.text(0.5, 0.9, equation_str, transform=plt.gca().transAxes, fontsize=10, ha='center', va='center')

plt.legend()
plt.show()
