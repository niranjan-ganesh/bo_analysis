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
x = data[:, :3]  # Views, Sentiments, and Comments
y = data[:, 3]    # Box Office Collection

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a logarithmic regression model for Views
model_views = LinearRegression()
model_views.fit(np.log1p(x_train[:, 0].reshape(-1, 1)), y_train)

# Create a logarithmic regression model for Sentiments
model_sentiments = LinearRegression()
model_sentiments.fit(x_train[:, 1].reshape(-1, 1), y_train)

# Create a logarithmic regression model for Comments
model_comments = LinearRegression()
model_comments.fit(np.log1p(x_train[:, 2].reshape(-1, 1)), y_train)

# Evaluate the models and generate equations
models = [model_views, model_sentiments, model_comments]
variable_names = ['Views', 'Sentiments', 'Comments']

for model, variable_name in zip(models, variable_names):
    # Make predictions on the test set
    y_pred = model.predict(np.log1p(x_test[:, models.index(model)].reshape(-1, 1)))

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print results
    print(f'\nResults for {variable_name}:')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Print equation
    coefficients = model.coef_
    intercept = model.intercept_
    equation = f'BO Collection = {intercept:.2f} + {coefficients[0]:.2f} * log({variable_name})'
    print(f'Equation: {equation}')

    # Visualize the data
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test[:, models.index(model)], y_test, label='Actual Data',s=10,alpha=.5)
    plt.scatter(x_test[:, models.index(model)], y_pred, color='red', label='Predicted Data',alpha=.5)
    plt.xlabel(f'{variable_name}')
    plt.ylabel('Box Office Collection')
    plt.title(f'Logarithmic Regression: {variable_name} vs. Box Office Collection')
    results = f'MSE: {mse:.2f}\nR-squared: {r2:.2f}\n{equation}'
    plt.text(0.05, 0.9, results, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.legend()
plt.show()
