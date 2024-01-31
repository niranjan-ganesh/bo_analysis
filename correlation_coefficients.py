import csv
import numpy as np

# Load data from the source file
try:
    data = np.genfromtxt("final_data_regression.csv", delimiter=',', dtype=str, skip_header=1)
except ValueError as e:
    print(f"Error: {e}")

# Extract views, sentiments, comments, and target (box office collection)
x = data[:, :3].astype(float)  # Views, Sentiments, and Comments
y = data[:, 3].astype(float)    # Box Office Collection

# Calculate correlation coefficients
correlation_views_bo = np.corrcoef(x[:, 0], y)[0, 1]
correlation_sentiment_bo = np.corrcoef(x[:, 1], y)[0, 1]
correlation_comments_bo = np.corrcoef(x[:, 2], y)[0, 1]
correlation_views_sentiment = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
correlation_views_comments = np.corrcoef(x[:, 0], x[:, 2])[0, 1]
correlation_sentiment_comments = np.corrcoef(x[:, 1], x[:, 2])[0, 1]

# Print correlation coefficients
print(f'Correlation between Views and Box Office Collection: {correlation_views_bo:.2f}')
print(f'Correlation between Sentiment and Box Office Collection: {correlation_sentiment_bo:.2f}')
print(f'Correlation between Comments and Box Office Collection: {correlation_comments_bo:.2f}')
print(f'Correlation between Views and Sentiment: {correlation_views_sentiment:.2f}')
print(f'Correlation between Views and Comments: {correlation_views_comments:.2f}')
print(f'Correlation between Sentiment and Comments: {correlation_sentiment_comments:.2f}')
