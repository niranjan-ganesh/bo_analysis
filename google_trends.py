from pytrends.request import TrendReq
import pandas as pd

# Set up pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# Read the list of movie names from the text file
with open('movie_list_for_trends.txt', 'r') as file:
    movie_names = [line.strip() for line in file]

# Initialize a dictionary to store Google Trends data
trends_data = {'Movie': [], 'Interest': []}

# Collect Google Trends historical data for each movie
for movie in movie_names:
    try:
        # Build the payload for the given movie with a historical timeframe and specific geo (India)
        pytrends.build_payload([movie], cat=0, timeframe='today 12-m', geo='IN', gprop='')

        # Get the interest over time
        interest_over_time_df = pytrends.interest_over_time()

        # Calculate the average interest
        average_interest = interest_over_time_df[movie].mean()

        # Store data in the dictionary
        trends_data['Movie'].append(movie)
        trends_data['Interest'].append(average_interest)

    except Exception as e:
        print(f"Error fetching Google Trends data for {movie}: {e}")


# Create a DataFrame from the Google Trends data
df_trends = pd.DataFrame(trends_data)

# Save the data to a CSV file
df_trends.to_csv('google_trends_data.csv', index=False)
