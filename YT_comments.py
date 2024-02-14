import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
import csv
import numpy as np

# List of API keys
API_KEYS = []

# Create a YouTube API client

def search_video(youtube, query):
    try:
        # Search for the video using the query
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            type='video',
            maxResults=1
        ).execute()

        # Get the video details from the search results
        video_details = search_response['items'][0]['snippet']

        return video_details

    except HttpError as e:
        print(f'An HTTP error {e.resp.status} occurred: {e.content}')

def create_youtube_client(api_key):
    return build('youtube', 'v3', developerKey=api_key)

def get_video_statistics(youtube, video_id):
    try:
        # Get video statistics to obtain total comments
        stats_response = youtube.videos().list(
            part='statistics',
            id=video_id
        ).execute()

        # Extract statistics
        statistics = stats_response['items'][0]['statistics']

        # Check if 'commentCount' key exists in the dictionary
        if 'commentCount' in statistics:
            total_comments = int(statistics['commentCount'])
        else:
            total_comments = 0

        return total_comments

    except HttpError as e:
        print(f'An HTTP error {e.resp.status} occurred: {e.content}')

if __name__ == '__main__':
    # List of movie names and box office collection
    try:
        movie_data = np.genfromtxt("Bollywood_Movie_Database.txt", delimiter=',', dtype=str, skip_header=1)
    except ValueError as e:
        print(f"Error: {e}")

    start_index = 1

    # Process 100 titles at a time in a loop
    for i in range(start_index - 1, min(start_index - 1 + 1000, len(movie_data)), 100):
        batch_movies = movie_data[i:i + 100]

        # Use a different API key in each iteration
        api_key = API_KEYS[i // 100 % len(API_KEYS)]
        youtube = create_youtube_client(api_key)

        # Generate dynamic CSV file paths
        combined_csv_file_path = f'movie_comments_{(i // 100) + 1}.csv'

        with open(combined_csv_file_path, 'w', newline='', encoding='utf-8') as combined_csv_file:
            csv_writer = csv.writer(combined_csv_file)
            csv_writer.writerow(['Movie', 'Total Comments'])

            for movie_entry in batch_movies:
                movie_title = movie_entry[2]

                # Search for the movie trailer on YouTube
                video_details = search_video(youtube, f'{movie_title} trailer')

                if video_details:
                    # Extract relevant information
                    video_id = video_details['thumbnails']['high']['url'].split('/')[-2]  # Extract video ID from thumbnail URL

                    # Get total comments for the video
                    total_comments = get_video_statistics(youtube, video_id)

                    # Write data to CSV
                    csv_writer.writerow([movie_title, total_comments])

                    print(f'Movie: {movie_title}')
                    print(f'Total Comments: {total_comments}\n')

                else:
                    print(f'No trailer found for {movie_title}')

        print(f'Data written to {combined_csv_file_path}')
