import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
import csv
import numpy as np

# List of API keys
API_KEYS = ['AIzaSyC3M9Y4QjqAU3-xb9oNXFBdFEsTsL0je68', 'AIzaSyBFPXToLku3Lbk_xpc2qUtqrHxKVwU2tW4','AIzaSyBrWNrZg6jxQU1TO0jEZ3ujuYY6zroSkgw','AIzaSyAJ9LAVIehBs0Hlbf79gm_OMHADGINIH0w','AIzaSyDd-LHcZ4WpGsb6qOM827cZnuBJA4syEuQ','AIzaSyBSUxrKe0JNdYbo2kr8epjiIEPNxcetpt8','AIzaSyC-UKhOfeNgzw3RXkHKAUcGzsoCIWm97xI','AIzaSyAQv4nh7OoaNcBXu-wBwuzonpbgqUpaUjM','AIzaSyDv30cGThqF4vW8v5UcFs0RuFKDwBdWW0M','AIzaSyBf4P5_wtcbQtt755E9oVhCTn4-Tdk3Hm0','AIzaSyDo58rvwLHU5hKGjWiXxwrJpz41DAOx5DE']
# API_KEYS = ['AIzaSyDfHaSNlmAeRySuzASf9gjyYjTeMswt3mg']
# Create YouTube API client
def create_youtube_client(api_key):
    return build('youtube', 'v3', developerKey=api_key)

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

def get_video_statistics(youtube, video_id):
    try:
        # Get video statistics to obtain total views
        stats_response = youtube.videos().list(
            part='statistics',
            id=video_id
        ).execute()

        # Extract statistics
        statistics = stats_response['items'][0]['statistics']

        # Get total views
        total_views = int(statistics['viewCount'])

        return total_views

    except HttpError as e:
        print(f'An HTTP error {e.resp.status} occurred: {e.content}')

def get_video_comments(youtube, video_id):
    comments = []

    try:
        # Get video comments
        comments_response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText'
        ).execute()

        # Extract comments
        for item in comments_response['items']:
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment_text)

        return comments

    except HttpError as e:
        print(f'An HTTP error {e.resp.status} occurred: {e.content}')
        return []

def perform_sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

if __name__ == '__main__':
    # List of movie names and box office collection
    try:
        movie_data = np.genfromtxt("Bollywood_Movie_Database.txt", delimiter=',', dtype=str, skip_header=1)
    except ValueError as e:
        print(f"Error: {e}")

    start_index = 100

    # Process 100 titles at a time in a loop
    for i in range(start_index - 1, min(start_index - 1 + 100, len(movie_data)), 100):
        batch_movies = movie_data[i:i + 100]

        # Use a different API key in each iteration
        api_key = API_KEYS[i // 100 % len(API_KEYS)]
        youtube = create_youtube_client(api_key)

        # Generate dynamic CSV file paths
        combined_csv_file_path = f'movie_data_100_200_{(i // 100) + 1}.csv'

        with open(combined_csv_file_path, 'w', newline='', encoding='utf-8') as combined_csv_file:
            csv_writer = csv.writer(combined_csv_file)
            csv_writer.writerow(['Movie', 'Views', 'Sentiment', 'Total BO Collection'])

            for movie_entry in batch_movies:
                movie_title = movie_entry[2]
                box_office_collection = movie_entry[3]

                # Search for the movie trailer on YouTube
                video_details = search_video(youtube, f'{movie_title} trailer')

                if video_details:
                    # Extract relevant information
                    video_title = video_details['title']
                    video_id = video_details['thumbnails']['high']['url'].split('/')[-2]  # Extract video ID from thumbnail URL

                    # Get total views for the video
                    total_views = get_video_statistics(youtube, video_id)

                    # Get and analyze video comments
                    comments = get_video_comments(youtube, video_id)
                    sentiment_scores = [perform_sentiment_analysis(comment) for comment in comments]
                    average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

                    # Write data to CSV
                    csv_writer.writerow([video_title, total_views, average_sentiment, box_office_collection])

                    print(f'Title: {video_title}')
                    print(f'Views: {total_views}')
                    print(f'Sentiment: {average_sentiment}')
                    print(f'Total BO Collection: {box_office_collection}\n')

                else:
                    print(f'No trailer found for {movie_title}')

        print(f'Data written to {combined_csv_file_path}')
