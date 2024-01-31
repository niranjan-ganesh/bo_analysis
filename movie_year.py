import csv
from imdb import IMDb

# Initialize IMDb object
ia = IMDb()

# Load movie data from the CSV file
movies_data = []
with open("Bollywood_Movie_Database.csv", 'r', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        movies_data.append(row)

# Skip header if present
header = movies_data[0]
movies_data = movies_data[1:]

# Search IMDb for each movie and print the release year
for movie_info in movies_data:
    movie_name = movie_info[0]
    try:
        # Search for the movie
        search_results = ia.search_movie(movie_name)


        if search_results:
            first_result = search_results[0]

            ia.update(first_result)

            # Print the release year
            release_year = first_result.data.get('year', 'N/A')
            print(f"Movie: {movie_name}, Release Year: {release_year}")
        else:
            print(f"Movie: {movie_name}, Release Year: Not Found")
    except Exception as e:
        print(f"Error searching for {movie_name}: {e}")

with open('movies_data_updated.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header
    csv_writer.writerow(header)

    # Write the updated data
    csv_writer.writerows(movies_data)

print("Release years added to the CSV file.")