import argparse
import gzip
import json
import os
import re
import sqlite3
import sys
import urllib.request
from datetime import datetime

# Configuration
DB_PATH = "imdb_reviews.db"
IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
BASICS_FILE = "title.basics.tsv.gz"
RATINGS_FILE = "title.ratings.tsv.gz"

# Extract movie ID from URL
def extract_movie_id(url):
    match = re.search(r'/title/(tt\d+)/', url)
    if match:
        return match.group(1)
    return None

# Download a dataset
def download_dataset(url, filename, force=False):
    if os.path.exists(filename) and not force:
        print(f"{filename} already exists. Skipping download.")
        return True
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Download of {filename} complete.")
        return True
    except Exception as e:
        print(f"Error downloading file {filename}: {e}")
        return False

# Ensure all required columns exist in the database
def ensure_columns_exist():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check existing columns
    cursor.execute("PRAGMA table_info(reviews)")
    columns = [info[1] for info in cursor.fetchall()]
    
    # Add required columns if they don't exist
    if "movie_genre" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN movie_genre TEXT")
        print("Added movie_genre column to the database")
    
    if "movie_average_rating" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN movie_average_rating TEXT")
        print("Added movie_average_rating column to the database")
    
    if "rating_vote_count" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN rating_vote_count TEXT")
        print("Added rating_vote_count column to the database")
    
    conn.commit()
    conn.close()

# Process the basics dataset and update the database
def process_basics_dataset():
    print("Processing basics dataset for genres...")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all unique movie URLs with their associated IDs
    cursor.execute("SELECT DISTINCT movie_url FROM reviews WHERE movie_genre IS NULL")
    urls = [row[0] for row in cursor.fetchall()]
    
    if not urls:
        print("No URLs found that need genre updates.")
        conn.close()
        return
    
    print(f"Found {len(urls)} unique movie URLs that need genre updates")
    
    # Create a dictionary of IMDb IDs from URLs
    imdb_id_to_url = {}
    for url in urls:
        imdb_id = extract_movie_id(url)
        if imdb_id:
            imdb_id_to_url[imdb_id] = url
    
    print(f"Extracted {len(imdb_id_to_url)} valid IMDb IDs from URLs")
    
    # Process the gzipped file directly without full extraction
    updated_count = 0
    processed_lines = 0
    
    try:
        with gzip.open(BASICS_FILE, 'rt', encoding='utf-8') as f:
            # Skip header
            next(f)
            
            # Process each line
            for line in f:
                processed_lines += 1
                
                parts = line.strip().split('\t')
                if len(parts) < 9:  # Skip lines with insufficient columns
                    continue
                
                imdb_id = parts[0]
                
                # Only process if this ID is in our database
                if imdb_id in imdb_id_to_url:
                    genres = parts[8]
                    
                    # Handle '\N' (null value in IMDb dataset)
                    if genres != '\\N':
                        # Convert genres to list and then to JSON
                        genres_list = genres.split(',')
                        genres_json = json.dumps(genres_list)
                        
                        # Update the database
                        url = imdb_id_to_url[imdb_id]
                        cursor.execute(
                            "UPDATE reviews SET movie_genre = ? WHERE movie_url = ?",
                            (genres_json, url)
                        )
                        
                        updated_count += cursor.rowcount
                        
                        # Remove this ID as it's been processed
                        del imdb_id_to_url[imdb_id]
                
                # Commit every 1000 lines and show progress
                if processed_lines % 10000 == 0:
                    conn.commit()
                    print(f"Processed {processed_lines} lines, updated {updated_count} reviews...")
                
                # If we've found all our IDs, we can stop
                if not imdb_id_to_url:
                    break
        
        # Final commit
        conn.commit()
        
        print(f"Basics dataset processing complete. Processed {processed_lines} lines.")
        print(f"Updated genre information for {updated_count} reviews.")
        
        if imdb_id_to_url:
            print(f"Could not find genre information for {len(imdb_id_to_url)} IMDb IDs.")
        
    except Exception as e:
        print(f"Error processing basics dataset: {e}")
    
    finally:
        conn.close()

# Process the ratings dataset and update the database
def process_ratings_dataset():
    print("Processing ratings dataset...")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all unique movie URLs with their associated IDs that need rating updates
    cursor.execute("SELECT DISTINCT movie_url FROM reviews WHERE movie_average_rating IS NULL OR rating_vote_count IS NULL")
    urls = [row[0] for row in cursor.fetchall()]
    
    if not urls:
        print("No URLs found that need rating updates.")
        conn.close()
        return
    
    print(f"Found {len(urls)} unique movie URLs that need rating updates")
    
    # Create a dictionary of IMDb IDs from URLs
    imdb_id_to_url = {}
    for url in urls:
        imdb_id = extract_movie_id(url)
        if imdb_id:
            imdb_id_to_url[imdb_id] = url
    
    print(f"Extracted {len(imdb_id_to_url)} valid IMDb IDs from URLs")
    
    # Process the gzipped file directly
    updated_count = 0
    processed_lines = 0
    
    try:
        with gzip.open(RATINGS_FILE, 'rt', encoding='utf-8') as f:
            # Skip header
            next(f)
            
            # Process each line
            for line in f:
                processed_lines += 1
                
                parts = line.strip().split('\t')
                if len(parts) < 3:  # Skip lines with insufficient columns
                    continue
                
                imdb_id = parts[0]
                
                # Only process if this ID is in our database
                if imdb_id in imdb_id_to_url:
                    avg_rating = parts[1]
                    num_votes = parts[2]
                    
                    # Update the database
                    url = imdb_id_to_url[imdb_id]
                    cursor.execute(
                        "UPDATE reviews SET movie_average_rating = ?, rating_vote_count = ? WHERE movie_url = ?",
                        (avg_rating, num_votes, url)
                    )
                    
                    updated_count += cursor.rowcount
                    
                    # Remove this ID as it's been processed
                    del imdb_id_to_url[imdb_id]
                
                # Commit every 1000 lines and show progress
                if processed_lines % 10000 == 0:
                    conn.commit()
                    print(f"Processed {processed_lines} lines, updated {updated_count} reviews...")
                
                # If we've found all our IDs, we can stop
                if not imdb_id_to_url:
                    break
        
        # Final commit
        conn.commit()
        
        print(f"Ratings dataset processing complete. Processed {processed_lines} lines.")
        print(f"Updated rating information for {updated_count} reviews.")
        
        if imdb_id_to_url:
            print(f"Could not find rating information for {len(imdb_id_to_url)} IMDb IDs.")
        
    except Exception as e:
        print(f"Error processing ratings dataset: {e}")
    
    finally:
        conn.close()

def main():
    start_time = datetime.now()
    print(f"Starting IMDb metadata import at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current user: {os.getenv('USER', 'taifuranowar')}")
    print(f"Current Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Import IMDb metadata to database')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading the datasets')
    parser.add_argument('--force-download', action='store_true', help='Force download even if files exist')
    parser.add_argument('--basics-only', action='store_true', help='Process only the basics dataset (genres)')
    parser.add_argument('--ratings-only', action='store_true', help='Process only the ratings dataset')
    args = parser.parse_args()
    
    try:
        # Ensure all required columns exist
        ensure_columns_exist()
        
        # Download datasets if needed
        if not args.skip_download:
            if not args.ratings_only:
                if not download_dataset(IMDB_BASICS_URL, BASICS_FILE, force=args.force_download):
                    print("Basics dataset download failed.")
                    if not args.basics_only:
                        print("Continuing with ratings dataset...")
                    else:
                        return
                    
            if not args.basics_only:
                if not download_dataset(IMDB_RATINGS_URL, RATINGS_FILE, force=args.force_download):
                    print("Ratings dataset download failed.")
                    if not args.ratings_only:
                        print("Continuing with basics dataset...")
                    else:
                        return
        
        # Process datasets and update database
        if not args.ratings_only:
            process_basics_dataset()
        
        if not args.basics_only:
            process_ratings_dataset()
        
        # Clean up
        if not args.skip_download:
            if os.path.exists(BASICS_FILE) and not args.ratings_only:
                os.remove(BASICS_FILE)
                print(f"Removed {BASICS_FILE}")
                
            if os.path.exists(RATINGS_FILE) and not args.basics_only:
                os.remove(RATINGS_FILE)
                print(f"Removed {RATINGS_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Import completed in {duration:.2f} seconds at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()