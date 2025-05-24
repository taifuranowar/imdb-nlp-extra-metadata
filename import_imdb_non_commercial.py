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
IMDB_DATASET_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
DOWNLOADED_FILE = "title.basics.tsv.gz"

# Extract movie ID from URL
def extract_movie_id(url):
    match = re.search(r'/title/(tt\d+)/', url)
    if match:
        return match.group(1)
    return None

# Download the dataset
def download_dataset(force=False):
    if os.path.exists(DOWNLOADED_FILE) and not force:
        print(f"{DOWNLOADED_FILE} already exists. Skipping download.")
        return True
    
    print(f"Downloading dataset from {IMDB_DATASET_URL}...")
    try:
        urllib.request.urlretrieve(IMDB_DATASET_URL, DOWNLOADED_FILE)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

# Process the dataset and update the database
def process_dataset_and_update_db():
    print("Processing dataset and updating database...")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if the movie_genre column exists
    cursor.execute("PRAGMA table_info(reviews)")
    columns = [info[1] for info in cursor.fetchall()]
    
    # Add the column if it doesn't exist
    if "movie_genre" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN movie_genre TEXT")
        print("Added movie_genre column to the database")
    
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
        with gzip.open(DOWNLOADED_FILE, 'rt', encoding='utf-8') as f:
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
                if processed_lines % 1000 == 0:
                    conn.commit()
                    print(f"Processed {processed_lines} lines, updated {updated_count} reviews...")
                
                # If we've found all our IDs, we can stop
                if not imdb_id_to_url:
                    break
        
        # Final commit
        conn.commit()
        
        print(f"Dataset processing complete. Processed {processed_lines} lines.")
        print(f"Updated genre information for {updated_count} reviews.")
        
        if imdb_id_to_url:
            print(f"Could not find genre information for {len(imdb_id_to_url)} IMDb IDs.")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
    
    finally:
        conn.close()

def main():
    start_time = datetime.now()
    print(f"Starting IMDb genre import at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current user: {os.getenv('USER', 'taifuranowar')}")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Import IMDb genre information to database')
    parser.add_argument('--skip-download', action='store_true', help='Skip downloading the dataset')
    parser.add_argument('--force-download', action='store_true', help='Force download even if file exists')
    args = parser.parse_args()
    
    try:
        # Download dataset if needed
        if not args.skip_download:
            if not download_dataset(force=args.force_download):
                print("Download failed. Exiting.")
                return
        
        # Process dataset and update database
        process_dataset_and_update_db()
        
        # Clean up
        if os.path.exists(DOWNLOADED_FILE) and not args.skip_download:
            os.remove(DOWNLOADED_FILE)
            print(f"Removed {DOWNLOADED_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Import completed in {duration:.2f} seconds at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()