import json
import os
import re
import signal
import sqlite3
import sys
import time
from datetime import datetime
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

# Database and checkpoint configuration
DB_PATH = "imdb_reviews.db"
CHECKPOINT_FILE = "imdb_scraper_checkpoint.json"

# Global variable to track if we're shutting down
shutting_down = False

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global shutting_down
    print("\nGraceful shutdown initiated. Completing current URL before exiting...")
    shutting_down = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Add necessary columns to the database if they don't exist
def add_required_columns():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check existing columns
    cursor.execute("PRAGMA table_info(reviews)")
    columns = [info[1] for info in cursor.fetchall()]
    
    # Add histogram column if it doesn't exist
    if "user_review_histogram" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN user_review_histogram TEXT")
        print("Added user_review_histogram column to the database")
    
    # Add movie_average_rating column if it doesn't exist
    if "movie_average_rating" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN movie_average_rating TEXT")
        print("Added movie_average_rating column to the database")
    
    # Add rating_vote_count column if it doesn't exist
    if "rating_vote_count" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN rating_vote_count TEXT")
        print("Added rating_vote_count column to the database")
    
    # Add movie_genre column if it doesn't exist
    if "movie_genre" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN movie_genre TEXT")
        print("Added movie_genre column to the database")
    
    # Add plot_keywords column if it doesn't exist
    if "plot_keywords" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN plot_keywords TEXT")
        print("Added plot_keywords column to the database")
    
    conn.commit()
    conn.close()

# Get all unique movie URLs from the database with their associated row IDs
def get_movie_urls_with_ids():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get distinct movie URLs with the minimum row ID for each URL
    cursor.execute("""
        SELECT movie_url, MIN(id) as first_id
        FROM reviews 
        WHERE user_review_histogram IS NULL
        GROUP BY movie_url
    """)
    
    url_data = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()
    return url_data

# Extract the movie ID from the URL
def extract_movie_id(url):
    match = re.search(r'/title/(tt\d+)/', url)
    if match:
        return match.group(1)
    return None

# Save checkpoint with the last processed database ID
def save_checkpoint(row_id, url, processed_count, total_count):
    checkpoint_data = {
        "last_processed_id": row_id,
        "last_processed_url": url,  # Just for logging/display purposes
        "processed_count": processed_count,
        "total_count": total_count,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"Checkpoint saved: {processed_count}/{total_count} URLs processed (Last ID: {row_id})")

# Load checkpoint to resume scraping
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    
    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoint_data = json.load(f)
    
    print(f"Checkpoint found from {checkpoint_data['timestamp']}")
    print(f"Resuming after database ID: {checkpoint_data['last_processed_id']}")
    print(f"Previously processed: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']} URLs")
    
    return checkpoint_data

# Update the database with all movie data
def update_movie_data(movie_url, movie_data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Extract data from the movie_data dictionary
    histogram_data = movie_data.get("histogram_data", {})
    average_rating = histogram_data.get("average_rating") 
    vote_count = histogram_data.get("vote_count")
    movie_genre = movie_data.get("movie_genre")
    plot_keywords = movie_data.get("plot_keywords")
    
    # Convert data to JSON strings
    genres_json = json.dumps(movie_genre) if movie_genre else None
    keywords_json = json.dumps(plot_keywords) if plot_keywords else None
    histogram_json = json.dumps(histogram_data) if histogram_data else None
    
    # Update all reviews for this movie URL with all fields
    cursor.execute(
        """
        UPDATE reviews 
        SET user_review_histogram = ?, 
            movie_average_rating = ?,
            rating_vote_count = ?,
            movie_genre = ?,
            plot_keywords = ?
        WHERE movie_url = ?
        """,
        (histogram_json, average_rating, vote_count, genres_json, keywords_json, movie_url)
    )
    
    updated_rows = cursor.rowcount
    conn.commit()
    conn.close()
    
    return updated_rows

# Extract plot keywords from the keywords page
def extract_plot_keywords(page, movie_url):
    try:
        # Extract movie ID from URL
        movie_id = extract_movie_id(movie_url)
        if not movie_id:
            print(f"Could not extract movie ID from URL: {movie_url}")
            return None
        
        # Construct keywords URL
        keywords_url = f"https://www.imdb.com/title/{movie_id}/keywords/"
        
        # Navigate to the keywords page
        print(f"Navigating to keywords page: {keywords_url}")
        page.goto(keywords_url, wait_until="domcontentloaded", timeout=60000)
        
        # Wait for keywords to load
        try:
            page.wait_for_selector('ul.ipc-metadata-list li[data-testid="list-summary-item"]', timeout=10000)
        except Exception as e:
            print(f"No keywords found for {movie_url}: {e}")
            return None
        
        # Extract all keyword elements
        keyword_elements = page.query_selector_all('li[data-testid="list-summary-item"] a.ipc-metadata-list-summary-item__t')
        
        # Extract the text from each keyword element
        keywords = [keyword.text_content().strip() for keyword in keyword_elements if keyword.text_content()]
        
        print(f"Found {len(keywords)} keywords for {movie_url}")
        return keywords
        
    except Exception as e:
        print(f"Error extracting plot keywords: {e}")
        return None

# Extract movie genres from the page
def extract_movie_genres(page):
    try:
        # Find the interests/genres container
        genres_container = page.query_selector('div[data-testid="interests"]')
        
        if not genres_container:
            return None
        
        # Extract all chip elements that contain the genres
        genre_elements = genres_container.query_selector_all('a.ipc-chip span.ipc-chip__text')
        
        # Extract the text from each genre element
        genres = [genre.text_content().strip() for genre in genre_elements if genre.text_content()]
        
        return genres
        
    except Exception as e:
        print(f"Error extracting movie genres: {e}")
        return None

# Extract histogram data using precise selectors
def extract_histogram_data(page):
    try:
        # Find the histogram container
        histogram_container = page.query_selector('div[data-testid="rating-histogram"]')
        
        if not histogram_container:
            return None
            
        # Extract review counts for each rating
        histogram = {}
        
        # Use the histogram bars to extract data
        for rating in range(1, 11):
            # Try to find the specific bar
            bar_selector = f'a[data-testid="rating-histogram-bar-{rating}"]'
            bar = page.query_selector(bar_selector)
            
            if bar:
                # Get the aria-label which contains the count information
                aria_label = bar.get_attribute('aria-label')
                if aria_label:
                    count_match = re.search(r'(\d+)', aria_label)
                    if count_match:
                        histogram[str(rating)] = int(count_match.group(1))
                    else:
                        histogram[str(rating)] = 0
            else:
                histogram[str(rating)] = 0
        
        # Get the overall rating and vote count
        rating_element = page.query_selector('span[data-testid="rating-histogram-star"] span.ipc-rating-star--rating')
        vote_count_element = page.query_selector('span[data-testid="rating-histogram-vote-count"]')
        
        rating = rating_element.text_content() if rating_element else None
        vote_count = vote_count_element.text_content() if vote_count_element else None
        
        # Compile the complete histogram data
        histogram_data = {
            "average_rating": rating,
            "vote_count": vote_count,
            "rating_distribution": histogram
        }
        
        return histogram_data
        
    except Exception as e:
        print(f"Error extracting histogram data: {e}")
        return None

# Extract all movie data from the page
def extract_movie_data(page, movie_url):
    # Initialize the movie data dictionary
    movie_data = {
        "histogram_data": None,
        "movie_genre": None,
        "plot_keywords": None
    }
    
    # Extract histogram data
    histogram_data = extract_histogram_data(page)
    if histogram_data:
        movie_data["histogram_data"] = histogram_data
    
    # Extract movie genres
    movie_genre = extract_movie_genres(page)
    if movie_genre:
        movie_data["movie_genre"] = movie_genre
    
    # Extract plot keywords (this will navigate to a different page)
    plot_keywords = extract_plot_keywords(page, movie_url)
    if plot_keywords:
        movie_data["plot_keywords"] = plot_keywords
    
    return movie_data

# Process all URLs in a single browser session
def process_urls(url_data, start_index=0):
    global shutting_down
    
    with sync_playwright() as p:
        # Launch a single browser instance - visible (not headless)
        print("Launching browser...")
        browser = p.chromium.launch(headless=False)
        
        # Create a single context for the entire session
        context = browser.new_context()
        
        # Create a single page for reuse
        page = context.new_page()
        
        # Process URLs sequentially
        completed = start_index
        total = len(url_data)
        
        for i in range(start_index, len(url_data)):
            url, row_id = url_data[i]
            
            # Check if we should shut down
            if shutting_down:
                print("Graceful shutdown activated. Saving checkpoint and exiting...")
                prev_id = url_data[i-1][1] if i > 0 else 0
                prev_url = url_data[i-1][0] if i > 0 else ""
                save_checkpoint(prev_id, prev_url, completed, total)
                break
                
            try:
                print(f"Processing: {url} (ID: {row_id})")
                
                # Navigate to the main movie URL
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                
                # Wait for either the histogram or the genres to load
                try:
                    page.wait_for_selector('div[data-testid="rating-histogram"], div[data-testid="interests"]', timeout=10000)
                except Exception:
                    print(f"No rating histogram or genres found for {url}")
                    continue
                
                # Extract all movie data (including keywords from separate page)
                movie_data = extract_movie_data(page, url)
                
                # Update the database if we found any data
                has_data = (movie_data["histogram_data"] or 
                           movie_data["movie_genre"] or 
                           movie_data["plot_keywords"])
                
                if has_data:
                    # Update the database with all collected data
                    updated = update_movie_data(url, movie_data)
                    print(f"Updated {updated} reviews for URL: {url}")
                    
                    # Log the data we found
                    if movie_data["histogram_data"]:
                        print(f"Average Rating: {movie_data['histogram_data'].get('average_rating')}, Vote Count: {movie_data['histogram_data'].get('vote_count')}")
                    
                    if movie_data["movie_genre"]:
                        print(f"Movie Genres: {', '.join(movie_data['movie_genre'])}")
                    
                    if movie_data["plot_keywords"]:
                        keyword_sample = movie_data["plot_keywords"][:5]  # Show first 5 keywords
                        print(f"Plot Keywords: {', '.join(keyword_sample)}... ({len(movie_data['plot_keywords'])} total)")
                else:
                    print(f"No data found for {url}")
                
                # Update completed count
                completed += 1
                
                # Save checkpoint every 5 URLs
                if completed % 5 == 0:
                    save_checkpoint(row_id, url, completed, total)
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
            
            # Update progress
            print(f"Progress: {completed}/{total} URLs processed ({(completed/total)*100:.1f}%)")
        
        # Final checkpoint save
        if not shutting_down and completed > start_index and completed < total:
            last_id = url_data[completed-1][1]
            last_url = url_data[completed-1][0]
            save_checkpoint(last_id, last_url, completed, total)
        
        # Close the browser
        print("Closing browser...")
        page.close()
        context.close()
        browser.close()
        
        return completed

def find_start_index(url_data, last_id):
    """Find the index to start from based on the last processed ID"""
    for i, (url, row_id) in enumerate(url_data):
        if row_id == last_id:
            return i + 1  # Start with the next one
    return 0  # If not found, start from the beginning

def main():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {current_time}")
    print(f"Current User's Login: {os.getenv('USER', 'taifuranowar')}")
    print(f"Starting IMDB data scraping...")
    
    # Check if the database exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found: {DB_PATH}")
        return
    
    # Add required columns to the database
    add_required_columns()
    
    # Get unique movie URLs with their IDs
    url_data = get_movie_urls_with_ids()
    print(f"Found {len(url_data)} unique movie URLs to process")
    
    # Check if there are any URLs to process
    if not url_data:
        print("No URLs to process. All movie data may have been collected already.")
        return
    
    # Check for checkpoint and determine starting point
    checkpoint = load_checkpoint()
    start_index = 0
    
    if checkpoint:
        # Find the index of the last processed URL by ID
        last_id = checkpoint["last_processed_id"]
        start_index = find_start_index(url_data, last_id)
        
        print(f"Resuming from URL index {start_index}")
        
        # Check if we've already completed all URLs
        if start_index >= len(url_data):
            print("All URLs have been processed according to checkpoint.")
            # Clean up checkpoint file
            os.remove(CHECKPOINT_FILE)
            return
    
    # Process the URLs
    start_time = time.time()
    total_processed = process_urls(url_data, start_index)
    
    # If we've processed all URLs successfully, remove the checkpoint file
    if total_processed == len(url_data) and not shutting_down:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("Checkpoint file removed as all URLs were processed successfully.")
    
    duration = time.time() - start_time
    print(f"Movie data scraping completed in {duration:.2f} seconds")
    if total_processed > start_index:
        print(f"Average time per URL: {duration/(total_processed-start_index):.2f} seconds")

if __name__ == "__main__":
    print("Press Ctrl+C to gracefully exit and save checkpoint")
    main()