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

    # Add columns as needed
    if "user_review_histogram" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN user_review_histogram TEXT")
        print("Added user_review_histogram column to the database")
    if "movie_average_rating" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN movie_average_rating TEXT")
        print("Added movie_average_rating column to the database")
    if "rating_vote_count" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN rating_vote_count TEXT")
        print("Added rating_vote_count column to the database")
    if "movie_genre" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN movie_genre TEXT")
        print("Added movie_genre column to the database")
    if "plot_keywords" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN plot_keywords TEXT")
        print("Added plot_keywords column to the database")
    if "movie_name" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN movie_name TEXT")
        print("Added movie_name column to the database")

    conn.commit()
    conn.close()

# Get all unique movie URLs from the database with their associated row IDs
def get_movie_urls_with_ids():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT movie_url, MIN(id) as first_id
        FROM reviews 
        WHERE user_review_histogram IS NULL
        GROUP BY movie_url
    """)
    url_data = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()
    return url_data

def extract_movie_id(url):
    match = re.search(r'/title/(tt\d+)/', url)
    if match:
        return match.group(1)
    return None

def save_checkpoint(row_id, url, processed_count, total_count):
    checkpoint_data = {
        "last_processed_id": row_id,
        "last_processed_url": url,
        "processed_count": processed_count,
        "total_count": total_count,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"Checkpoint saved: {processed_count}/{total_count} URLs processed (Last ID: {row_id})")

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
    histogram_data = movie_data.get("histogram_data", {})
    average_rating = histogram_data.get("average_rating")
    vote_count = histogram_data.get("vote_count")
    movie_genre = movie_data.get("movie_genre")
    plot_keywords = movie_data.get("plot_keywords")
    movie_name = movie_data.get("movie_name")

    genres_json = json.dumps(movie_genre) if movie_genre else None
    keywords_json = json.dumps(plot_keywords) if plot_keywords else None
    histogram_json = json.dumps(histogram_data) if histogram_data else None

    cursor.execute(
        """
        UPDATE reviews 
        SET user_review_histogram = ?, 
            movie_average_rating = ?,
            rating_vote_count = ?,
            movie_genre = ?,
            plot_keywords = ?,
            movie_name = ?
        WHERE movie_url = ?
        """,
        (histogram_json, average_rating, vote_count, genres_json, keywords_json, movie_name, movie_url)
    )

    updated_rows = cursor.rowcount
    conn.commit()
    conn.close()
    return updated_rows

def extract_movie_name(page):
    try:
        h1 = page.query_selector('h1[data-testid="hero__pageTitle"] span[data-testid="hero__primary-text"]')
        if h1:
            return h1.text_content().strip()
        # Fallback: Try just the h1
        h1_only = page.query_selector('h1[data-testid="hero__pageTitle"]')
        if h1_only:
            return h1_only.text_content().strip()
        return None
    except Exception as e:
        print(f"Error extracting movie name: {e}")
        return None

def extract_plot_keywords(page, movie_url):
    try:
        movie_id = extract_movie_id(movie_url)
        if not movie_id:
            print(f"Could not extract movie ID from URL: {movie_url}")
            return None
        keywords_url = f"https://www.imdb.com/title/{movie_id}/keywords/"
        print(f"Navigating to keywords page: {keywords_url}")
        page.goto(keywords_url, wait_until="domcontentloaded", timeout=60000)
        try:
            page.wait_for_selector('ul.ipc-metadata-list li[data-testid="list-summary-item"]', timeout=10000)
        except Exception as e:
            print(f"No keywords found for {movie_url}: {e}")
            return None
        keyword_elements = page.query_selector_all('li[data-testid="list-summary-item"] a.ipc-metadata-list-summary-item__t')
        keywords = [keyword.text_content().strip() for keyword in keyword_elements if keyword.text_content()]
        print(f"Found {len(keywords)} keywords for {movie_url}")
        return keywords
    except Exception as e:
        print(f"Error extracting plot keywords: {e}")
        return None

def extract_movie_genres(page):
    try:
        genres_container = page.query_selector('div[data-testid="interests"]')
        if not genres_container:
            return None
        genre_elements = genres_container.query_selector_all('a.ipc-chip span.ipc-chip__text')
        genres = [genre.text_content().strip() for genre in genre_elements if genre.text_content()]
        return genres
    except Exception as e:
        print(f"Error extracting movie genres: {e}")
        return None

def extract_histogram_data(page):
    try:
        histogram_container = page.query_selector('div[data-testid="rating-histogram"]')
        if not histogram_container:
            return None
        histogram = {}
        for rating in range(1, 11):
            bar_selector = f'a[data-testid="rating-histogram-bar-{rating}"]'
            bar = page.query_selector(bar_selector)
            if bar:
                aria_label = bar.get_attribute('aria-label')
                if aria_label:
                    count_match = re.search(r'(\d+)', aria_label)
                    if count_match:
                        histogram[str(rating)] = int(count_match.group(1))
                    else:
                        histogram[str(rating)] = 0
            else:
                histogram[str(rating)] = 0
        rating_element = page.query_selector('span[data-testid="rating-histogram-star"] span.ipc-rating-star--rating')
        vote_count_element = page.query_selector('span[data-testid="rating-histogram-vote-count"]')
        rating = rating_element.text_content() if rating_element else None
        vote_count = vote_count_element.text_content() if vote_count_element else None
        histogram_data = {
            "average_rating": rating,
            "vote_count": vote_count,
            "rating_distribution": histogram
        }
        return histogram_data
    except Exception as e:
        print(f"Error extracting histogram data: {e}")
        return None

def extract_movie_data(page, movie_url):
    movie_data = {
        "histogram_data": None,
        "movie_genre": None,
        "plot_keywords": None,
        "movie_name": None
    }
    
    # First extract movie name from the main page
    movie_data["movie_name"] = extract_movie_name(page)
    
    # Then extract other data that exists on the main page
    movie_data["histogram_data"] = extract_histogram_data(page)
    movie_data["movie_genre"] = extract_movie_genres(page)
    
    # Finally extract plot keywords (which requires navigation to another page)
    movie_data["plot_keywords"] = extract_plot_keywords(page, movie_url)
    
    return movie_data

def process_urls(url_data, start_index=0):
    global shutting_down
    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
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
                
                # Navigate to the main movie page
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                
                # Wait for important elements to load
                try:
                    # Wait for either histogram, genres, or the page title
                    page.wait_for_selector('div[data-testid="rating-histogram"], div[data-testid="interests"], h1[data-testid="hero__pageTitle"]', timeout=10000)
                except Exception:
                    print(f"No main data elements found for {url}")
                    continue
                
                # Extract all movie data
                movie_data = extract_movie_data(page, url)
                
                # Check if we found any data
                has_data = (movie_data["histogram_data"] or 
                           movie_data["movie_genre"] or 
                           movie_data["plot_keywords"] or
                           movie_data["movie_name"])
                
                if has_data:
                    # Update the database with all collected data
                    updated = update_movie_data(url, movie_data)
                    print(f"Updated {updated} reviews for URL: {url}")
                    
                    # Log the data we found
                    if movie_data["movie_name"]:
                        print(f"Movie Name: {movie_data['movie_name']}")
                    
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