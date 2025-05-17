import json
import os
import re
import signal
import sqlite3
import sys
import time
from datetime import datetime

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

# Add a column for the histogram data if it doesn't exist
def add_histogram_column():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if the column already exists
    cursor.execute("PRAGMA table_info(reviews)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if "user_review_histogram" not in columns:
        cursor.execute("ALTER TABLE reviews ADD COLUMN user_review_histogram TEXT")
        print("Added user_review_histogram column to the database")
    
    conn.commit()
    conn.close()

# Get all unique movie URLs from the database
def get_unique_movie_urls():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT movie_url FROM reviews WHERE user_review_histogram IS NULL")
    urls = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return urls

# Save checkpoint with the last processed URL
def save_checkpoint(url, processed_count, total_count):
    checkpoint_data = {
        "last_processed_url": url,
        "processed_count": processed_count,
        "total_count": total_count,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"Checkpoint saved: {processed_count}/{total_count} URLs processed")

# Load checkpoint to resume scraping
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    
    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoint_data = json.load(f)
    
    print(f"Checkpoint found from {checkpoint_data['timestamp']}")
    print(f"Resuming after URL: {checkpoint_data['last_processed_url']}")
    print(f"Previously processed: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']} URLs")
    
    return checkpoint_data

# Update the database with the histogram data
def update_histogram_data(movie_url, histogram_data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert histogram data to JSON string
    histogram_json = json.dumps(histogram_data)
    
    # Update all reviews for this movie URL
    cursor.execute(
        "UPDATE reviews SET user_review_histogram = ? WHERE movie_url = ?",
        (histogram_json, movie_url)
    )
    
    updated_rows = cursor.rowcount
    conn.commit()
    conn.close()
    
    return updated_rows

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

# Process all URLs in a single browser session
def process_urls(urls, start_index=0):
    global shutting_down
    
    with sync_playwright() as p:
        # Launch a single browser instance - visible (not headless)
        print("Launching browser...")
        browser = p.chromium.launch(headless=False)
        
        # Create a single context for the entire session
        context = browser.new_context()
        
        # Process URLs sequentially
        completed = start_index
        total = len(urls)
        
        for i in range(start_index, len(urls)):
            url = urls[i]
            
            # Check if we should shut down
            if shutting_down:
                print("Graceful shutdown activated. Saving checkpoint and exiting...")
                save_checkpoint(urls[i-1] if i > 0 else "", completed, total)
                break
                
            try:
                print(f"Processing: {url}")
                
                # Create a new page
                page = context.new_page()
                
                # Navigate to the URL
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                
                # Wait for the histogram to load
                try:
                    page.wait_for_selector('div[data-testid="rating-histogram"]', timeout=10000)
                except Exception:
                    print(f"Histogram not found for {url}")
                    page.close()
                    continue
                
                # Extract histogram data
                histogram_data = extract_histogram_data(page)
                
                # Close the page
                page.close()
                
                if histogram_data:
                    # Update the database with histogram data
                    updated = update_histogram_data(url, histogram_data)
                    print(f"Updated {updated} reviews for URL: {url}")
                    print(f"Histogram data: {histogram_data}")
                else:
                    print(f"No histogram data found for {url}")
                
                # Update completed count and save checkpoint
                completed += 1
                
                # Save checkpoint every 5 URLs
                if completed % 5 == 0:
                    save_checkpoint(url, completed, total)
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
                # Try to close the page if it's still open
                try:
                    page.close()
                except:
                    pass
            
            # Update progress
            print(f"Progress: {completed}/{total} URLs processed ({(completed/total)*100:.1f}%)")
        
        # Final checkpoint save
        if not shutting_down and completed > start_index:
            save_checkpoint(urls[completed-1], completed, total)
        
        # Close the browser
        print("Closing browser...")
        context.close()
        browser.close()
        
        return completed

def main():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {current_time}")
    print(f"Current User's Login: {os.getenv('USER', 'taifuranowar')}")
    print(f"Starting IMDB histogram scraping...")
    
    # Check if the database exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found: {DB_PATH}")
        return
    
    # Add the histogram column to the database
    add_histogram_column()
    
    # Get unique movie URLs
    all_urls = get_unique_movie_urls()
    print(f"Found {len(all_urls)} unique movie URLs to process")
    
    # Check if there are any URLs to process
    if not all_urls:
        print("No URLs to process. All histograms may have been collected already.")
        return
    
    # Check for checkpoint and determine starting point
    checkpoint = load_checkpoint()
    start_index = 0
    
    if checkpoint:
        # Find the index of the last processed URL
        last_url = checkpoint["last_processed_url"]
        if last_url in all_urls:
            start_index = all_urls.index(last_url) + 1
        
        print(f"Resuming from URL index {start_index}")
        
        # Check if we've already completed all URLs
        if start_index >= len(all_urls):
            print("All URLs have been processed according to checkpoint.")
            # Clean up checkpoint file
            os.remove(CHECKPOINT_FILE)
            return
    
    # Process the URLs
    start_time = time.time()
    total_processed = process_urls(all_urls, start_index)
    
    # If we've processed all URLs successfully, remove the checkpoint file
    if total_processed == len(all_urls) and not shutting_down:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("Checkpoint file removed as all URLs were processed successfully.")
    
    duration = time.time() - start_time
    print(f"Histogram scraping completed in {duration:.2f} seconds")
    if total_processed > 0:
        print(f"Average time per URL: {duration/(total_processed-start_index):.2f} seconds")

if __name__ == "__main__":
    print("Press Ctrl+C to gracefully exit and save checkpoint")
    main()