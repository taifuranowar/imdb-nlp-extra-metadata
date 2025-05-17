import asyncio
import json
import os
import re
import sqlite3
import time
from datetime import datetime

from playwright.async_api import async_playwright

# Database configuration
DB_PATH = "imdb_reviews.db"

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
async def extract_histogram_data(page):
    try:
        # Find the histogram container
        histogram_container = await page.query_selector('div[data-testid="rating-histogram"]')
        
        if not histogram_container:
            return None
            
        # Extract review counts for each rating
        histogram = {}
        
        # Use the histogram bars to extract data
        for rating in range(1, 11):
            # Try to find the specific bar
            bar_selector = f'a[data-testid="rating-histogram-bar-{rating}"]'
            bar = await page.query_selector(bar_selector)
            
            if bar:
                # Get the aria-label which contains the count information
                aria_label = await bar.get_attribute('aria-label')
                if aria_label:
                    count_match = re.search(r'(\d+)', aria_label)
                    if count_match:
                        histogram[str(rating)] = int(count_match.group(1))
                    else:
                        histogram[str(rating)] = 0
            else:
                histogram[str(rating)] = 0
        
        # Get the overall rating and vote count
        rating_element = await page.query_selector('span[data-testid="rating-histogram-star"] span.ipc-rating-star--rating')
        vote_count_element = await page.query_selector('span[data-testid="rating-histogram-vote-count"]')
        
        rating = await rating_element.text_content() if rating_element else None
        vote_count = await vote_count_element.text_content() if vote_count_element else None
        
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
async def process_urls(urls, concurrency=5):
    async with async_playwright() as p:
        # Launch a single browser instance
        print("Launching browser...")
        browser = await p.chromium.launch(headless=True)
        
        # Create a single context for the entire session
        context = await browser.new_context()
        
        # Process URLs concurrently, but with a limit
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_url(url):
            async with semaphore:
                try:
                    print(f"Processing: {url}")
                    
                    # Create a new page
                    page = await context.new_page()
                    
                    # Navigate to the URL
                    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    
                    # Wait for the histogram to load
                    try:
                        await page.wait_for_selector('div[data-testid="rating-histogram"]', timeout=10000)
                    except:
                        print(f"Histogram not found for {url}")
                        await page.close()
                        return
                    
                    # Extract histogram data
                    histogram_data = await extract_histogram_data(page)
                    
                    # Close the page
                    await page.close()
                    
                    if histogram_data:
                        # Update the database with histogram data
                        updated = update_histogram_data(url, histogram_data)
                        print(f"Updated {updated} reviews for URL: {url}")
                        print(f"Histogram data: {histogram_data}")
                    else:
                        print(f"No histogram data found for {url}")
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error processing {url}: {e}")
                    # Try to close the page if it's still open
                    try:
                        await page.close()
                    except:
                        pass
        
        # Create tasks for all URLs
        tasks = [process_url(url) for url in urls]
        
        # Process URLs with progress tracking
        completed = 0
        total = len(tasks)
        
        for task in asyncio.as_completed(tasks):
            await task
            completed += 1
            print(f"Progress: {completed}/{total} URLs processed ({(completed/total)*100:.1f}%)")
        
        # Close the browser
        print("Closing browser...")
        await context.close()
        await browser.close()

async def main():
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Current Date and Time (UTC): {current_time}")
    print(f"Current User's Login: {os.getenv('USER', 'taifuranowar')}")
    print(f"Starting IMDB histogram scraping...")
    
    # Check if the database exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found: {DB_PATH}")
        return
    
    # Add the histogram column to the database
    add_histogram_column()
    
    # Get unique movie URLs
    urls = get_unique_movie_urls()
    print(f"Found {len(urls)} unique movie URLs to process")
    
    # Check if there are any URLs to process
    if not urls:
        print("No URLs to process. All histograms may have been collected already.")
        return
    
    # Process the URLs
    start_time = time.time()
    await process_urls(urls)
    
    duration = time.time() - start_time
    print(f"Histogram scraping completed in {duration:.2f} seconds")
    print(f"Average time per URL: {duration/len(urls):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())