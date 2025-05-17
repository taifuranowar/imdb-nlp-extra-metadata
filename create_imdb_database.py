import os
import re
import sqlite3
from datetime import datetime

# Database configuration
DB_PATH = "imdb_reviews.db"

def create_database():
    """Create SQLite database with appropriate schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY,
        user_review_sentiment TEXT,
        user_review_rating INTEGER,
        user_review_text TEXT,
        movie_url TEXT,
        dataset_split TEXT
    )
    ''')
    
    conn.commit()
    return conn, cursor

def extract_rating_from_filename(filename):
    """Extract rating from filename (e.g., '0_3.txt' -> 3)"""
    match = re.match(r'^\d+_(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    return None

def extract_id_from_filename(filename):
    """Extract id from filename (e.g., '0_3.txt' -> 0)"""
    match = re.match(r'^(\d+)_\d+\.txt$', filename)
    if match:
        return int(match.group(1))
    return None

def clean_movie_url(url):
    """Remove 'usercomments' from the end of movie URLs"""
    if url.endswith('/usercomments'):
        return url[:-12]  # Remove the '/usercomments' part
    return url

def process_dataset(dataset_path, conn, cursor):
    """Process the IMDB dataset and insert into database"""
    total_inserted = 0
    
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        
        for sentiment in ['pos', 'neg']:
            sentiment_path = os.path.join(split_path, sentiment)
            url_file = os.path.join(split_path, f'urls_{sentiment}.txt')
            
            # Read all URLs
            if os.path.exists(url_file):
                with open(url_file, 'r', encoding='utf-8') as f:
                    urls = f.read().splitlines()
                    # Clean URLs by removing 'usercomments'
                    urls = [clean_movie_url(url) for url in urls]
            else:
                print(f"Warning: URL file not found: {url_file}")
                continue
            
            # Process each review file
            for filename in os.listdir(sentiment_path):
                if not filename.endswith('.txt'):
                    continue
                
                # Extract ID and rating from filename
                file_id = extract_id_from_filename(filename)
                rating = extract_rating_from_filename(filename)
                
                if file_id is None or rating is None:
                    print(f"Warning: Could not parse filename: {filename}")
                    continue
                
                # Get URL for this review
                if file_id < len(urls):
                    movie_url = urls[file_id]
                else:
                    print(f"Warning: No URL found for ID: {file_id}")
                    movie_url = "unknown"
                
                # Read review text
                review_path = os.path.join(sentiment_path, filename)
                try:
                    with open(review_path, 'r', encoding='utf-8') as f:
                        review_text = f.read()
                except UnicodeDecodeError:
                    # Fall back to latin-1 encoding if utf-8 fails
                    with open(review_path, 'r', encoding='latin-1') as f:
                        review_text = f.read()
                
                # Insert into database
                cursor.execute('''
                INSERT INTO reviews (id, user_review_sentiment, user_review_rating, 
                                    user_review_text, movie_url, dataset_split)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (total_inserted, sentiment, rating, review_text, movie_url, split))
                
                total_inserted += 1
                
                # Commit every 1000 records
                if total_inserted % 1000 == 0:
                    conn.commit()
                    print(f"Processed {total_inserted} reviews...")
    
    # Final commit
    conn.commit()
    print(f"Total records inserted: {total_inserted}")

def main():
    print(f"Starting IMDB database creation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current user: {os.getenv('USER', 'taifuranowar')}")
    
    # Path to the IMDB dataset
    dataset_path = "./aclImdb"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    # Create database and tables
    conn, cursor = create_database()
    
    try:
        # Process dataset
        process_dataset(dataset_path, conn, cursor)
        
        # Verify data
        cursor.execute("SELECT COUNT(*) FROM reviews")
        count = cursor.fetchone()[0]
        print(f"Database created successfully with {count} reviews.")
        
        # Show sample data
        cursor.execute("""
            SELECT id, user_review_sentiment, user_review_rating, 
                   substr(user_review_text, 1, 50), 
                   movie_url, dataset_split 
            FROM reviews LIMIT 5
        """)
        print("\nSample data:")
        for row in cursor.fetchall():
            print(row)
    
    finally:
        # Close connection
        conn.close()
    
    print(f"Database creation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()