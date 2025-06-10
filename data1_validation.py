import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

# Connect to SQLite Database/home/maira/imdb/imdb-nlp-extra-metadata/aclImdb/imdb_reviews.db
db_path = "/home/maira/imdb/imdb-nlp-extra-metadata/imdb_reviews.db"
try:
    conn = sqlite3.connect(db_path)
    query = """
    SELECT id, user_review_sentiment, user_review_rating, user_review_text, movie_url, dataset_split, movie_genre, movie_average_rating, rating_vote_count FROM reviews
    """
    df = pd.read_sql_query(query, conn)
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()
finally:
    conn.close()

if not df.empty:
    # Data type conversions
    numeric_cols = ['user_review_rating', 'movie_average_rating', 'rating_vote_count']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values
    df = df.dropna()

    ### VALIDATION CHECKS ###
    
    # 1. Missing Values Check
    print("===MISSING VALUES ===")
    print(df.isnull().sum())

    # 2. Outlier Detection (Using IQR Method)
    print("=== OUTLIER DETECTION ===")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        print(f"{col}: {outlier_mask.sum()} potential outliers detected")

    # 3. Class Imbalance Analysis
    print("=== CLASS DISTRIBUTION ===")
    if 'user_review_sentiment' in df.columns:
        sentiment_counts = df['user_review_sentiment'].value_counts()
        print(sentiment_counts)

    ### TEXT PREPROCESSING FOR NLP ###
    print("=== TEXT DATA VALIDATION ===")

    # Remove empty or very short reviews
    df['review_length'] = df['user_review_text'].apply(lambda x: len(str(x).split()))
    print(f"Minimum words in a review: {df['review_length'].min()}")
    print(f"Maximum words in a review: {df['review_length'].max()}")
    
    # Remove duplicate reviews
    print(f"Duplicate reviews removed: {df.duplicated(subset=['user_review_text']).sum()}")
    df = df.drop_duplicates(subset=['user_review_text'])
    print("Duplicate reviews:")
    print(df[df.duplicated(subset=['user_review_text'], keep=False)])

    ### DATA VISUALIZATION ###
    plt.figure(figsize=(15, 10))

    # User Review Rating Distribution
    plt.subplot(2, 2, 1)
    counts, bins, patches = plt.hist(df['user_review_rating'], bins=20, color='skyblue', alpha=0.7)
    kde = gaussian_kde(df['user_review_rating'])
    x = np.linspace(bins[0], bins[-1], 1000)
    plt.plot(x, kde(x)*counts.sum()/(len(bins)-1), color='darkblue')
    plt.title('User Review Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # Movie Average Rating Distribution
    plt.subplot(2, 2, 2)
    counts, bins, patches = plt.hist(df['movie_average_rating'], bins=20, color='salmon', alpha=0.7)
    kde = gaussian_kde(df['movie_average_rating'])
    x = np.linspace(bins[0], bins[-1], 1000)
    plt.plot(x, kde(x)*counts.sum()/(len(bins)-1), color='darkred')
    plt.title('Movie Average Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # Vote Count Distribution (Log Scale)
    plt.subplot(2, 2, 3)
    plt.hist(df['rating_vote_count'], bins=np.logspace(
        np.log10(df['rating_vote_count'].min()), 
        np.log10(df['rating_vote_count'].max()), 
        20), color='lightgreen', alpha=0.7)
    plt.xscale('log')
    plt.title('Rating Vote Count Distribution (Log Scale)')
    plt.xlabel('Vote Count (log scale)')
    plt.ylabel('Frequency')

    # Scatter Plot of User Ratings vs. Movie Average Rating
    plt.subplot(2, 2, 4)
    plt.scatter(df['movie_average_rating'], df['user_review_rating'], alpha=0.3, color='purple')
    plt.title('User Rating vs Movie Average Rating')
    plt.xlabel('Movie Average Rating')
    plt.ylabel('User Review Rating')

    plt.tight_layout()
    plt.show()

else:
    print("No data loaded - please check your database connection and query")