# import sqlite3
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import seaborn as sns
    # from scipy.stats import gaussian_kde
    # import matplotlib.ticker as ticker

    # # Connect to SQLite Database
    # db_path = "/home/maira/imdb/imdb-nlp-extra-metadata/imdb_reviews.db"
    # try:
    #     conn = sqlite3.connect(db_path)
    #     query = """
    #     SELECT id, user_review_sentiment, user_review_rating, user_review_text, movie_url, dataset_split, movie_genre, movie_average_rating, rating_vote_count FROM reviews
    #     """
    #     df = pd.read_sql_query(query, conn)
    # except Exception as e:
    #     print(f"Error loading data: {e}")
    #     df = pd.DataFrame()
    # finally:
    #     conn.close()

    # if not df.empty:
    #     # Convert numeric columns
    #     numeric_cols = ['user_review_rating', 'movie_average_rating', 'rating_vote_count']
    #     for col in numeric_cols:
    #         df[col] = pd.to_numeric(df[col], errors='coerce')

    #     # Drop rows with missing values
    #     df = df.dropna()

    #     ### CLASS DISTRIBUTION VISUALIZATION ###
    #     plt.figure(figsize=(6, 4))
    #     if 'user_review_sentiment' in df.columns:
    #         sns.countplot(x=df['user_review_sentiment'], palette="coolwarm")
    #         plt.title('Sentiment Class Distribution', fontsize=14, fontweight='bold')
    #         plt.xlabel('Sentiment Category', fontsize=12)
    #         plt.ylabel('Count', fontsize=12)
    #         plt.grid(True, linestyle='--', alpha=0.5)
    #         plt.show()

    #     ### DATA VISUALIZATION ###
    #     plt.figure(figsize=(15, 10))

    #     # User Review Rating Histogram
    #     plt.subplot(2, 2, 1)
    #     counts, bins, patches = plt.hist(df['user_review_rating'], bins=20, color='skyblue', alpha=0.7)
    #     kde = gaussian_kde(df['user_review_rating'])
    #     x = np.linspace(bins[0], bins[-1], 1000)
    #     plt.plot(x, kde(x) * counts.sum() / (len(bins) - 1), color='darkblue')
    #     plt.title('User Review Rating Distribution', fontsize=14, fontweight='bold')
    #     plt.xlabel('User Rating (1 to 10)', fontsize=12)
    #     plt.ylabel('Frequency', fontsize=12)
    #     plt.xticks(range(1, 11), fontsize=10)
    #     plt.grid(True, linestyle='--', alpha=0.5)

    #     # Movie Average Rating Histogram
    #     plt.subplot(2, 2, 2)
    #     counts, bins, patches = plt.hist(df['movie_average_rating'], bins=20, color='salmon', alpha=0.7)
    #     kde = gaussian_kde(df['movie_average_rating'])
    #     x = np.linspace(bins[0], bins[-1], 1000)
    #     plt.plot(x, kde(x) * counts.sum() / (len(bins) - 1), color='darkred')
    #     plt.title('Movie Average Rating Distribution', fontsize=14, fontweight='bold')
    #     plt.xlabel('Average Movie Rating (1 to 10)', fontsize=12)
    #     plt.ylabel('Frequency', fontsize=12)
    #     plt.xticks(range(int(df['movie_average_rating'].min()), int(df['movie_average_rating'].max() + 1)), fontsize=10)
    #     plt.grid(True, linestyle='--', alpha=0.5)

    #     # Vote Count Distribution (Log Scale)
    #     plt.subplot(2, 2, 3)
    #     plt.hist(df['rating_vote_count'], bins=np.logspace(
    #         np.log10(df['rating_vote_count'].min()), 
    #         np.log10(df['rating_vote_count'].max()), 
    #         20), color='lightgreen', alpha=0.7)
    #     plt.xscale('log')
    #     plt.title('Rating Vote Count Distribution (Log Scale)', fontsize=14, fontweight='bold')
    #     plt.xlabel('Vote Count (Log Scale)', fontsize=12)
    #     plt.ylabel('Frequency', fontsize=12)
    #     plt.grid(True, linestyle='--', alpha=0.5)

    #     # Scatter Plot of User Ratings vs. Movie Average Rating
    #     plt.subplot(2, 2, 4)
    #     plt.scatter(df['movie_average_rating'], df['user_review_rating'], alpha=0.3, color='purple')
    #     plt.title('User Rating vs Movie Average Rating', fontsize=14, fontweight='bold')
    #     plt.xlabel('Movie Average Rating', fontsize=12)
    #     plt.ylabel('User Review Rating', fontsize=12)
    #     plt.xticks(range(int(df['movie_average_rating'].min()), int(df['movie_average_rating'].max() + 1)), fontsize=10)
    #     plt.yticks(range(1, 11), fontsize=10)
    #     plt.grid(True, linestyle='--', alpha=0.5)

    #     plt.tight_layout()
    #     plt.show()

    # else:
    #     print("No data loaded - please check your database connection and query")

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker

# Connect to SQLite Database
db_path = "imdb_reviews.db"
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
    # Convert numeric columns
    numeric_cols = ['user_review_rating', 'movie_average_rating', 'rating_vote_count']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values
    df = df.dropna()

    ### CLASS DISTRIBUTION VISUALIZATION ###
    plt.figure(figsize=(6, 4))
    if 'user_review_sentiment' in df.columns:
        sns.countplot(x=df['user_review_sentiment'], palette="coolwarm")
        plt.title('Sentiment Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment Category', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    ### DATA VISUALIZATION ###
    plt.figure(figsize=(15, 10))

    # User Review Rating Histogram
    plt.subplot(2, 2, 1)
    counts, bins, patches = plt.hist(df['user_review_rating'], bins=20, color='skyblue', alpha=0.7)
    kde = gaussian_kde(df['user_review_rating'])
    x = np.linspace(bins[0], bins[-1], 1000)
    plt.plot(x, kde(x) * counts.sum() / (len(bins) - 1), color='darkblue')
    plt.title('User Review Rating Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('User Rating (1 to 10)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(range(1, 11), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Movie Average Rating Histogram
    plt.subplot(2, 2, 2)
    counts, bins, patches = plt.hist(df['movie_average_rating'], bins=20, color='salmon', alpha=0.7)
    kde = gaussian_kde(df['movie_average_rating'])
    x = np.linspace(bins[0], bins[-1], 1000)
    plt.plot(x, kde(x) * counts.sum() / (len(bins) - 1), color='darkred')
    plt.title('Movie Average Rating Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Average Movie Rating (1 to 10)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(range(int(df['movie_average_rating'].min()), int(df['movie_average_rating'].max() + 1)), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Vote Count Distribution (Log Scale)
    plt.subplot(2, 2, 3)
    plt.hist(df['rating_vote_count'], bins=np.logspace(
        np.log10(df['rating_vote_count'].min()), 
        np.log10(df['rating_vote_count'].max()), 
        20), color='lightgreen', alpha=0.7)
    plt.xscale('log')
    plt.title('Rating Vote Count Distribution (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Vote Count (Log Scale)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Top 10 Movie Genres Barplot (replaces scatter plot)
    plt.subplot(2, 2, 4)
    # Clean up genre strings: remove brackets and quotes, keep commas
    cleaned_genres = df['movie_genre'].astype(str).str.replace(r"[\[\]']", "", regex=True)
    top_genres = cleaned_genres.value_counts().nlargest(10)
    sns.barplot(x=top_genres.index, y=top_genres.values, palette="magma")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.xlabel("Movie Genre", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Top 10 Movie Genres", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle="--", alpha=0.5)
    plt.subplots_adjust(bottom=0.3)  # Add extra space at bottom

    plt.tight_layout()
    plt.show()

else:
    print("No data loaded - please check your database connection and query")





# =============================
# # Detect outliers using IQR
# def detect_outliers(data):
#     Q1 = data.quantile(0.25)
#     Q3 = data.quantile(0.75)
#     IQR = Q3 - Q1
#     return data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]

# # Extract outliers
# outliers_rating = detect_outliers(df['user_review_rating'])
# outliers_movie_avg = detect_outliers(df['movie_average_rating'])
# outliers_vote_count = detect_outliers(df['rating_vote_count'])

# # Plot Bell Curve for Outliers
# plt.figure(figsize=(12, 6))

# # Movie Average Rating Outliers Bell Curve
# sns.histplot(outliers_movie_avg, bins=20, kde=True, color="red", alpha=0.7)
# plt.title("Bell Curve for Movie Average Rating Outliers", fontsize=14, fontweight="bold")
# plt.xlabel("Movie Average Rating", fontsize=12)
# plt.ylabel("Density", fontsize=12)
# plt.grid(True, linestyle="--", alpha=0.5)

# plt.show()
