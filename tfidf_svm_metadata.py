import os
import re
import json
import sqlite3
import numpy as np
import time
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
from collections import defaultdict

# Constants
DB_PATH = "imdb_reviews.db"

def create_archive_directory(base_dir="./training/tfidf-svm-metadata-imdb-archive"):
    """Create timestamped archive directory at the beginning of the experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive_dir = f"{base_dir}_{timestamp}"
    
    # Create the training directory if it doesn't exist
    os.makedirs(os.path.dirname(archive_dir), exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    # Create subdirectories for organization
    os.makedirs(os.path.join(archive_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(archive_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(archive_dir, "metadata"), exist_ok=True)

    print(f"Created archive directory: {archive_dir}")
    return archive_dir

def load_data_from_database():
    """Load reviews, labels, metadata and compute genre/rating statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query to get review data with metadata - FIXED COLUMN NAMES
    cursor.execute("""
        SELECT 
            user_review_text, 
            CASE WHEN user_review_sentiment = 'pos' THEN 1 ELSE 0 END as sentiment, 
            dataset_split,
            movie_genre,
            movie_average_rating,
            rating_vote_count
        FROM reviews
    """)
    
    # Process results
    train_reviews = []
    train_labels = []
    train_metadata = []
    test_reviews = []
    test_labels = []
    test_metadata = []
    
    # Collect genre statistics
    genre_sentiment_stats = defaultdict(lambda: {"pos": 0, "neg": 0, "total": 0})
    rating_sentiment_stats = defaultdict(lambda: {"pos": 0, "neg": 0, "total": 0})
    
    for row in cursor.fetchall():
        review_text = row[0]
        sentiment = row[1]
        split = row[2]
        genre_json = row[3]
        rating = row[4]  # Changed from movie_rating to movie_average_rating
        votes = row[5]   # Changed from movie_votes to rating_vote_count
        
        # Skip rows with missing data
        if not review_text:
            continue
        
        # Parse genres
        genres = []
        if genre_json:
            try:
                genres = json.loads(genre_json)
            except:
                pass
        
        # Create metadata dict
        metadata = {
            "genres": genres,
            "rating": float(rating) if rating else None,
            "votes": int(votes) if votes else None
        }
        
        # Update genre statistics
        for genre in genres:
            genre_sentiment_stats[genre]["total"] += 1
            if sentiment == 1:
                genre_sentiment_stats[genre]["pos"] += 1
            else:
                genre_sentiment_stats[genre]["neg"] += 1
        
        # Update rating statistics
        if rating:
            rating_key = round(float(rating))
            rating_sentiment_stats[rating_key]["total"] += 1
            if sentiment == 1:
                rating_sentiment_stats[rating_key]["pos"] += 1
            else:
                rating_sentiment_stats[rating_key]["neg"] += 1
        
        # Add to appropriate split
        if split == 'train':
            train_reviews.append(review_text)
            train_labels.append(sentiment)
            train_metadata.append(metadata)
        elif split == 'test':
            test_reviews.append(review_text)
            test_labels.append(sentiment)
            test_metadata.append(metadata)
    
    conn.close()
    
    print("Computing genre and rating frequency statistics...")
    
    # Calculate frequency statistics
    genre_freq_stats = {}
    for genre, stats in genre_sentiment_stats.items():
        if stats["total"] > 10:  # Only consider genres with sufficient data
            genre_freq_stats[genre] = {
                "pos_freq": stats["pos"] / stats["total"],
                "neg_freq": stats["neg"] / stats["total"],
                "idf": np.log(len(train_reviews) / stats["total"])  # IDF-like score
            }
    
    rating_freq_stats = {}
    for rating, stats in rating_sentiment_stats.items():
        if stats["total"] > 0:
            rating_freq_stats[rating] = {
                "pos_freq": stats["pos"] / stats["total"],
                "neg_freq": stats["neg"] / stats["total"],
            }
    
    # Print some interesting statistics
    print(f"Found {len(genre_freq_stats)} genres with sufficient data.")
    
    # Find genres with strongest positive/negative correlations
    pos_genres = sorted([(g, s["pos_freq"]) for g, s in genre_freq_stats.items()], 
                         key=lambda x: x[1], reverse=True)[:5]
    neg_genres = sorted([(g, s["neg_freq"]) for g, s in genre_freq_stats.items()], 
                         key=lambda x: x[1], reverse=True)[:5]
    
    print("Genres most associated with positive sentiment:")
    for genre, freq in pos_genres:
        print(f"  - {genre}: {freq*100:.1f}% positive")
    
    print("Genres most associated with negative sentiment:")
    for genre, freq in neg_genres:
        print(f"  - {genre}: {freq*100:.1f}% negative")
    
    return (train_reviews, train_labels, train_metadata, 
            test_reviews, test_labels, test_metadata,
            genre_freq_stats, rating_freq_stats)

def enhance_reviews_with_metadata(reviews, metadata, genre_freq_stats, rating_freq_stats):
    """Enhance reviews with metadata frequency information"""
    enhanced_reviews = []
    
    print("Enhancing reviews with metadata frequency information...")
    for i, review in enumerate(tqdm(reviews, desc="Enhancing reviews")):
        # Start with the original review
        enhanced_review = review
        
        # Add genre frequency information
        if metadata[i]["genres"]:
            genre_info = []
            for genre in metadata[i]["genres"]:
                if genre in genre_freq_stats:
                    # Add genre token with frequency info
                    pos_freq = genre_freq_stats[genre]["pos_freq"]
                    neg_freq = genre_freq_stats[genre]["neg_freq"]
                    idf = genre_freq_stats[genre]["idf"]
                    
                    # If a genre is strongly associated with a sentiment, emphasize it
                    if pos_freq > 0.7:
                        # Add multiple instances based on strength of correlation
                        genre_info.append(f"GENRE_{genre}_POSITIVE " * int(pos_freq * 10))
                    elif neg_freq > 0.7:
                        genre_info.append(f"GENRE_{genre}_NEGATIVE " * int(neg_freq * 10))
                    else:
                        # Add a single instance for neutral genres
                        genre_info.append(f"GENRE_{genre} ")
            
            enhanced_review += " " + " ".join(genre_info)
        
        # Add rating frequency information
        if metadata[i]["rating"]:
            rating_key = round(metadata[i]["rating"])
            if rating_key in rating_freq_stats:
                pos_freq = rating_freq_stats[rating_key]["pos_freq"]
                neg_freq = rating_freq_stats[rating_key]["neg_freq"]
                
                # Add rating tokens based on sentiment correlation
                if pos_freq > 0.7:
                    enhanced_review += f" RATING_HIGH_{rating_key} " * int(pos_freq * 5)
                elif neg_freq > 0.7:
                    enhanced_review += f" RATING_LOW_{rating_key} " * int(neg_freq * 5)
                else:
                    enhanced_review += f" RATING_{rating_key} "
        
        # Add vote count information - higher vote counts have higher reliability
        if metadata[i]["votes"]:
            votes = metadata[i]["votes"]
            # Log transform to handle wide range of votes
            log_votes = np.log1p(votes)
            # Normalize to 1-5 range for token repetition
            vote_weight = min(5, max(1, int(log_votes / 2)))
            enhanced_review += f" VOTES_COUNT_{vote_weight} " * vote_weight
        
        enhanced_reviews.append(enhanced_review)
    
    # Show an example of enhanced review
    if len(enhanced_reviews) > 0:
        print("\nExample of an enhanced review with metadata:")
        sample_idx = min(100, len(enhanced_reviews) - 1)
        
        print("Original:")
        print(reviews[sample_idx][:200] + "..." if len(reviews[sample_idx]) > 200 else reviews[sample_idx])
        
        print("\nEnhanced:")
        enhanced_sample = enhanced_reviews[sample_idx]
        print(enhanced_sample[:200] + "..." if len(enhanced_sample) > 200 else enhanced_sample)
    
    return enhanced_reviews

def clean_text(text):
    """Clean the text by removing HTML tags and normalizing whitespace"""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
    return text.strip()

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def finalize_archive(model, test_reviews, test_labels, y_pred, eval_results, 
                     config, archive_dir, genre_stats=None, rating_stats=None):
    """Save model, predictions, and evaluation results"""
    print(f"Finalizing experiment archive in {archive_dir}...")
    
    # 1. Save the model
    print("Saving model...")
    model_path = os.path.join(archive_dir, "model", "tfidf_svm_metadata_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # 2. Save test data samples
    print("Saving test data samples...")
    metadata_dir = os.path.join(archive_dir, "metadata")
    
    # Save sample test data (100 samples or less)
    sample_size = min(100, len(test_reviews))
    sample_data = []
    for i in range(sample_size):
        sample_data.append({
            "text": test_reviews[i],
            "label": int(test_labels[i]),
            "prediction": int(y_pred[i])
        })
    
    with open(os.path.join(metadata_dir, "test_samples.json"), "w") as f:
        json.dump(sample_data, f, indent=2)
    
    # 3. Save predictions
    print("Saving predictions...")
    predictions_dir = os.path.join(archive_dir, "predictions")
    np.save(os.path.join(predictions_dir, "predictions.npy"), y_pred)
    np.save(os.path.join(predictions_dir, "labels.npy"), test_labels)
    
    # 4. Save genre and rating statistics if available
    if genre_stats:
        with open(os.path.join(metadata_dir, "genre_statistics.json"), "w") as f:
            # Convert to a serializable format
            serializable_genre_stats = {}
            for genre, stats in genre_stats.items():
                serializable_genre_stats[genre] = dict(stats)
            json.dump(serializable_genre_stats, f, indent=2)
    
    if rating_stats:
        with open(os.path.join(metadata_dir, "rating_statistics.json"), "w") as f:
            # Convert to a serializable format
            serializable_rating_stats = {}
            for rating, stats in rating_stats.items():
                serializable_rating_stats[str(rating)] = dict(stats)
            json.dump(serializable_rating_stats, f, indent=2)
    
    # 5. Save evaluation results and config
    print("Saving evaluation metrics and configuration...")
    
    # Enhanced config with all paths and settings
    enhanced_config = {
        **config,
        "paths": {
            "model_path": os.path.abspath(model_path),
            "predictions_dir": os.path.abspath(predictions_dir)
        }
    }
    
    # Save the config
    with open(os.path.join(archive_dir, "experiment_config.json"), "w") as f:
        json.dump(enhanced_config, f, indent=2)
    
    # Save evaluation results
    with open(os.path.join(archive_dir, "evaluation_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # 6. Save experiment summary
    summary = {
        "experiment_id": os.path.basename(archive_dir),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "TF-IDF + SVM with Metadata Integration",
        "accuracy": eval_results.get("accuracy", 0) * 100,
        "f1_score": eval_results.get("f1", 0) * 100,
        "approach": "Frequency-based metadata integration with TF-IDF"
    }
    
    with open(os.path.join(archive_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Archive complete! All data saved to {archive_dir}")
    return archive_dir

def main():
    # Create archive directory
    archive_dir = create_archive_directory()
    
    # Load data from database with metadata and statistics
    print("Loading data from database...")
    (train_reviews, train_labels, train_metadata, 
     test_reviews, test_labels, test_metadata,
     genre_freq_stats, rating_freq_stats) = load_data_from_database()
    
    # Clean text
    print("Cleaning text...")
    train_reviews = [clean_text(review) for review in train_reviews]
    test_reviews = [clean_text(review) for review in test_reviews]
    
    print(f"Training dataset size: {len(train_reviews)}")
    print(f"Testing dataset size: {len(test_reviews)}")
    
    # Enhance reviews with metadata frequency tokens
    enhanced_train_reviews = enhance_reviews_with_metadata(
        train_reviews, train_metadata, genre_freq_stats, rating_freq_stats
    )
    
    enhanced_test_reviews = enhance_reviews_with_metadata(
        test_reviews, test_metadata, genre_freq_stats, rating_freq_stats
    )
    
    # Create and train TF-IDF + SVM pipeline
    print("Training TF-IDF + SVM model with metadata...")
    start_time = time.time()

    # Create TF-IDF features with progress reporting
    print("  Generating TF-IDF features...")
    # Increased max_features to accommodate additional metadata tokens
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(tqdm(enhanced_train_reviews, desc="Vectorizing"))

    # Train SVM with progress reporting
    print("  Training SVM classifier...")
    svm = LinearSVC(C=1.0, verbose=1, max_iter=1000)
    svm.fit(X_train_tfidf, train_labels)

    # Combine into pipeline for predictions
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('svm', svm)
    ])

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print("Making predictions on test set...")
    start_time = time.time()
    y_pred = pipeline.predict(enhanced_test_reviews)
    inference_time = (time.time() - start_time) / len(test_reviews)
    
    # Compute evaluation metrics
    eval_results = compute_metrics(test_labels, y_pred)
    eval_results["training_time_seconds"] = training_time
    eval_results["inference_time_per_sample_ms"] = inference_time * 1000
    
    print(f"Evaluation Results:")
    print(f"Accuracy: {eval_results['accuracy'] * 100:.2f}%")
    print(f"Precision: {eval_results['precision'] * 100:.2f}%")
    print(f"Recall: {eval_results['recall'] * 100:.2f}%")
    print(f"F1 Score: {eval_results['f1'] * 100:.2f}%")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Inference time per sample: {inference_time * 1000:.2f} ms")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, y_pred, target_names=["Negative", "Positive"]))
    
    # Create comprehensive model archive - configuration
    config = {
        "tfidf_params": {
            "max_features": 15000,
            "ngram_range": "1-2"
        },
        "svm_params": {
            "kernel": "linear",
            "C": 1.0
        },
        "metadata_integration": "frequency-based token enhancement",
        "experiment_date": datetime.now().strftime("%Y-%m-%d"),
        "experiment_time": datetime.now().strftime("%H:%M:%S"),
        "description": "IMDB sentiment analysis using TF-IDF + SVM with metadata frequency integration"
    }
    
    # Finalize archive
    finalize_archive(
        model=pipeline,
        test_reviews=test_reviews,
        test_labels=test_labels,
        y_pred=y_pred,
        eval_results=eval_results,
        config=config,
        archive_dir=archive_dir,
        genre_stats=genre_freq_stats,
        rating_stats=rating_freq_stats
    )
    
    print(f"Complete experiment archive created at {archive_dir}")

if __name__ == "__main__":
    main()