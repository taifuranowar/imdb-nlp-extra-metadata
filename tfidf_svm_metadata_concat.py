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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from scipy import sparse
from tqdm import tqdm
from collections import defaultdict, Counter

# Constants
DB_PATH = "imdb_reviews.db"

def create_archive_directory(base_dir="./training/tfidf-svm-concat-metadata-imdb-archive"):
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
    
    # Query to get review data with metadata
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
    
    # Collect unique genres for one-hot encoding
    all_genres = set()
    
    for row in cursor.fetchall():
        review_text = row[0]
        sentiment = row[1]
        split = row[2]
        genre_json = row[3]
        rating = row[4]
        votes = row[5]
        
        # Skip rows with missing data
        if not review_text:
            continue
        
        # Parse genres
        genres = []
        if genre_json:
            try:
                genres = json.loads(genre_json)
                all_genres.update(genres)
            except:
                pass
        
        # Create metadata dict
        metadata = {
            "genres": genres,
            "rating": float(rating) if rating else None,
            "votes": int(votes) if votes else None
        }
        
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
    
    # Convert genres to list for consistent ordering
    all_genres = sorted(list(all_genres))
    print(f"Found {len(all_genres)} unique genres.")
    
    return (train_reviews, train_labels, train_metadata, 
            test_reviews, test_labels, test_metadata, all_genres)

def create_metadata_features(metadata_list, all_genres):
    """Create numerical features from metadata for concatenation"""
    print("Creating metadata features for concatenation...")
    
    # Number of samples and features
    n_samples = len(metadata_list)
    n_genre_features = len(all_genres)
    n_rating_features = 1  # Rating is a single feature
    n_votes_features = 1   # Vote count is a single feature
    n_features = n_genre_features + n_rating_features + n_votes_features
    
    # Create sparse matrix for efficiency
    X_metadata = np.zeros((n_samples, n_features))
    
    # Create a genre lookup for faster access
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
    
    # Populate metadata features
    for i, metadata in enumerate(tqdm(metadata_list, desc="Processing metadata")):
        # Genre one-hot encoding
        for genre in metadata['genres']:
            if genre in genre_to_idx:
                X_metadata[i, genre_to_idx[genre]] = 1
        
        # Rating feature (normalized between 0 and 1)
        if metadata['rating'] is not None:
            # Scale from 1-10 to 0-1
            X_metadata[i, n_genre_features] = (metadata['rating'] - 1) / 9
        else:
            # Use mean rating if missing (5.5)
            X_metadata[i, n_genre_features] = 0.5
        
        # Vote count feature (log-transformed and normalized)
        if metadata['votes'] is not None:
            # Log transform to handle skew
            log_votes = np.log1p(metadata['votes'])
            # Normalize based on typical range (1 to ~14 after log transform)
            X_metadata[i, n_genre_features + 1] = min(1.0, log_votes / 12.0)
        else:
            X_metadata[i, n_genre_features + 1] = 0
    
    print(f"Created metadata feature matrix with shape: {X_metadata.shape}")
    
    # Create feature names for debugging/analysis
    feature_names = all_genres + ['rating', 'vote_count']
    
    return X_metadata, feature_names

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

def finalize_archive(model, tfidf_model, metadata_feature_names, test_reviews, test_labels, test_metadata, y_pred, 
                     eval_results, config, archive_dir):
    """Save model, predictions, and evaluation results"""
    print(f"Finalizing experiment archive in {archive_dir}...")
    
    # 1. Save the model
    print("Saving model...")
    model_path = os.path.join(archive_dir, "model", "tfidf_svm_concat_metadata_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save TF-IDF model separately
    tfidf_path = os.path.join(archive_dir, "model", "tfidf_model.pkl")
    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf_model, f)
    
    # 2. Save metadata feature names
    with open(os.path.join(archive_dir, "metadata", "metadata_features.json"), 'w') as f:
        json.dump(metadata_feature_names, f, indent=2)
    
    # 3. Save test data samples
    print("Saving test data samples...")
    metadata_dir = os.path.join(archive_dir, "metadata")
    
    # Save sample test data (100 samples or less)
    sample_size = min(100, len(test_reviews))
    sample_data = []
    for i in range(sample_size):
        sample_data.append({
            "text": test_reviews[i],
            "metadata": test_metadata[i],
            "label": int(test_labels[i]),
            "prediction": int(y_pred[i])
        })
    
    with open(os.path.join(metadata_dir, "test_samples.json"), "w") as f:
        json.dump(sample_data, f, indent=2)
    
    # 4. Save predictions
    print("Saving predictions...")
    predictions_dir = os.path.join(archive_dir, "predictions")
    np.save(os.path.join(predictions_dir, "predictions.npy"), y_pred)
    np.save(os.path.join(predictions_dir, "labels.npy"), test_labels)
    
    # 5. Save evaluation results and config
    print("Saving evaluation metrics and configuration...")
    
    # Enhanced config with all paths and settings
    enhanced_config = {
        **config,
        "paths": {
            "model_path": os.path.abspath(model_path),
            "tfidf_path": os.path.abspath(tfidf_path),
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
        "model": "TF-IDF + SVM with Metadata Concatenation",
        "accuracy": eval_results.get("accuracy", 0) * 100,
        "f1_score": eval_results.get("f1", 0) * 100,
        "approach": "Feature concatenation (TF-IDF + metadata features)"
    }
    
    with open(os.path.join(archive_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Archive complete! All data saved to {archive_dir}")
    return archive_dir

def main():
    # Create archive directory
    archive_dir = create_archive_directory()
    
    # Load data from database with metadata
    print("Loading data from database...")
    (train_reviews, train_labels, train_metadata, 
     test_reviews, test_labels, test_metadata, all_genres) = load_data_from_database()
    
    # Clean text
    print("Cleaning text...")
    train_reviews = [clean_text(review) for review in train_reviews]
    test_reviews = [clean_text(review) for review in test_reviews]
    
    print(f"Training dataset size: {len(train_reviews)}")
    print(f"Testing dataset size: {len(test_reviews)}")
    
    # Create metadata features
    X_train_metadata, metadata_feature_names = create_metadata_features(train_metadata, all_genres)
    X_test_metadata, _ = create_metadata_features(test_metadata, all_genres)
    
    # Print metadata feature statistics
    print("\nMetadata Feature Statistics:")
    print(f"Total features: {X_train_metadata.shape[1]}")
    print(f"Genre features: {len(all_genres)}")
    print(f"Rating feature: 1")
    print(f"Vote count feature: 1")
    
    # Show the most common genres
    genre_counts = Counter()
    for meta in train_metadata:
        for genre in meta['genres']:
            genre_counts[genre] += 1
    
    print("\nMost common genres in training data:")
    for genre, count in genre_counts.most_common(5):
        print(f"  - {genre}: {count} reviews")
    
    # Generate TF-IDF features for text
    print("\nGenerating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(tqdm(train_reviews, desc="Vectorizing"))
    print(f"TF-IDF features shape: {X_train_tfidf.shape}")
    
    # Convert TF-IDF to dense for numpy concatenation (only for demonstration, in practice keep sparse)
    print("\nConcatenating TF-IDF and metadata features...")
    
    # Combine features by horizontal stacking (column-wise)
    X_train_combined = sparse.hstack([X_train_tfidf, X_train_metadata]).tocsr()
    print(f"Combined training features shape: {X_train_combined.shape}")
    
    # Create and train SVM model
    print("\nTraining SVM model with combined features...")
    start_time = time.time()
    
    svm = LinearSVC(C=1.0, verbose=1, max_iter=1000)
    svm.fit(X_train_combined, train_labels)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print("\nMaking predictions on test set...")
    start_time = time.time()
    
    # Generate TF-IDF features for test set
    X_test_tfidf = tfidf.transform(test_reviews)
    
    # Combine test features
    X_test_combined = sparse.hstack([X_test_tfidf, X_test_metadata]).tocsr()
    
    # Predict
    y_pred = svm.predict(X_test_combined)
    inference_time = (time.time() - start_time) / len(test_reviews)
    
    # Compute evaluation metrics
    eval_results = compute_metrics(test_labels, y_pred)
    eval_results["training_time_seconds"] = training_time
    eval_results["inference_time_per_sample_ms"] = inference_time * 1000
    
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {eval_results['accuracy'] * 100:.2f}%")
    print(f"Precision: {eval_results['precision'] * 100:.2f}%")
    print(f"Recall: {eval_results['recall'] * 100:.2f}%")
    print(f"F1 Score: {eval_results['f1'] * 100:.2f}%")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Inference time per sample: {inference_time * 1000:.2f} ms")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, y_pred, target_names=["Negative", "Positive"]))
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    
    # Get SVM coefficients
    coef = svm.coef_[0]
    
    # Analyze TF-IDF feature importance
    tfidf_feature_names = np.array(tfidf.get_feature_names_out())
    tfidf_indices = np.argsort(coef[:len(tfidf_feature_names)])
    
    print("\nTop negative TF-IDF features:")
    for idx in tfidf_indices[:5]:
        print(f"  - {tfidf_feature_names[idx]}: {coef[idx]:.4f}")
    
    print("\nTop positive TF-IDF features:")
    for idx in tfidf_indices[-5:]:
        print(f"  - {tfidf_feature_names[idx]}: {coef[idx]:.4f}")
    
    # Analyze metadata feature importance
    metadata_start_idx = len(tfidf_feature_names)
    metadata_coef = coef[metadata_start_idx:]
    metadata_indices = np.argsort(metadata_coef)
    
    print("\nMetadata feature importance:")
    print("\nMost negative metadata features:")
    for idx in metadata_indices[:5]:
        if idx < len(metadata_feature_names):
            print(f"  - {metadata_feature_names[idx]}: {metadata_coef[idx]:.4f}")
    
    print("\nMost positive metadata features:")
    for idx in metadata_indices[-5:]:
        if idx < len(metadata_feature_names):
            print(f"  - {metadata_feature_names[idx]}: {metadata_coef[idx]:.4f}")
    
    # Create comprehensive model archive - configuration
    config = {
        "tfidf_params": {
            "max_features": 10000,
            "ngram_range": "1-2"
        },
        "svm_params": {
            "kernel": "linear",
            "C": 1.0
        },
        "metadata_integration": "feature concatenation",
        "num_metadata_features": len(metadata_feature_names),
        "experiment_date": datetime.now().strftime("%Y-%m-%d"),
        "experiment_time": datetime.now().strftime("%H:%M:%S"),
        "description": "IMDB sentiment analysis using TF-IDF + SVM with metadata feature concatenation"
    }
    
    # Finalize archive
    finalize_archive(
        model=svm,
        tfidf_model=tfidf,
        metadata_feature_names=metadata_feature_names,
        test_reviews=test_reviews,
        test_labels=test_labels,
        test_metadata=test_metadata,
        y_pred=y_pred,
        eval_results=eval_results,
        config=config,
        archive_dir=archive_dir
    )
    
    print(f"Complete experiment archive created at {archive_dir}")

if __name__ == "__main__":
    main()