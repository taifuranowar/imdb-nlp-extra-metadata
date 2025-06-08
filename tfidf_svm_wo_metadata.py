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

# Constants
DB_PATH = "imdb_reviews.db"

def create_archive_directory(base_dir="./training/tfidf-svm-imdb-archive"):
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
    """Load reviews, labels, and metadata from SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query to get all necessary fields
    cursor.execute("""
        SELECT 
            user_review_text, 
            CASE WHEN user_review_sentiment = 'pos' THEN 1 ELSE 0 END as sentiment, 
            dataset_split
        FROM reviews
    """)
    
    # Process results
    train_reviews = []
    train_labels = []
    test_reviews = []
    test_labels = []
    
    for row in cursor.fetchall():
        review_text = row[0]
        sentiment = row[1]
        split = row[2]
        
        # Skip rows with missing data
        if not review_text:
            continue
        
        # Add to appropriate split
        if split == 'train':
            train_reviews.append(review_text)
            train_labels.append(sentiment)
        elif split == 'test':
            test_reviews.append(review_text)
            test_labels.append(sentiment)
    
    conn.close()
    
    return train_reviews, train_labels, test_reviews, test_labels

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

def finalize_archive(model, test_reviews, test_labels, y_pred, eval_results, config, archive_dir):
    """Save model, predictions, and evaluation results"""
    print(f"Finalizing experiment archive in {archive_dir}...")
    
    # 1. Save the model
    print("Saving model...")
    model_path = os.path.join(archive_dir, "model", "tfidf_svm_model.pkl")
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
    
    # 4. Save evaluation results and config
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
    
    # 5. Save experiment summary
    summary = {
        "experiment_id": os.path.basename(archive_dir),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "TF-IDF + SVM",
        "accuracy": eval_results.get("accuracy", 0) * 100,
        "f1_score": eval_results.get("f1", 0) * 100,
    }
    
    with open(os.path.join(archive_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Archive complete! All data saved to {archive_dir}")
    return archive_dir

def main():
    # Create archive directory
    archive_dir = create_archive_directory()
    
    # Load data from database
    print("Loading data from database...")
    train_reviews, train_labels, test_reviews, test_labels = load_data_from_database()
    
    # Clean text
    print("Cleaning text...")
    train_reviews = [clean_text(review) for review in train_reviews]
    test_reviews = [clean_text(review) for review in test_reviews]
    
    print(f"Training dataset size: {len(train_reviews)}")
    print(f"Testing dataset size: {len(test_reviews)}")
    
    # Create and train TF-IDF + SVM pipeline
    print("Training TF-IDF + SVM model...")
    start_time = time.time()

    # Create TF-IDF features with progress reporting
    print("  Generating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(tqdm(train_reviews, desc="Vectorizing"))

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
    y_pred = pipeline.predict(test_reviews)
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
            "max_features": 10000,
            "ngram_range": "1-2"
        },
        "svm_params": {
            "kernel": "linear",
            "C": 1.0
        },
        "experiment_date": datetime.now().strftime("%Y-%m-%d"),
        "experiment_time": datetime.now().strftime("%H:%M:%S"),
        "description": "IMDB sentiment analysis using TF-IDF + SVM"
    }
    
    # Finalize archive
    finalize_archive(
        model=pipeline,
        test_reviews=test_reviews,
        test_labels=test_labels,
        y_pred=y_pred,
        eval_results=eval_results,
        config=config,
        archive_dir=archive_dir
    )
    
    print(f"Complete experiment archive created at {archive_dir}")

if __name__ == "__main__":
    main()