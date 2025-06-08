import os
import re
import json
import sqlite3
import numpy as np
import time
import pickle
from datetime import datetime
import multiprocessing
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Constants
DB_PATH = "imdb_reviews.db"
WORD2VEC_SIZE = 100  # Word vector dimensionality
WORD2VEC_WINDOW = 5  # Context window size
WORD2VEC_MIN_COUNT = 5  # Minimum word count
WORD2VEC_WORKERS = multiprocessing.cpu_count()  # Use all available CPUs

# Make sure NLTK data directory exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Download necessary NLTK data
print("Setting up NLTK resources...")

# Try to download all potentially needed resources
nltk_resources = ['punkt', 'punkt_tab', 'tokenizers/punkt']

for resource in nltk_resources:
    try:
        print(f"Downloading NLTK resource: {resource}")
        nltk.download(resource, quiet=False)
    except Exception as e:
        print(f"Error downloading {resource}: {e}")

# Alternative: Implement a simple tokenizer that doesn't rely on NLTK
def simple_tokenize(text):
    """Simple tokenizer that splits on whitespace and removes punctuation"""
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation with spaces
    for char in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~':
        text = text.replace(char, ' ')
    # Split on whitespace and filter out empty strings
    return [token for token in text.split() if token]

# Modify the tokenize_reviews function to use the simple tokenizer as fallback
def tokenize_reviews(reviews):
    """Tokenize reviews into words with fallback to simple tokenizer"""
    print("Tokenizing reviews...")
    tokenized_reviews = []
    use_simple_tokenizer = False
    
    for i, review in enumerate(tqdm(reviews, desc="Tokenizing")):
        try:
            if use_simple_tokenizer:
                # Use simple tokenizer
                tokens = simple_tokenize(review)
            else:
                # Try NLTK tokenizer
                tokens = word_tokenize(review.lower())
                # Keep only alphabetic tokens
                tokens = [word for word in tokens if word.isalpha()]
        except LookupError:
            print("NLTK tokenizer failed. Switching to simple tokenizer...")
            use_simple_tokenizer = True
            tokens = simple_tokenize(review)
        
        tokenized_reviews.append(tokens)
        
        # Print sample of first tokenized review
        if i == 0:
            print(f"Sample tokenization (first 10 tokens): {tokens[:10]}")
    
    return tokenized_reviews

def create_archive_directory(base_dir="./training/word2vec-svm-imdb-archive"):
    """Create timestamped archive directory at the beginning of the experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive_dir = f"{base_dir}_{timestamp}"
    
    # Create the training directory if it doesn't exist
    os.makedirs(os.path.dirname(archive_dir), exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    # Create subdirectories for organization
    os.makedirs(os.path.join(archive_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(archive_dir, "word2vec"), exist_ok=True)
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

def document_vectors(word2vec_model, tokenized_docs):
    """Convert documents to vectors by averaging word vectors"""
    doc_vectors = []
    vector_size = word2vec_model.vector_size
    
    for doc in tqdm(tokenized_docs, desc="Creating document vectors"):
        # Filter out words not in vocabulary
        doc_words = [word for word in doc if word in word2vec_model.wv.key_to_index]
        
        if len(doc_words) > 0:
            # Average word vectors
            doc_vector = np.mean([word2vec_model.wv[word] for word in doc_words], axis=0)
        else:
            # If no words in vocabulary, use zero vector
            doc_vector = np.zeros(vector_size)
        
        doc_vectors.append(doc_vector)
    
    return np.array(doc_vectors)

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

def finalize_archive(word2vec_model, svm_model, test_reviews, test_labels, y_pred, eval_results, config, archive_dir):
    """Save models, predictions, and evaluation results"""
    print(f"Finalizing experiment archive in {archive_dir}...")
    
    # 1. Save the SVM model
    print("Saving SVM model...")
    svm_path = os.path.join(archive_dir, "model", "word2vec_svm_model.pkl")
    with open(svm_path, 'wb') as f:
        pickle.dump(svm_model, f)
    
    # 2. Save the Word2Vec model
    print("Saving Word2Vec model...")
    w2v_path = os.path.join(archive_dir, "word2vec", "word2vec.model")
    word2vec_model.save(w2v_path)
    
    # 3. Save test data samples
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
            "svm_model_path": os.path.abspath(svm_path),
            "word2vec_model_path": os.path.abspath(w2v_path),
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
        "model": "Word2Vec + SVM",
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
    train_reviews = [clean_text(review) for review in tqdm(train_reviews, desc="Cleaning train")]
    test_reviews = [clean_text(review) for review in tqdm(test_reviews, desc="Cleaning test")]
    
    print(f"Training dataset size: {len(train_reviews)}")
    print(f"Testing dataset size: {len(test_reviews)}")
    
    # Tokenize reviews for Word2Vec
    print("Tokenizing reviews for Word2Vec...")
    train_tokenized = tokenize_reviews(train_reviews)
    test_tokenized = tokenize_reviews(test_reviews)
    
    # Train Word2Vec model
    print("Training Word2Vec model...")
    start_time = time.time()
    print(f"  Using {WORD2VEC_WORKERS} CPU cores for training")
    
    # Progress updates for Word2Vec training
    class ProgressCallback:
        def __init__(self, total_epochs=5):
            self.epoch = 0
            self.total_epochs = total_epochs
            self.start_time = time.time()
            
        def on_train_begin(self, model):
            print(f"  Training Word2Vec model with {self.total_epochs} epochs")
            
        def on_train_end(self, model):
            elapsed_time = time.time() - self.start_time
            print(f"  Training completed in {elapsed_time:.1f}s")
            
        def on_epoch_begin(self, model):
            pass
            
        def on_epoch_end(self, model):
            self.epoch += 1
            elapsed_time = time.time() - self.start_time
            remaining_time = (elapsed_time / self.epoch) * (self.total_epochs - self.epoch) if self.epoch < self.total_epochs else 0
            print(f"  Epoch {self.epoch}/{self.total_epochs} - Elapsed: {elapsed_time:.1f}s, ETA: {remaining_time:.1f}s")
    
    # Train Word2Vec model (with progress reporting)
    word2vec_model = Word2Vec(
        sentences=train_tokenized,
        vector_size=WORD2VEC_SIZE,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=WORD2VEC_WORKERS,
        callbacks=[ProgressCallback(5)]  # Default is 5 epochs
    )
    
    w2v_training_time = time.time() - start_time
    print(f"Word2Vec training completed in {w2v_training_time:.2f} seconds")
    print(f"Vocabulary size: {len(word2vec_model.wv.index_to_key)} words")
    
    # Create document vectors
    print("Creating document vectors...")
    start_time = time.time()
    X_train_vectors = document_vectors(word2vec_model, train_tokenized)
    X_test_vectors = document_vectors(word2vec_model, test_tokenized)
    
    doc_vec_time = time.time() - start_time
    print(f"Document vectorization completed in {doc_vec_time:.2f} seconds")
    
    # Train SVM classifier
    print("Training SVM classifier...")
    start_time = time.time()
    
    # We're using a simple LinearSVC
    svm = LinearSVC(C=1.0, max_iter=10000)
    
    # Time-based progress reporting for SVM training
    print("  SVM training started (will report progress every 10 seconds)...")
    
    # Use threading to report progress during SVM training
    import threading
    stop_progress = False
    
    def report_progress():
        start = time.time()
        while not stop_progress:
            elapsed = time.time() - start
            print(f"  SVM training in progress... (elapsed: {elapsed:.1f}s)")
            time.sleep(10)  # Update every 10 seconds
    
    # Start progress reporting thread
    progress_thread = threading.Thread(target=report_progress)
    progress_thread.daemon = True
    progress_thread.start()
    
    # Actual SVM training
    try:
        svm.fit(X_train_vectors, train_labels)
    finally:
        stop_progress = True
        progress_thread.join(timeout=1.0)
    
    svm_training_time = time.time() - start_time
    print(f"SVM training completed in {svm_training_time:.2f} seconds")
    
    # Make predictions
    print("Making predictions on test set...")
    start_time = time.time()
    y_pred = svm.predict(X_test_vectors)
    inference_time = (time.time() - start_time) / len(test_reviews)
    
    # Compute evaluation metrics
    eval_results = compute_metrics(test_labels, y_pred)
    eval_results["word2vec_training_time"] = w2v_training_time
    eval_results["svm_training_time"] = svm_training_time
    eval_results["total_training_time"] = w2v_training_time + svm_training_time + doc_vec_time
    eval_results["inference_time_per_sample_ms"] = inference_time * 1000
    
    print(f"Evaluation Results:")
    print(f"Accuracy: {eval_results['accuracy'] * 100:.2f}%")
    print(f"Precision: {eval_results['precision'] * 100:.2f}%")
    print(f"Recall: {eval_results['recall'] * 100:.2f}%")
    print(f"F1 Score: {eval_results['f1'] * 100:.2f}%")
    print(f"Word2Vec training time: {w2v_training_time:.2f} seconds")
    print(f"SVM training time: {svm_training_time:.2f} seconds")
    print(f"Total training time: {eval_results['total_training_time']:.2f} seconds")
    print(f"Inference time per sample: {inference_time * 1000:.2f} ms")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, y_pred, target_names=["Negative", "Positive"]))
    
    # Create comprehensive model archive - configuration
    config = {
        "word2vec_params": {
            "vector_size": WORD2VEC_SIZE,
            "window": WORD2VEC_WINDOW,
            "min_count": WORD2VEC_MIN_COUNT,
            "workers": WORD2VEC_WORKERS
        },
        "svm_params": {
            "C": 1.0,
            "max_iter": 10000
        },
        "experiment_date": datetime.now().strftime("%Y-%m-%d"),
        "experiment_time": datetime.now().strftime("%H:%M:%S"),
        "description": "IMDB sentiment analysis using Word2Vec + SVM"
    }
    
    # Finalize archive
    finalize_archive(
        word2vec_model=word2vec_model,
        svm_model=svm,
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