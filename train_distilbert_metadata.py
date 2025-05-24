import os
import re
import json
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Constants
DB_PATH = "imdb_reviews.db"
MAX_LENGTH = 512

class IMDBMetadataDataset(Dataset):
    def __init__(self, reviews, labels, metadata, tokenizer, max_length):
        self.reviews = reviews
        self.labels = labels
        self.metadata = metadata  # Dictionary with genres and ratings
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        
        # Extract metadata and add to review text
        metadata_str = ""
        
        # Add genres if available
        if self.metadata[idx].get('genres'):
            genres = ", ".join(self.metadata[idx]['genres'])
            metadata_str += f" [Genres: {genres}]"
            
        # Add ratings if available
        if self.metadata[idx].get('rating'):
            metadata_str += f" [Rating: {self.metadata[idx]['rating']}]"
            
        if self.metadata[idx].get('votes'):
            metadata_str += f" [Votes: {self.metadata[idx]['votes']}]"
            
        # Combine review with metadata
        enhanced_review = review + metadata_str
        
        encoding = self.tokenizer(
            enhanced_review,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data_from_database():
    """Load reviews, labels, and metadata from SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query to get all necessary fields
    cursor.execute("""
        SELECT 
            user_review_text, 
            CASE WHEN user_review_sentiment = 'pos' THEN 1 ELSE 0 END as sentiment, 
            movie_genre,
            movie_average_rating,
            rating_vote_count,
            dataset_split
        FROM reviews
    """)
    
    # Process results
    train_reviews = []
    train_labels = []
    train_metadata = []
    
    test_reviews = []
    test_labels = []
    test_metadata = []
    
    for row in cursor.fetchall():
        review_text = row[0]
        sentiment = row[1]
        genre_json = row[2]
        rating = row[3]
        votes = row[4]
        split = row[5]
        
        # Skip rows with missing data
        if not review_text:
            continue
        
        # Parse genres from JSON
        genres = []
        if genre_json:
            try:
                genres = json.loads(genre_json)
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Create metadata dictionary
        metadata = {
            'genres': genres,
            'rating': rating,
            'votes': votes
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
    
    return train_reviews, train_labels, train_metadata, test_reviews, test_labels, test_metadata

def clean_text(text):
    """Clean the text by removing HTML tags and normalizing whitespace"""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
    return text.strip()

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate accuracy
    acc = accuracy_score(labels, preds)
    
    # Calculate precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will be on CPU, which will be much slower.")
        print("If you have a GPU, make sure CUDA drivers are properly installed.")
    
    # Load data from database
    print("Loading data from database...")
    train_reviews, train_labels, train_metadata, test_reviews, test_labels, test_metadata = load_data_from_database()
    
    # Clean text
    print("Cleaning text...")
    train_reviews = [clean_text(review) for review in train_reviews]
    test_reviews = [clean_text(review) for review in test_reviews]
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = IMDBMetadataDataset(
        train_reviews, 
        train_labels, 
        train_metadata, 
        tokenizer, 
        MAX_LENGTH
    )
    
    test_dataset = IMDBMetadataDataset(
        test_reviews, 
        test_labels, 
        test_metadata, 
        tokenizer, 
        MAX_LENGTH
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Training arguments - enable CUDA if available
    training_args = TrainingArguments(
        output_dir='./metadata-imdb-results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=500,
        save_steps=1500,
        eval_steps=1500,
        do_eval=True,
        no_cuda=False,  # Set to False to use GPU
        dataloader_pin_memory=True  # Enable pin memory for GPU
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    gpu_info = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Starting training on {gpu_info}...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    
    print(f"Evaluation Results: {eval_results}")
    print(f"Final accuracy: {eval_results['eval_accuracy'] * 100:.2f}%")
    print(f"Precision: {eval_results['eval_precision'] * 100:.2f}%")
    print(f"Recall: {eval_results['eval_recall'] * 100:.2f}%")
    print(f"F1 Score: {eval_results['eval_f1'] * 100:.2f}%")
    
    # Save the model
    model.save_pretrained("./distilbert-metadata-imdb-final")
    tokenizer.save_pretrained("./distilbert-metadata-imdb-final")

if __name__ == "__main__":
    main()