import os
import re
import json
import sqlite3
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    RobertaConfig,
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
from collections import Counter, defaultdict

# Constants
DB_PATH = "imdb_reviews.db"
MAX_LENGTH = 512

def get_all_genres_from_database():
    """Extract all unique genres from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query to get all genre data
    cursor.execute("SELECT movie_genre FROM reviews WHERE movie_genre IS NOT NULL")
    
    # Process results
    genre_counter = Counter()
    
    for (genre_json,) in cursor.fetchall():
        if genre_json:
            try:
                genres = json.loads(genre_json)
                for genre in genres:
                    genre_counter[genre] += 1
            except (json.JSONDecodeError, TypeError):
                pass
    
    conn.close()
    
    # Get the most common genres (e.g., top 20)
    top_genres = [genre for genre, _ in genre_counter.most_common(20)]
    print(f"Found {len(top_genres)} most common genres: {', '.join(top_genres)}")
    
    return top_genres

def calculate_genre_sentiment_stats(reviews, labels, metadata, common_genres):
    """Calculate genre-specific sentiment statistics from the training data"""
    print("Calculating genre-specific sentiment statistics...")
    
    # Initialize counters
    genre_pos_counts = defaultdict(int)
    genre_total_counts = defaultdict(int)
    
    # Count occurrences
    for i, meta in enumerate(metadata):
        if meta.get('genres'):
            label = labels[i]
            for genre in meta['genres']:
                if genre in common_genres:
                    genre_total_counts[genre] += 1
                    if label == 1:  # Positive sentiment
                        genre_pos_counts[genre] += 1
    
    # Calculate positive ratios for each genre
    genre_sentiment_stats = {}
    for genre in common_genres:
        total = genre_total_counts.get(genre, 0)
        if total > 0:
            pos_ratio = genre_pos_counts.get(genre, 0) / total
        else:
            pos_ratio = 0.5  # Default for no data
        genre_sentiment_stats[genre] = pos_ratio
    
    # Print statistics
    print("Genre sentiment statistics:")
    for genre, ratio in sorted(genre_sentiment_stats.items(), key=lambda x: x[1]):
        print(f"  {genre:15s}: {ratio:.3f} ({genre_pos_counts.get(genre, 0)}/{genre_total_counts.get(genre, 0)} positive)")
    
    return genre_sentiment_stats

class IMDBIntegratedDataset(Dataset):
    def __init__(self, reviews, labels, metadata, tokenizer, max_length, common_genres, genre_sentiment_stats=None):
        self.reviews = reviews
        self.labels = labels
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.common_genres = common_genres
        self.genre_sentiment_stats = genre_sentiment_stats or {}
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        
        # Process review text
        text_encoding = self.tokenizer(
            review,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process metadata as separate features
        metadata_features = []
        
        # Add rating (normalized)
        rating = self.metadata[idx].get('rating', 0)
        if rating:
            normalized_rating = float(rating) / 10.0
        else:
            normalized_rating = 0.0
        metadata_features.append(normalized_rating)
        
        # Add vote count (log-normalized)
        votes = self.metadata[idx].get('votes', 0)
        if votes:
            log_votes = np.log1p(float(votes)) / 15.0
        else:
            log_votes = 0.0
        metadata_features.append(log_votes)
        
        # Create genre features with sentiment context
        movie_genres = self.metadata[idx].get('genres', [])
        
        # For each genre, add both presence and sentiment statistics
        for genre in self.common_genres:
            # Genre presence (one-hot)
            is_present = 1 if genre in movie_genres else 0
            metadata_features.append(is_present)
            
            # Add genre sentiment statistics if available
            if is_present and genre in self.genre_sentiment_stats:
                genre_sent_ratio = self.genre_sentiment_stats[genre]
            else:
                genre_sent_ratio = 0.5  # Neutral for absent genres
            metadata_features.append(genre_sent_ratio)
        
        return {
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'metadata_features': torch.tensor(metadata_features, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class RobertaIntegratedMetadata(nn.Module):
    def __init__(self, num_labels=2, metadata_size=22):
        super().__init__()
        
        # Load RoBERTa for text processing
        self.config = RobertaConfig.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        # Metadata processing (smaller network since metadata will be injected into classifier)
        self.metadata_projection = nn.Linear(metadata_size, 64)
        
        # Modified classifier - takes both text and metadata features
        self.pre_classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(0.1)  # RoBERTa default dropout
        
        # New classifier that integrates metadata at the classification level
        self.integrated_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size + 64, self.config.hidden_size),
            nn.GELU(), # RoBERTa uses GELU activation
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, metadata_features, labels=None):
        # Process text through RoBERTa (fine-tune the whole model)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Process text through standard parts
        pooled_output = self.pre_classifier(hidden_state)
        pooled_output = nn.GELU()(pooled_output)  # RoBERTa uses GELU
        pooled_output = self.dropout(pooled_output)
        
        # Process metadata
        metadata_projected = nn.GELU()(self.metadata_projection(metadata_features))
        
        # Integrate metadata directly into the classifier decision
        combined = torch.cat([pooled_output, metadata_projected], dim=1)
        logits = self.integrated_classifier(combined)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return (loss, logits) if loss is not None else logits

class IntegratedMetadataTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract labels (expected to be "labels" from our dataset)
        if 'label' in inputs:
            labels = inputs.pop("label")
        elif 'labels' in inputs:
            labels = inputs.pop("labels")
        else:
            print("Available keys in inputs:", list(inputs.keys()))
            raise KeyError("Neither 'label' nor 'labels' found in inputs")
            
        outputs = model(**inputs)
        
        if isinstance(outputs, tuple):
            loss = outputs[0]
            logits = outputs[1]
        else:
            logits = outputs
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return (loss, {"logits": logits}) if return_outputs else loss

def create_archive_directory(base_dir="./training/roberta-update-classifier-head-metadata-imdb-archive"):
    """Create timestamped archive directory at the beginning of the experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive_dir = f"{base_dir}_{timestamp}"
    
    # Create the training directory if it doesn't exist
    os.makedirs(os.path.dirname(archive_dir), exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    # Create subdirectories for organization
    os.makedirs(os.path.join(archive_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(archive_dir, "tokenizer"), exist_ok=True)
    os.makedirs(os.path.join(archive_dir, "training_outputs"), exist_ok=True)
    os.makedirs(os.path.join(archive_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(archive_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(archive_dir, "predictions"), exist_ok=True)

    print(f"Created archive directory: {archive_dir}")
    return archive_dir

def finalize_archive(model, tokenizer, test_dataset, training_args, eval_results, config, archive_dir, genre_sentiment_stats=None):
    """Finalize the archive with evaluation results and model artifacts"""
    print(f"Finalizing experiment archive in {archive_dir}...")
    
    # 1. Save model & tokenizer
    print("Saving model and tokenizer...")
    model_dir = os.path.join(archive_dir, "model")
    tokenizer_dir = os.path.join(archive_dir, "tokenizer")
    
    # Custom model needs special handling
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({
            "model_type": "roberta-integrated-metadata",
            "num_labels": 2,
            "metadata_size": config.get("metadata_size", 22)
        }, f, indent=2)
    
    tokenizer.save_pretrained(tokenizer_dir)
    
    # 2. Save test data structure with metadata
    print("Saving test data structure...")
    metadata_dir = os.path.join(archive_dir, "metadata")
    
    # Save metadata format and config
    with open(os.path.join(metadata_dir, "metadata_config.json"), "w") as f:
        json.dump({
            "fusion_technique": config.get("fusion_technique", "updated_classifier_head"),
            "metadata_fields": config.get("metadata", {}),
            "genre_sentiment_embedding": True
        }, f, indent=2)
    
    # Save genre sentiment statistics if available
    if genre_sentiment_stats:
        with open(os.path.join(metadata_dir, "genre_sentiment_stats.json"), "w") as f:
            json.dump(genre_sentiment_stats, f, indent=2)
    
    # Save sample test data
    sample_size = min(100, len(test_dataset))
    sample_data = []
    for i in range(sample_size):
        sample_data.append({
            "text": test_dataset.reviews[i] if hasattr(test_dataset, "reviews") else "",
            "label": int(test_dataset.labels[i]),
            "metadata": test_dataset.metadata[i] if hasattr(test_dataset, "metadata") else {}
        })
    
    with open(os.path.join(metadata_dir, "test_samples.json"), "w") as f:
        json.dump(sample_data, f, indent=2)
        
    # 3. Save model predictions on test set
    print("Saving model predictions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=8)  # Smaller batch size for RoBERTa
    
    all_preds = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'metadata_features': batch['metadata_features'].to(device)
            }
            labels = batch["labels"]  # Note: Using "labels" key here
            
            outputs = model(**inputs)
            
            if isinstance(outputs, tuple):
                logits = outputs[1]
            else:
                logits = outputs
                
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.append(preds.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    
    # Save predictions
    predictions_dir = os.path.join(archive_dir, "predictions")
    np.save(os.path.join(predictions_dir, "predictions.npy"), all_preds)
    np.save(os.path.join(predictions_dir, "logits.npy"), all_logits)
    np.save(os.path.join(predictions_dir, "labels.npy"), all_labels)
    
    # 4. Save evaluation metrics and comprehensive config
    print("Saving evaluation metrics and configuration...")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Measure inference time for a single sample
    sample_input = test_dataset[0]
    inputs = {
        'input_ids': sample_input['input_ids'].unsqueeze(0).to(device),
        'attention_mask': sample_input['attention_mask'].unsqueeze(0).to(device),
        'metadata_features': sample_input['metadata_features'].unsqueeze(0).to(device)
    }
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(**inputs)
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # Multiple iterations for accuracy
            _ = model(**inputs)
    inference_time = (time.time() - start_time) / 100 * 1000  # ms
    
    # Enhanced config with all paths and settings
    enhanced_config = {
        **config,
        "paths": {
            "model_dir": os.path.abspath(model_dir),
            "tokenizer_dir": os.path.abspath(tokenizer_dir),
            "training_output_dir": os.path.abspath(training_args.output_dir),
            "log_dir": os.path.abspath(training_args.logging_dir),
            "predictions_dir": os.path.abspath(predictions_dir)
        },
        "training": training_args.to_dict(),
        "system_info": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    }
    
    # Save the config
    with open(os.path.join(archive_dir, "experiment_config.json"), "w") as f:
        json.dump(enhanced_config, f, indent=2)
    
    # Add additional metrics to eval_results
    complete_results = {
        **eval_results,
        "parameter_count": int(total_params),
        "trainable_parameter_count": int(trainable_params),
        "inference_time_ms": float(inference_time),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device)
    }
    
    with open(os.path.join(archive_dir, "evaluation_results.json"), "w") as f:
        json.dump(complete_results, f, indent=2)
    
    # 5. Save experiment summary
    summary = {
        "experiment_id": os.path.basename(archive_dir),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": config.get("model_name", "roberta-base"),
        "fusion_technique": config.get("fusion_technique", "update classifier head with metadata"),
        "metadata_fields": config.get("metadata", {}),
        "genre_sentiment_embedding": True,
        "accuracy": eval_results.get("eval_accuracy", 0) * 100,
        "f1_score": eval_results.get("eval_f1", 0) * 100,
        "parameter_count": int(total_params),
        "inference_time_ms": float(inference_time),
    }
    
    with open(os.path.join(archive_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Archive complete! All data saved to {archive_dir}")
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
    
    # Create archive directory at the beginning
    archive_dir = create_archive_directory()
    
    # Get common genres first
    print("Extracting common genres from the database...")
    common_genres = get_all_genres_from_database()
    
    # Save the determined genres to the archive
    os.makedirs(os.path.join(archive_dir, "metadata"), exist_ok=True)
    with open(os.path.join(archive_dir, "metadata", "common_genres.json"), "w") as f:
        json.dump(common_genres, f, indent=2)
    
    # Load data from database
    print("Loading data from database...")
    train_reviews, train_labels, train_metadata, test_reviews, test_labels, test_metadata = load_data_from_database()
    
    # Clean text
    print("Cleaning text...")
    train_reviews = [clean_text(review) for review in train_reviews]
    test_reviews = [clean_text(review) for review in test_reviews]
    
    # Calculate genre sentiment statistics
    print("Calculating genre sentiment statistics...")
    genre_sentiment_stats = calculate_genre_sentiment_stats(
        train_reviews, train_labels, train_metadata, common_genres
    )
    
    # Save genre sentiment statistics to the archive
    with open(os.path.join(archive_dir, "metadata", "genre_sentiment_stats.json"), "w") as f:
        json.dump(genre_sentiment_stats, f, indent=2)
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Calculate metadata size: 2 base features (rating, votes) + 2*genres (presence + sentiment for each)
    metadata_size = 2 + (len(common_genres) * 2)
    
    # Create custom model for integrated metadata fusion
    model = RobertaIntegratedMetadata(num_labels=2, metadata_size=metadata_size)
    print(f"Created integrated metadata model with size: {metadata_size} (2 base features + {len(common_genres)*2} genre features)")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = IMDBIntegratedDataset(
        train_reviews, 
        train_labels, 
        train_metadata, 
        tokenizer, 
        MAX_LENGTH,
        common_genres,
        genre_sentiment_stats
    )
    
    test_dataset = IMDBIntegratedDataset(
        test_reviews, 
        test_labels, 
        test_metadata, 
        tokenizer, 
        MAX_LENGTH,
        common_genres,
        genre_sentiment_stats
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Training arguments - using archive directory paths
    # RoBERTa is larger than DistilBERT, so we might need smaller batch sizes
    training_args = TrainingArguments(
        output_dir=os.path.join(archive_dir, "training_outputs"),
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Reduced for RoBERTa
        per_device_eval_batch_size=8,   # Reduced for RoBERTa
        gradient_accumulation_steps=2,  # Add gradient accumulation to maintain effective batch size
        weight_decay=0.01,
        logging_dir=os.path.join(archive_dir, "logs"),
        logging_steps=500,
        save_steps=1500,
        eval_steps=1500,
        do_eval=True,
        no_cuda=False,  # Set to False to use GPU
        dataloader_pin_memory=True,  # Enable pin memory for GPU
        fp16=True  # Enable mixed precision training for faster execution
    )
    
    # Create custom trainer
    trainer = IntegratedMetadataTrainer(
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
    
    # Create comprehensive model archive
    config = {
        "fusion_technique": "updated_classifier_head_with_metadata",
        "metadata": {
            "genres": True,
            "genre_list": common_genres,
            "genre_sentiment_stats": True,
            "ratings": True,
            "votes": True
        },
        "model_name": "roberta-base",
        "epochs": 3,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "max_length": MAX_LENGTH,
        "metadata_size": metadata_size,
        "experiment_date": datetime.now().strftime("%Y-%m-%d"),
        "experiment_time": datetime.now().strftime("%H:%M:%S"),
        "description": "IMDB sentiment analysis with genre sentiment embeddings integrated into RoBERTa classifier"
    }
    
    # Finalize archive with evaluation results and predictions
    finalize_archive(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        training_args=training_args,
        eval_results=eval_results,
        config=config,
        archive_dir=archive_dir,
        genre_sentiment_stats=genre_sentiment_stats
    )
    
    print(f"Complete experiment archive created at {archive_dir}")


if __name__ == "__main__":
    main()