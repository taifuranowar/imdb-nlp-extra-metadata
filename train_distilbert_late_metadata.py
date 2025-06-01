import os
import re
import json
import sqlite3
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertModel,
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime

# Constants
DB_PATH = "imdb_reviews.db"
MAX_LENGTH = 512

class LateFusionDistilBertModel(nn.Module):
    """DistilBERT with late fusion for metadata integration"""
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2, metadata_dim=10):
        super().__init__()
        
        # Text encoder (DistilBERT)
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.text_dim = self.distilbert.config.hidden_size  # 768 for DistilBERT
        
        # Metadata encoder
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer
        fusion_dim = self.text_dim + 32  # Text features + encoded metadata
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification head
        self.classifier = nn.Linear(64, num_labels)
        
    def forward(self, input_ids, attention_mask, metadata_features, labels=None):
        # Process text with DistilBERT
        text_outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Process metadata
        metadata_encoded = self.metadata_encoder(metadata_features)  # [batch_size, 32]
        
        # Late fusion: concatenate text and metadata features
        fused_features = torch.cat([text_features, metadata_encoded], dim=1)
        
        # Final classification
        fused_encoded = self.fusion_layer(fused_features)
        logits = self.classifier(fused_encoded)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
    
        # Use proper SequenceClassifierOutput instead of custom type
        from transformers.modeling_outputs import SequenceClassifierOutput
    
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

class IMDBLateFusionDataset(Dataset):
    def __init__(self, reviews, labels, metadata, tokenizer, max_length):
        self.reviews = reviews
        self.labels = labels
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def _encode_metadata(self, metadata_dict):
        """Convert metadata to numerical features"""
        features = np.zeros(10)  # Fixed size feature vector
        
        # Genre encoding (one-hot for top genres)
        top_genres = ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller']
        if metadata_dict.get('genres'):
            for i, genre in enumerate(top_genres):
                if genre in metadata_dict['genres']:
                    features[i] = 1.0
    
        # Rating (normalized)
        rating_value = 0.0
        if metadata_dict.get('rating'):
            try:
                rating_value = float(metadata_dict['rating'])
                features[5] = rating_value / 10.0  # Normalize to 0-1
            except (ValueError, TypeError):
                features[5] = 0.5  # Default middle value
    
        # Votes (log-normalized)
        votes_value = 0.0
        if metadata_dict.get('votes'):
            try:
                votes_value = float(metadata_dict['votes'])
                features[6] = min(np.log10(votes_value + 1) / 6.0, 1.0)  # Log normalize and cap at 1
            except (ValueError, TypeError):
                features[6] = 0.0
    
        # Additional features (can be expanded)
        #features[7] = len(metadata_dict.get('genres', [])) / 5.0  # Number of genres normalized
        #features[8] = 1.0 if rating_value > 7.0 else 0.0  # High rating flag - use converted rating_value
        #features[9] = 1.0 if votes_value > 10000 else 0.0  # Popular movie flag - use converted votes_value
    
        return features
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        
        # Tokenize only the text (no metadata concatenation)
        encoding = self.tokenizer(
            review,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Encode metadata separately
        metadata_features = self._encode_metadata(self.metadata[idx])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata_features': torch.tensor(metadata_features, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class LateFusionTrainer(Trainer):
    """Custom trainer for late fusion model"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"), 
            metadata_features=inputs.get("metadata_features"),
            labels=labels
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def create_archive_directory(base_dir="./training/distilbert-late-fusion-metadata-imdb-archive"):
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

def finalize_archive(model, tokenizer, test_dataset, training_args, eval_results, config, archive_dir):
    """Finalize the archive with evaluation results and model artifacts"""
    print(f"Finalizing experiment archive in {archive_dir}...")
    
    # 1. Save model & tokenizer
    print("Saving model and tokenizer...")
    model_dir = os.path.join(archive_dir, "model")
    tokenizer_dir = os.path.join(archive_dir, "tokenizer")
    
    # Save the custom model
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    
    # Save model config
    model_config = {
        "model_type": "late_fusion_distilbert",
        "text_dim": model.text_dim,
        "metadata_dim": 10,
        "num_labels": 2
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    
    tokenizer.save_pretrained(tokenizer_dir)
    
    # 2. Save test data structure with metadata
    print("Saving test data structure...")
    metadata_dir = os.path.join(archive_dir, "metadata")
    
    # Save metadata format and config
    with open(os.path.join(metadata_dir, "metadata_config.json"), "w") as f:
        json.dump({
            "fusion_type": config.get("fusion_technique", "late_fusion"),
            "metadata_fields": config.get("metadata", {}),
            "metadata_encoding": "numerical_features"
        }, f, indent=2)
    
    # Save sample test data (100 samples or less)
    sample_size = min(100, len(test_dataset))
    sample_data = []
    for i in range(sample_size):
        sample_data.append({
            "text": test_dataset.reviews[i] if hasattr(test_dataset, "reviews") else "",
            "label": int(test_dataset.labels[i]),
            "metadata": test_dataset.metadata[i] if hasattr(test_dataset, "metadata") else {},
            "metadata_features": test_dataset._encode_metadata(test_dataset.metadata[i]).tolist()
        })
    
    with open(os.path.join(metadata_dir, "test_samples.json"), "w") as f:
        json.dump(sample_data, f, indent=2)
        
    # 3. Save model predictions on test set
    print("Saving model predictions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    all_preds = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch["labels"]
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.append(preds.cpu().numpy())
            all_logits.append(outputs.logits.cpu().numpy())
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
        "model": config.get("model_name", "distilbert-base-uncased"),
        "fusion_technique": config.get("fusion_technique", "late_fusion"),
        "metadata_fields": config.get("metadata", {}),
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
        print("If you have a GPU, make sure CUDA drivers are properly installed.")
    
    # Create archive directory at the beginning
    archive_dir = create_archive_directory()
    
    # Load data from database
    print("Loading data from database...")
    train_reviews, train_labels, train_metadata, test_reviews, test_labels, test_metadata = load_data_from_database()
    
    # Clean text
    print("Cleaning text...")
    train_reviews = [clean_text(review) for review in train_reviews]
    test_reviews = [clean_text(review) for review in test_reviews]
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create late fusion model
    print("Creating late fusion model...")
    model = LateFusionDistilBertModel(
        model_name='distilbert-base-uncased',
        num_labels=2,
        metadata_dim=10
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = IMDBLateFusionDataset(
        train_reviews, 
        train_labels, 
        train_metadata, 
        tokenizer, 
        MAX_LENGTH
    )
    
    test_dataset = IMDBLateFusionDataset(
        test_reviews, 
        test_labels, 
        test_metadata, 
        tokenizer, 
        MAX_LENGTH
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Training arguments - SAME AS EARLY FUSION
    training_args = TrainingArguments(
        output_dir=os.path.join(archive_dir, "training_outputs"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_dir=os.path.join(archive_dir, "logs"),
        logging_steps=500,
        save_steps=1500,
        eval_steps=1500,
        do_eval=True,
        no_cuda=False,  # Set to False to use GPU
        dataloader_pin_memory=True  # Enable pin memory for GPU
    )
    
    # Create custom trainer
    trainer = LateFusionTrainer(
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
    
    # Create comprehensive model archive - comprehensive configuration
    config = {
        "fusion_technique": "late_fusion",
        "metadata": {
            "genres": True,
            "ratings": True,
            "votes": True
        },
        "model_name": "distilbert-base-uncased",
        "fusion_architecture": {
            "text_encoder": "distilbert-base-uncased",
            "metadata_encoder": "2-layer MLP (64->32)",
            "fusion_layer": "2-layer MLP (800->256->64)",
            "classifier": "linear (64->2)"
        },
        "epochs": 3,
        "batch_size": 16,
        "max_length": MAX_LENGTH,
        "experiment_date": datetime.now().strftime("%Y-%m-%d"),
        "experiment_time": datetime.now().strftime("%H:%M:%S"),
        "description": "IMDB sentiment analysis with metadata using late fusion technique"
    }
    
    # Finalize archive with evaluation results and predictions
    finalize_archive(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        training_args=training_args,
        eval_results=eval_results,
        config=config,
        archive_dir=archive_dir
    )
    
    print(f"Complete experiment archive created at {archive_dir}")


if __name__ == "__main__":
    main()