import os
import re
import tarfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score

# Constants
DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATASET_PATH = "./aclImdb"
MAX_LENGTH = 512

class IMDBDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            review,
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

def download_and_extract_dataset():
    """Download and extract the IMDB dataset if it doesn't exist"""
    if not os.path.exists(DATASET_PATH):
        print(f"Downloading IMDB dataset from {DATASET_URL}...")
        filename, _ = urllib.request.urlretrieve(DATASET_URL, "aclImdb_v1.tar.gz")
        
        print("Extracting dataset...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        
        os.remove(filename)
        print("Dataset downloaded and extracted successfully.")
    else:
        print("Dataset already exists. Skipping download.")

def load_dataset_from_directory(directory):
    """Load reviews and labels from directory"""
    reviews = []
    labels = []
    
    # Load positive reviews (label=1)
    pos_dir = os.path.join(directory, 'pos')
    if os.path.exists(pos_dir):
        for filename in os.listdir(pos_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    reviews.append(text)
                    labels.append(1)
    
    # Load negative reviews (label=0)
    neg_dir = os.path.join(directory, 'neg')
    if os.path.exists(neg_dir):
        for filename in os.listdir(neg_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    reviews.append(text)
                    labels.append(0)
    
    return reviews, labels

def clean_text(text):
    """Clean the text by removing HTML tags and normalizing whitespace"""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
    return text.strip()

def compute_metrics(pred):
    """Compute accuracy metric"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will be on CPU, which will be much slower.")
        print("If you have a GPU, make sure CUDA drivers are properly installed.")
    
    # Download and extract the dataset
    download_and_extract_dataset()
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    # Load and preprocess the training data
    print("Loading training data...")
    train_reviews, train_labels = load_dataset_from_directory(os.path.join(DATASET_PATH, 'train'))
    
    # Clean text
    print("Cleaning text...")
    train_reviews = [clean_text(review) for review in train_reviews]
    
    # Load and preprocess the test data
    print("Loading test data...")
    test_reviews, test_labels = load_dataset_from_directory(os.path.join(DATASET_PATH, 'test'))
    test_reviews = [clean_text(review) for review in test_reviews]
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = IMDBDataset(train_reviews, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = IMDBDataset(test_reviews, test_labels, tokenizer, MAX_LENGTH)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Training arguments - enable CUDA if available
    training_args = TrainingArguments(
        output_dir='./stanford-imdb-results',
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
    
    # Save the model
    model.save_pretrained("./distilbert-stanford-imdb-final")
    tokenizer.save_pretrained("./distilbert-stanford-imdb-final")

if __name__ == "__main__":
    main()