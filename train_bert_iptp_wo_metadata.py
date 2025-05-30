import os
import re
import tarfile
import urllib.request
import numpy as np
import time
import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BertForMaskedLM,  # Add this for pre-training
    DataCollatorForLanguageModeling,  # Add this for MLM
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime

# Set CUDA memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear GPU cache at the start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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

# Add this new class for pre-training dataset
class IMDBPretrainingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def create_archive_directory(base_dir="./training/bert-itpt-wo-metadata-imdb-archive"):
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
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    
    # 2. Save sample test data (100 samples or less)
    print("Saving test data samples...")
    sample_size = min(100, len(test_dataset))
    sample_data = []
    for i in range(sample_size):
        sample_data.append({
            "text": test_dataset.reviews[i] if hasattr(test_dataset, "reviews") else "",
            "label": int(test_dataset.labels[i])
        })
    
    with open(os.path.join(archive_dir, "test_samples.json"), "w") as f:
        json.dump(sample_data, f, indent=2)
        
    # 3. Save model predictions on test set
    print("Saving model predictions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Create dataloader with smaller batch size for memory efficiency
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    all_preds = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch["label"]
            
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
        'attention_mask': sample_input['attention_mask'].unsqueeze(0).to(device)
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
        "model": config.get("model_name", "nlptown/bert-base-multilingual-uncased-sentiment"),
        "fusion_technique": config.get("fusion_technique", "text_only"),
        "itpt_enabled": config.get("itpt_enabled", True),
        "accuracy": eval_results.get("eval_accuracy", 0) * 100,
        "f1_score": eval_results.get("eval_f1", 0) * 100,
        "parameter_count": int(total_params),
        "inference_time_ms": float(inference_time),
    }
    
    with open(os.path.join(archive_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Archive complete! All data saved to {archive_dir}")
    return archive_dir

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

def continue_pretraining(model, tokenizer, train_texts, archive_dir):
    """Continue pre-training BERT on domain-specific data using MLM"""
    print("Starting domain-specific pre-training (ITPT)...")
    
    # Create pre-training dataset
    pretrain_dataset = IMDBPretrainingDataset(train_texts, tokenizer, MAX_LENGTH)
    
    # Data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Pre-training arguments
    pretrain_args = TrainingArguments(
        output_dir=os.path.join(archive_dir, "pretraining_outputs"),
        num_train_epochs=1,  # Usually 1-2 epochs for domain adaptation
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        logging_dir=os.path.join(archive_dir, "pretrain_logs"),
        logging_steps=500,
        save_steps=2000,
        no_cuda=False,
        dataloader_pin_memory=False,
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        save_total_limit=1,  # Only keep the latest checkpoint
    )
    
    # Create trainer for pre-training
    pretrain_trainer = Trainer(
        model=model,
        args=pretrain_args,
        train_dataset=pretrain_dataset,
        data_collator=data_collator,
    )
    
    # Continue pre-training
    print("Continuing pre-training on IMDB domain data...")
    pretrain_trainer.train()
    
    # Save the pre-trained model
    pretrain_model_dir = os.path.join(archive_dir, "domain_pretrained_model")
    model.save_pretrained(pretrain_model_dir)
    tokenizer.save_pretrained(pretrain_model_dir)
    
    print(f"Domain pre-training complete! Model saved to {pretrain_model_dir}")
    return pretrain_model_dir

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will be on CPU, which will be much slower.")
        print("If you have a GPU, make sure CUDA drivers are properly installed.")
    
    # Create archive directory at the beginning
    archive_dir = create_archive_directory()
    
    # Download and extract the dataset
    download_and_extract_dataset()
    
    # Load base BERT model for pre-training
    print("Loading base BERT model for ITPT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load BERT for Masked Language Modeling (pre-training)
    pretrain_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    # Load and preprocess the training data
    print("Loading training data for pre-training...")
    train_reviews, train_labels = load_dataset_from_directory(os.path.join(DATASET_PATH, 'train'))
    
    # Also load unsupervised data for pre-training
    unsup_reviews, _ = load_dataset_from_directory(os.path.join(DATASET_PATH, 'train/unsup'))
    
    # Combine all text data for pre-training
    all_pretrain_texts = train_reviews + unsup_reviews
    
    # Clean text
    print("Cleaning text...")
    all_pretrain_texts = [clean_text(review) for review in all_pretrain_texts]
    train_reviews = [clean_text(review) for review in train_reviews]
    
    # Step 1: Continue pre-training on domain data
    pretrain_model_dir = continue_pretraining(
        pretrain_model, tokenizer, all_pretrain_texts, archive_dir
    )
    
    # Step 2: Load the domain-adapted model for fine-tuning
    print("Loading domain-adapted model for fine-tuning...")
    model = BertForSequenceClassification.from_pretrained(
        pretrain_model_dir,
        num_labels=2
    )
    
    # Load and preprocess the test data
    print("Loading test data...")
    test_reviews, test_labels = load_dataset_from_directory(os.path.join(DATASET_PATH, 'test'))
    test_reviews = [clean_text(review) for review in test_reviews]
    
    # Create datasets for fine-tuning
    print("Creating datasets for fine-tuning...")
    train_dataset = IMDBDataset(train_reviews, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = IMDBDataset(test_reviews, test_labels, tokenizer, MAX_LENGTH)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Fine-tuning arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(archive_dir, "finetuning_outputs"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        logging_dir=os.path.join(archive_dir, "finetune_logs"),
        logging_steps=500,
        save_steps=1500,
        eval_steps=1500,
        do_eval=True,
        no_cuda=False,
        dataloader_pin_memory=False,
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        eval_accumulation_steps=10
    )
    
    # Create trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Fine-tune the model
    print("Starting fine-tuning on sentiment classification...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    
    print(f"Evaluation Results: {eval_results}")
    print(f"Final accuracy: {eval_results['eval_accuracy'] * 100:.2f}%")
    print(f"Precision: {eval_results['eval_precision'] * 100:.2f}%")
    print(f"Recall: {eval_results['eval_recall'] * 100:.2f}%")
    print(f"F1 Score: {eval_results['eval_f1'] * 100:.2f}%")
    
    # Create comprehensive experiment configuration
    config = {
        "fusion_technique": "text_only",
        "model_name": "bert-base-uncased",
        "itpt_enabled": True,
        "itpt_description": "True ITPT: Continued pre-training on IMDB domain data with MLM",
        "pretrain_epochs": 1,
        "finetune_epochs": 3,
        "batch_size": 8,
        "max_length": MAX_LENGTH,
        "experiment_date": datetime.now().strftime("%Y-%m-%d"),
        "experiment_time": datetime.now().strftime("%H:%M:%S"),
        "description": "IMDB sentiment analysis using BERT with true ITPT (domain-continued pre-training + fine-tuning)"
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
    
    print(f"Complete ITPT experiment archive created at {archive_dir}")


if __name__ == "__main__":
    main()