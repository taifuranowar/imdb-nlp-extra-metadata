import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score

def main():
    # Load IMDB dataset
    print("Loading IMDB dataset...")
    dataset = load_dataset("stanfordnlp/imdb")
    
    # Load DistilBERT tokenizer and model
    print("Loading DistilBERT model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2
    )
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )
    
    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing dataset",
    )
    
    # Convert to PyTorch format
    tokenized_datasets.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "label"]
    )
    
    # Define compute metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}
    
    # Use minimal training arguments
    training_args = TrainingArguments(
        output_dir="./distilbert-imdb-results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
        save_steps=1500,       # Save model once per epoch
        eval_steps=1500,       # Evaluate once per epoch
        do_eval=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    
    print(f"Evaluation Results: {eval_results}")
    print(f"Final accuracy: {eval_results['eval_accuracy'] * 100:.2f}%")
    
    # Save the model
    model.save_pretrained("./distilbert-imdb-final")
    tokenizer.save_pretrained("./distilbert-imdb-final")

if __name__ == "__main__":
    main()