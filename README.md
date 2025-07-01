# imdb-nlp-extra-metadata

## Overview

This project explores the integration of movie-specific metadata (such as genre, vote count, and average rating) with textual input to enhance sentiment classification accuracy. We evaluate the impact of metadata on model performance using transformer-based models (DistilBERT, RoBERTa, BERT_base) and traditional baselines (TFIDF+SVM).

## Motivation

Traditional sentiment analysis models rely solely on text, potentially missing valuable contextual cues in metadata. By fusing metadata with text, we aim to improve the robustness and context-awareness of sentiment predictions.

## Data Validation

We ensure high-quality data through integrity checks and visualizations, including:
- Sentiment class distribution
- User and movie rating distributions
- Vote count (log scale)
- Top movie genres

## Methodology

We investigate several metadata fusion techniques:
- Early fusion (concatenating metadata with text features)
- Late fusion
- Cross-attention

Early fusion yielded the greatest improvement in accuracy.

## Results & Evaluation

- Adding metadata (early fusion) leads to statistically significant improvements for DistilBERT, BERT+IPTP, and TFIDF models.
- For RoBERTa, the improvement is not statistically significant and is very small.
- McNemar’s test is used to assess statistical significance of model differences.

## How to Run

1. **Install dependencies**  
   Make sure you have Python 3.x and required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download data**  
   Obtain the IMDb dataset and place it in the `data/` directory. The data should be organized as follows:
   ```
   data/
       aclImdb_v1/
           train/
               pos/
               neg/
           test/
               pos/
               neg/
   ```

3. **Import IMDb Metadata**  
   To import IMDb movie genres, average ratings, and vote counts into your database, run:
   ```bash
   python import_imdb_non_commercial.py
   ```
   
   Optional arguments:
   - `--skip-download` : Skip downloading IMDb datasets if already present
   - `--force-download` : Force re-download of IMDb datasets
   - `--basics-only` : Only process genres (skip ratings)
   - `--ratings-only` : Only process ratings (skip genres)

   Example:
   ```bash
   python import_imdb_non_commercial.py --skip-download
   ```

4. **Create the IMDb Reviews Database**  
   To create the initial SQLite database and import the IMDb reviews (text, ratings, sentiment, and URLs), run:
   ```bash
   python create_imdb_database.py
   ```
   This will process the dataset in the `./aclImdb` directory and create the `imdb_reviews.db` file with all reviews and metadata fields.

5. **Train and Evaluate TF-IDF + SVM (Without Metadata)**  
   To train a sentiment classifier using only review text (no metadata) with TF-IDF features and a linear SVM, run:
   ```bash
   python tfidf_svm_wo_metadata.py
   ```
   This will save the trained model, predictions, evaluation metrics, and experiment summary in a timestamped folder under `./training/tfidf-svm-imdb-archive_YYYYMMDD_HHMM/`.

6. **Train and Evaluate TF-IDF + SVM (With Metadata)**  
   To train a sentiment classifier using both review text and movie metadata (genre, rating, vote count) with TF-IDF features and a linear SVM, run:
   ```bash
   python tfidf_svm_metadata.py
   ```
   This will save the trained model, predictions, evaluation metrics, metadata statistics, and experiment summary in a timestamped folder under `./training/tfidf-svm-metadata-imdb-archive_YYYYMMDD_HHMM/`.

7. **Train and Evaluate DistilBERT (Without Metadata)**  
   To train a DistilBERT-based sentiment classifier using only review text (no metadata), run:
   ```bash
   python train_distilbert_wo_metadata.py
   ```
   This will save the trained model, tokenizer, predictions, evaluation metrics, and experiment summary in a timestamped folder under `./training/distilbert-wo-metadata-imdb-archive_YYYYMMDD_HHMM/`.

8. **Train and Evaluate DistilBERT (With Metadata)**  
   To train a DistilBERT-based sentiment classifier using both review text and movie metadata (genre, rating, vote count), run:
   ```bash
   python train_distilbert_metadata.py
   ```
   This will save the trained model, tokenizer, predictions, evaluation metrics, metadata, and experiment summary in a timestamped folder under `./training/distilbert-metadata-imdb-archive_YYYYMMDD_HHMM/`.

9. **Train and Evaluate RoBERTa (Without Metadata)**  
   To train a RoBERTa-based sentiment classifier using only review text (no metadata), run:
   ```bash
   python train_RoBERTa_wo_metadata.py
   ```
   This will save the trained model, tokenizer, predictions, evaluation metrics, and experiment summary in a timestamped folder under `./training/roberta-wo-metadata-imdb-archive_YYYYMMDD_HHMM/`.

10. **Train and Evaluate RoBERTa (With Metadata)**  
    To train a RoBERTa-based sentiment classifier using both review text and movie metadata (genre, rating, vote count), run:
    ```bash
    python train_RoBERTa_metadata.py
    ```
    This will save the trained model, tokenizer, predictions, evaluation metrics, metadata, and experiment summary in a timestamped folder under `./training/roberta-metadata-imdb-archive_YYYYMMDD_HHMM/`.

11. **Train and Evaluate BERT+ITPT (Without Metadata)**  
    To train a BERT-based sentiment classifier with domain-adaptive pre-training (ITPT) using only review text (no metadata), run:
    ```bash
    python train_bert_iptp_wo_metadata.py
    ```
    This will perform continued pre-training (ITPT) on IMDB data, then fine-tune for sentiment classification. All models, predictions, evaluation metrics, and experiment summaries will be saved in a timestamped folder under `./training/bert-itpt-wo-metadata-imdb-archive_YYYYMMDD_HHMM/`.

12. **Train and Evaluate BERT+ITPT (With Metadata)**  
    To train a BERT-based sentiment classifier with domain-adaptive pre-training (ITPT) using both review text and movie metadata (genre, rating, vote count), run:
    ```bash
    python train_bert_iptp_metadata.py
    ```
    This will perform continued pre-training (ITPT) on IMDB data with metadata, then fine-tune for sentiment classification. All models, predictions, evaluation metrics, metadata, and experiment summaries will be saved in a timestamped folder under `./training/bert-itpt-metadata-imdb-archive_YYYYMMDD_HHMM/`.


