# Bengali Book Genre Classification with BanglaBERT

## Project Overview
This repository contains a machine learning project for classifying Bengali book summaries into seven genres: Fiction, Thriller, Children's Book, Political, Science-Fiction, War, and Motivational. The project leverages the `csebuetnlp/banglabert` model, a BERT-based transformer fine-tuned for Bengali language processing, implemented in Python using PyTorch and the Hugging Face Transformers library. The codebase includes data preprocessing, model training, and evaluation scripts, with a focus on handling mixed Bengali-English text through cleaning and translation.

## Features
- **Data Preprocessing**: Cleans book summaries by removing URLs, emojis, and excessive punctuation, standardizes text, and translates English segments to Bengali using `googletrans`.
- **Dataset**: Processes a dataset of ~4.5K book summaries labeled across seven genres, split into training (80%) and validation (20%) sets.
- **Model**: Fine-tunes `csebuetnlp/banglabert` for multi-class classification (7 classes) with a custom classifier head.
- **Training**: Implements a training pipeline with PyTorch, using Adam optimizer, cross-entropy loss, and a batch size of 2, targeting a learning rate of 1e-6.
- **Evaluation**: Tracks training and validation loss/accuracy per epoch, with progress monitoring via `tqdm`.
- **Environment**: Utilizes Jupyter notebooks (`Data preprocessing.ipynb` and `Training.ipynb`) for data preparation and model training.

## Repository Contents
- **Notebooks**:
  - `Data preprocessing.ipynb`: Handles data loading, cleaning, translation of English to Bengali, and feature engineering (e.g., sentence length calculation).
  - `Training.ipynb`: Implements the `TextDataset` class, `BertClassifier` model, and training loop for fine-tuning BanglaBERT.
- **Data**:
  - `train.csv`: Training dataset with book summaries and genre labels.
  - `test.csv`: Test dataset with book summaries (labels not included).
  - `cleaned_data.csv`: Preprocessed training data with cleaned and translated text.
- **Dependencies**:
  - Python 3.11.5, PyTorch, Transformers, Pandas, NumPy, Matplotlib, Seaborn, googletrans, langid, tqdm.

## Requirements
- **Python Libraries**:
  ```bash
  pip install pandas numpy matplotlib seaborn torch transformers googletrans==4.0.0-rc1 langid tqdm
