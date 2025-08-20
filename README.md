# Security TTP RAG Model

This project implements a Retrieval-Augmented Generation (RAG) model for security Tactics, Techniques, and Procedures (TTP) mapping using the `tumeteor/Security-TTP-Mapping` dataset.

## Features

- Data preprocessing and embedding generation
- Vector database setup with ChromaDB/FAISS
- RAG model training with fine-tuning capabilities
- Inference pipeline for security TTP queries
- Evaluation metrics and benchmarking

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run data preprocessing:
```bash
python src/data_preprocessing.py
```

3. Train the RAG model:
```bash
python src/train_rag.py
```

4. Run inference:
```bash
python src/inference.py
```

## Project Structure

- `src/`: Source code
  - `data_preprocessing.py`: Dataset loading and preprocessing
  - `train_rag.py`: RAG model training pipeline
  - `inference.py`: Inference and evaluation
  - `config.py`: Configuration settings
- `data/`: Processed datasets
- `models/`: Saved model checkpoints
- `embeddings/`: Vector embeddings storage
