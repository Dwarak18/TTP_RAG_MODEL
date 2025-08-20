"""
Configuration settings for the Security TTP RAG model
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Dataset settings
    dataset_name: str = "tumeteor/Security-TTP-Mapping"
    data_dir: str = "data"
    
    # Model settings
    base_model: str = "microsoft/DialoGPT-medium"  # Can change to other models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    
    # RAG settings
    chunk_size: int = 256
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7
    
    # Vector database settings
    vector_db_type: str = "chromadb"  # or "faiss"
    vector_db_path: str = "embeddings/chroma_db"
    embedding_dim: int = 384
    
    # Paths
    model_save_path: str = "models/rag_model"
    embeddings_path: str = "embeddings"
    
    # Device settings
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    fp16: bool = True
    
    # Evaluation settings
    eval_batch_size: int = 8
    eval_steps: int = 500
    
    # Logging
    logging_steps: int = 100
    save_steps: int = 1000

# Global config instance
config = Config()
