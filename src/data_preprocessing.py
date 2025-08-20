"""
Data preprocessing for Security TTP RAG model
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityTTPPreprocessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.dataset: Optional[Any] = None
        self.processed_data = []
        
    def load_dataset(self):
        """Load the Security TTP Mapping dataset"""
        logger.info("Loading Security TTP Mapping dataset...")
        try:
            self.dataset = load_dataset(config.dataset_name)
            if self.dataset:
                logger.info(f"Dataset loaded successfully. Keys: {list(self.dataset.keys())}")
                
                # Print dataset structure for inspection
                if 'train' in self.dataset:
                    sample = self.dataset['train'][0]
                    logger.info(f"Sample data structure: {sample.keys()}")
                    logger.info(f"Sample: {sample}")
            
            return self.dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def explore_dataset_structure(self):
        """Explore and understand the dataset structure"""
        if not self.dataset:
            self.load_dataset()
        
        if self.dataset:
            for split_name, split_data in self.dataset.items():
                logger.info(f"\n=== {split_name.upper()} SPLIT ===")
                logger.info(f"Number of examples: {len(split_data)}")
                
                if len(split_data) > 0:
                    sample = split_data[0]
                    logger.info(f"Columns: {list(sample.keys())}")
                    
                    # Print first few examples
                    for i, example in enumerate(split_data[:3]):
                        logger.info(f"\nExample {i+1}:")
                        for key, value in example.items():
                            logger.info(f"  {key}: {str(value)[:200]}...")
    
    def create_text_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), config.chunk_size - config.chunk_overlap):
            chunk = ' '.join(words[i:i + config.chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
        
        return chunks
    
    def preprocess_for_rag(self):
        """Preprocess dataset for RAG training"""
        if not self.dataset:
            self.load_dataset()
        
        logger.info("Preprocessing data for RAG...")
        
        # Use the train split if available, otherwise use the first available split
        if self.dataset and 'train' in self.dataset:
            data = self.dataset['train']
        elif self.dataset:
            data = self.dataset[list(self.dataset.keys())[0]]
        else:
            raise ValueError("No dataset loaded")
        
        for idx, example in enumerate(data):
            try:
                # Extract relevant fields (adjust based on actual dataset structure)
                # Common fields might be: technique, description, tactic, etc.
                
                # Create a comprehensive text representation
                text_parts = []
                context_parts = []
                
                for key, value in example.items():
                    if isinstance(value, str) and len(value.strip()) > 0:
                        text_parts.append(f"{key}: {value}")
                        
                        # Separate context and query parts
                        if key.lower() in ['description', 'technique', 'procedure', 'detail']:
                            context_parts.append(value)
                
                full_text = " | ".join(text_parts)
                context_text = " ".join(context_parts)
                
                # Create chunks for retrieval
                chunks = self.create_text_chunks(context_text) if context_text else [full_text]
                
                for chunk_idx, chunk in enumerate(chunks):
                    processed_item = {
                        'id': f"{idx}_{chunk_idx}",
                        'original_id': idx,
                        'text': chunk,
                        'full_context': full_text,
                        'metadata': {k: v for k, v in example.items() if not isinstance(v, str) or len(str(v)) < 100}
                    }
                    self.processed_data.append(processed_item)
                
                if idx % 100 == 0:
                    logger.info(f"Processed {idx} examples...")
                    
            except Exception as e:
                logger.warning(f"Error processing example {idx}: {e}")
                continue
        
        logger.info(f"Preprocessing complete. Total chunks: {len(self.processed_data)}")
        return self.processed_data
    
    def generate_embeddings(self):
        """Generate embeddings for the processed text chunks"""
        if not self.processed_data:
            self.preprocess_for_rag()
        
        logger.info("Generating embeddings...")
        
        texts = [item['text'] for item in self.processed_data]
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=True)
            all_embeddings.extend(batch_embeddings)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"Generated embeddings for {i + len(batch_texts)} texts...")
        
        # Add embeddings to processed data
        for i, embedding in enumerate(all_embeddings):
            self.processed_data[i]['embedding'] = embedding.tolist()
        
        logger.info("Embedding generation complete.")
        return all_embeddings
    
    def save_processed_data(self):
        """Save processed data to disk"""
        os.makedirs(config.data_dir, exist_ok=True)
        
        # Save processed data
        output_path = os.path.join(config.data_dir, 'processed_ttp_data.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed data saved to {output_path}")
        
        # Save embeddings separately for faster loading
        embeddings = np.array([item['embedding'] for item in self.processed_data])
        embeddings_path = os.path.join(config.embeddings_path, 'ttp_embeddings.npy')
        os.makedirs(config.embeddings_path, exist_ok=True)
        np.save(embeddings_path, embeddings)
        
        logger.info(f"Embeddings saved to {embeddings_path}")
        
        # Save metadata
        metadata = [{'id': item['id'], 'text': item['text'], 'metadata': item['metadata']} 
                   for item in self.processed_data]
        metadata_path = os.path.join(config.data_dir, 'ttp_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def create_training_pairs(self):
        """Create question-answer pairs for RAG training"""
        if not self.processed_data:
            logger.error("No processed data available. Run preprocessing first.")
            return
        
        training_pairs = []
        
        for item in self.processed_data:
            # Create synthetic questions based on the content
            text = item['text']
            metadata = item['metadata']
            
            # Generate different types of questions
            questions = []
            
            # Technique-based questions
            if 'technique' in str(metadata).lower():
                questions.append(f"What is the technique described in this security context?")
                questions.append(f"Explain the security technique mentioned.")
            
            # Tactic-based questions
            if 'tactic' in str(metadata).lower():
                questions.append(f"What tactic is being used in this security scenario?")
                questions.append(f"Describe the security tactic.")
            
            # General questions
            questions.extend([
                f"What does this security information describe?",
                f"Explain this security procedure.",
                f"What are the key points of this security context?"
            ])
            
            for question in questions:
                training_pairs.append({
                    'question': question,
                    'context': text,
                    'answer': text,  # In RAG, the context is often the answer
                    'metadata': metadata
                })
        
        # Save training pairs
        pairs_path = os.path.join(config.data_dir, 'training_pairs.json')
        with open(pairs_path, 'w', encoding='utf-8') as f:
            json.dump(training_pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {len(training_pairs)} training pairs. Saved to {pairs_path}")
        return training_pairs

def main():
    """Main preprocessing pipeline"""
    preprocessor = SecurityTTPPreprocessor()
    
    # Load and explore dataset
    preprocessor.load_dataset()
    preprocessor.explore_dataset_structure()
    
    # Preprocess for RAG
    preprocessor.preprocess_for_rag()
    
    # Generate embeddings
    preprocessor.generate_embeddings()
    
    # Create training pairs
    preprocessor.create_training_pairs()
    
    # Save all processed data
    preprocessor.save_processed_data()
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
