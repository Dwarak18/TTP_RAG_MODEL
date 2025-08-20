"""
Complete training pipeline for Security TTP RAG model
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import SecurityTTPPreprocessor
from train_rag import SecurityTTPRAG
from inference import SecurityTTPInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Complete training pipeline"""
    print("=== Security TTP RAG Model Training Pipeline ===\n")
    
    try:
        # Step 1: Data Preprocessing
        print("Step 1: Data Preprocessing")
        print("-" * 30)
        preprocessor = SecurityTTPPreprocessor()
        preprocessor.load_dataset()
        preprocessor.explore_dataset_structure()
        preprocessor.preprocess_for_rag()
        preprocessor.generate_embeddings()
        preprocessor.create_training_pairs()
        preprocessor.save_processed_data()
        print("✓ Data preprocessing completed!\n")
        
        # Step 2: Model Training
        print("Step 2: RAG Model Training")
        print("-" * 30)
        rag_model = SecurityTTPRAG()
        rag_model.load_training_data()
        rag_model.train()
        print("✓ Model training completed!\n")
        
        # Step 3: Inference and Evaluation
        print("Step 3: Model Evaluation")
        print("-" * 30)
        inference = SecurityTTPInference()
        inference.evaluate_on_test_questions()
        print("✓ Model evaluation completed!\n")
        
        print("=== Training Pipeline Completed Successfully! ===")
        print("\nYou can now:")
        print("1. Run 'python src/inference.py' for interactive chat")
        print("2. Import and use the SecurityTTPInference class in your own code")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
