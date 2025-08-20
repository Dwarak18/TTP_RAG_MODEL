"""
RAG Model Training for Security TTP
"""

import os
import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import logging
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Vector database for retrieval using ChromaDB"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.vector_db_path)
        self.collection: Optional[Any] = None
        self.embedding_model = SentenceTransformer(config.embedding_model)
    
    def create_collection(self, name: str = "security_ttp"):
        """Create or get collection"""
        try:
            self.collection = self.client.get_collection(name=name)
            logger.info(f"Retrieved existing collection: {name}")
        except:
            self.collection = self.client.create_collection(name=name)
            logger.info(f"Created new collection: {name}")
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """Add documents to vector database"""
        if not self.collection:
            self.create_collection()
        
        logger.info(f"Adding {len(documents)} documents to vector database...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            ids = [doc['id'] for doc in batch]
            texts = [doc['text'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]
            
            # Generate embeddings if not present
            if 'embedding' not in batch[0]:
                embeddings = self.embedding_model.encode(texts).tolist()
            else:
                embeddings = [doc['embedding'] for doc in batch]
            
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            if i % (batch_size * 5) == 0:
                logger.info(f"Added {i + len(batch)} documents...")
        
        logger.info("Documents added successfully!")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant documents"""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        top_k = top_k or config.top_k_retrieval
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return retrieved_docs

class SecurityTTPRAG:
    """RAG Model for Security TTP"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map="auto" if config.device == "cuda" else None
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vector_db = VectorDatabase()
        self.training_data: Optional[List[Dict]] = None
    
    def load_training_data(self):
        """Load processed training data"""
        logger.info("Loading training data...")
        
        # Load processed data
        data_path = os.path.join(config.data_dir, 'processed_ttp_data.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        # Load training pairs
        pairs_path = os.path.join(config.data_dir, 'training_pairs.json')
        with open(pairs_path, 'r', encoding='utf-8') as f:
            training_pairs = json.load(f)
        
        # Setup vector database
        self.vector_db.create_collection()
        self.vector_db.add_documents(processed_data)
        
        self.training_data = training_pairs
        logger.info(f"Loaded {len(training_pairs)} training examples")
        
        return training_pairs
    
    def create_rag_prompts(self, examples: List[Dict]) -> List[str]:
        """Create RAG-style prompts with retrieved context"""
        rag_prompts = []
        
        for example in examples:
            question = example['question']
            expected_answer = example['answer']
            
            # Retrieve relevant context
            retrieved_docs = self.vector_db.retrieve(question)
            
            # Build context from retrieved documents
            context_parts = []
            for doc in retrieved_docs:
                context_parts.append(doc['text'])
            
            context = "\n\n".join(context_parts[:3])  # Use top 3 documents
            
            # Create RAG prompt
            prompt = f"""Context: {context}

Question: {question}

Answer: {expected_answer}"""
            
            rag_prompts.append(prompt)
        
        return rag_prompts
    
    def prepare_training_dataset(self):
        """Prepare dataset for training"""
        if not self.training_data:
            self.load_training_data()
        
        logger.info("Preparing training dataset...")
        
        # Create RAG prompts
        if self.training_data:
            rag_prompts = self.create_rag_prompts(self.training_data)
        else:
            raise ValueError("No training data available")
        
        # Tokenize
        tokenized_data = []
        for prompt in rag_prompts:
            tokens = self.tokenizer(
                prompt,
                truncation=True,
                padding='max_length',
                max_length=config.max_length,
                return_tensors="pt"
            )
            tokenized_data.append({
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze(),
                'labels': tokens['input_ids'].squeeze().clone()
            })
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(tokenized_data)
        
        # Split into train/eval
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size
        
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, train_size + eval_size))
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def train(self):
        """Train the RAG model"""
        logger.info("Starting RAG model training...")
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_training_dataset()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.model_save_path,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            learning_rate=config.learning_rate,
            fp16=config.fp16,
            logging_steps=config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(config.model_save_path)
        
        logger.info(f"Model saved to {config.model_save_path}")
    
    def generate_response(self, question: str) -> str:
        """Generate response using RAG"""
        # Retrieve relevant context
        retrieved_docs = self.vector_db.retrieve(question)
        
        # Build context
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(doc['text'])
        
        context = "\n\n".join(context_parts[:3])
        
        # Create prompt
        prompt = f"""Context: {context}

Question: {question}

Answer:"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_length
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        answer = response[len(prompt):].strip()
        
        return answer

def main():
    """Main training pipeline"""
    # Initialize RAG model
    rag_model = SecurityTTPRAG()
    
    # Load training data
    rag_model.load_training_data()
    
    # Train the model
    rag_model.train()
    
    logger.info("RAG model training completed successfully!")
    
    # Test inference
    test_question = "What is a common attack technique used in cybersecurity?"
    response = rag_model.generate_response(test_question)
    logger.info(f"Test question: {test_question}")
    logger.info(f"Response: {response}")

if __name__ == "__main__":
    main()
