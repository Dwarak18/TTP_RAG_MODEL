"""
Inference and Evaluation for Security TTP RAG Model
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Optional
import logging
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityTTPInference:
    """Inference class for Security TTP RAG model"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.model_save_path
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.vector_db: Optional[Any] = None
        self.embedding_model = SentenceTransformer(config.embedding_model)
        
        self.load_model()
        self.setup_vector_db()
    
    def load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if config.fp16 else torch.float32,
                device_map="auto" if config.device == "cuda" else None
            )
            
            if self.tokenizer and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.warning(f"Could not load fine-tuned model: {e}")
            logger.info("Loading base model instead...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.float16 if config.fp16 else torch.float32,
                device_map="auto" if config.device == "cuda" else None
            )
            
            if self.tokenizer and self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def setup_vector_db(self):
        """Setup vector database for retrieval"""
        try:
            client = chromadb.PersistentClient(path=config.vector_db_path)
            self.vector_db = client.get_collection(name="security_ttp")
            logger.info("Vector database loaded successfully!")
        except Exception as e:
            logger.error(f"Could not load vector database: {e}")
            logger.info("Please run data preprocessing first.")
    
    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant context for a query"""
        if not self.vector_db:
            return []
        
        top_k = top_k or config.top_k_retrieval
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            results = self.vector_db.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            retrieved_docs = []
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def generate_response(self, question: str, max_new_tokens: int = 150) -> Dict[str, Any]:
        """Generate response using RAG"""
        # Retrieve relevant context
        retrieved_docs = self.retrieve_context(question)
        
        # Build context from retrieved documents
        context_parts = []
        for doc in retrieved_docs[:3]:  # Use top 3 documents
            context_parts.append(doc['text'])
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Create RAG prompt
        prompt = f"""Context: {context}

Question: {question}

Answer:"""
        
        # Tokenize
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
            
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_length - max_new_tokens
        )
        
        if config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        if not self.model:
            raise ValueError("Model not loaded")
            
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        answer = full_response[len(prompt):].strip()
        
        return {
            'question': question,
            'answer': answer,
            'context': context,
            'retrieved_docs': retrieved_docs,
            'prompt': prompt
        }
    
    def batch_inference(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Run inference on a batch of questions"""
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            result = self.generate_response(question)
            results.append(result)
        
        return results
    
    def interactive_chat(self):
        """Interactive chat interface"""
        logger.info("Starting interactive chat. Type 'quit' to exit.")
        print("\n=== Security TTP RAG Assistant ===")
        print("Ask questions about security tactics, techniques, and procedures.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nThinking...")
                result = self.generate_response(question)
                
                print(f"\nAnswer: {result['answer']}")
                
                # Show retrieved context if verbose
                show_context = input("\nShow retrieved context? (y/n): ").strip().lower()
                if show_context == 'y':
                    print(f"\nRetrieved Context:")
                    for i, doc in enumerate(result['retrieved_docs'][:3]):
                        print(f"\n--- Document {i+1} ---")
                        print(f"Text: {doc['text'][:200]}...")
                        if doc['distance']:
                            print(f"Similarity: {1 - doc['distance']:.3f}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def evaluate_on_test_questions(self):
        """Evaluate model on predefined test questions"""
        test_questions = [
            "What is phishing and how does it work?",
            "Explain lateral movement techniques in cybersecurity.",
            "What are common persistence mechanisms used by attackers?",
            "How do attackers perform privilege escalation?",
            "What is command and control (C2) in cyber attacks?",
            "Describe common data exfiltration methods.",
            "What are living-off-the-land techniques?",
            "Explain defense evasion tactics used by malware.",
            "How do attackers perform reconnaissance?",
            "What is the difference between tactics and techniques in MITRE ATT&CK?"
        ]
        
        logger.info("Evaluating model on test questions...")
        results = self.batch_inference(test_questions)
        
        # Save evaluation results
        eval_path = os.path.join(config.data_dir, 'evaluation_results.json')
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {eval_path}")
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        for i, result in enumerate(results):
            print(f"\nQ{i+1}: {result['question']}")
            print(f"A{i+1}: {result['answer'][:100]}...")
        
        return results

def main():
    """Main inference pipeline"""
    # Initialize inference
    inference = SecurityTTPInference()
    
    # Run evaluation
    inference.evaluate_on_test_questions()
    
    # Start interactive chat
    inference.interactive_chat()

if __name__ == "__main__":
    main()
