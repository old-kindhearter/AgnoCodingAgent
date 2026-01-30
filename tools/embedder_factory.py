import time
import threading
from typing import List, Optional
from sentence_transformers import SentenceTransformer

class EmbedderSingleton:
    """
    Minimal singleton for embedding model.
    """
    _instance: Optional['EmbedderSingleton'] = None
    _lock = threading.Lock()
    
    MODEL_ID = "/srv/AgnoCodingAgent/cache/jinaai/jina-embeddings-v2-base-code"
    
    def __init__(self):
        import torch
         
        # Simple device detection
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"Loading embedding model on {self.device}...")
        start = time.perf_counter()
        
        # Load model - SentenceTransformer handles caching internally
        self.model = SentenceTransformer(self.MODEL_ID, device=self.device)
        
        # Set reasonable max length for code
        self.model.max_seq_length = 512
        
        elapsed = time.perf_counter() - start
        print(f"Model loaded in {elapsed:.2f}s")
    
    @classmethod
    def get(cls) -> 'EmbedderSingleton':
        """Thread-safe singleton access"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def encode(self, text: str) -> List[float]:
        """Encode single text"""
        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).tolist()
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encode multiple texts"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        ).tolist()