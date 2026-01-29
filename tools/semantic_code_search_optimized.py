import os
import time
import threading
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer
from agno.tools import Toolkit
from dotenv import load_dotenv

# Suppress Jina model warnings
import warnings
warnings.filterwarnings("ignore", message="Some weights of BertModel were not initialized")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")


class EmbedderSingleton:
    """
    Minimal singleton for embedding model.
    """
    _instance: Optional['EmbedderSingleton'] = None
    _lock = threading.Lock()
    
    MODEL_ID = "jinaai/jina-embeddings-v2-base-code"
    
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

class ChromaDBConnection:
    """
    Simple ChromaDB connection manager.
    Reuses connections but without complex pooling overhead.
    """
    _connections: Dict[str, Tuple[chromadb.ClientAPI, chromadb.Collection]] = {}
    
    @classmethod
    def get_collection(cls, db_path: str) -> Tuple[chromadb.Collection, str]:
        """Get or create connection to ChromaDB collection"""
        if db_path not in cls._connections:
            print(f"Connecting to: {db_path}")
            client = chromadb.PersistentClient(path=db_path)
            
            # Auto-detect collection name
            collections = client.list_collections()
            if not collections:
                raise ValueError(f"No collections in {db_path}")
            
            collection_name = collections[0].name
            collection = client.get_collection(collection_name)
            
            print(f"Collection '{collection_name}' ({collection.count()} docs)")
            cls._connections[db_path] = (client, collection)
        
        _, collection = cls._connections[db_path]
        return collection, collection.name
    
    @classmethod
    def clear(cls):
        """Clear all connections"""
        cls._connections.clear()

class CodeSearch(Toolkit):
    """
    - Uses functools.lru_cache
    - Minimal abstraction overhead
    - Simple connection reuse
    """
    
    def __init__(self):
        super().__init__(name="semantic_code_search", tools=[self.semantic_code_search])
    
    def semantic_code_search( 
        self, 
        vec_repo_path: str, 
        query: str,
        max_results: int = 5
    ) -> List[str]:
        """
        Search the local vector database for code relevant to the query.
        
        Args:
            vec_repo_path: Path to vector database
            query: Search query
            max_results: Number of results (default 5)
        
        Returns:
            List of matching code snippets
        """
        start = time.perf_counter()
        vec_repo_path = os.path.abspath(vec_repo_path)
        
        # Get cached resources
        embedder = EmbedderSingleton.get()
        collection,_ = ChromaDBConnection.get_collection(vec_repo_path)
        
        # Encode query
        query_embedding = embedder.encode(query)
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            include=["documents"]
        )
        
        documents = results["documents"][0] if results["documents"] else []
        
        elapsed = (time.perf_counter() - start) * 1000
        print(f"Search: {len(documents)} results in {elapsed:.1f}ms")
        
        return documents
    
    def search_batch(
        self,
        vec_repo_path: str,
        queries: List[str],
        max_results: int = 5
    ) -> Dict[str, List[str]]:
        
        start = time.perf_counter()
        vec_repo_path = os.path.abspath(vec_repo_path)
        
        # Check cache first
        results = {}
        uncached_queries = []
        uncached_indices = []
            
        # Get resources
        embedder = EmbedderSingleton.get()
        collection, _ = ChromaDBConnection.get_collection(vec_repo_path)
            
        # Batch embed - this IS faster than individual encoding
        embed_start = time.perf_counter()
        embeddings = embedder.encode_batch(uncached_queries)
        embed_time = (time.perf_counter() - embed_start) * 1000
            
        # Batch search
        search_start = time.perf_counter()
        raw_results = collection.query(
            query_embeddings=embeddings,
            n_results=max_results,
            include=["documents"]
        )
        search_time = (time.perf_counter() - search_start) * 1000
            
        # Process results
        for i, query in enumerate(uncached_queries):
            docs = raw_results["documents"][i] if raw_results["documents"] else []
            results[query] = docs
            
        print(f"Batch: {len(uncached_queries)} queries, embed={embed_time:.1f}ms, search={search_time:.1f}ms")
        
        total = (time.perf_counter() - start) * 1000
        print(f"Total batch: {total:.1f}ms for {len(queries)} queries")
        
        return results
    
    @staticmethod
    def warmup(vec_repo_path: Optional[str] = None):
        """
        Warmup the search engine.
        
        Args:
            vec_repo_path: Optional path to preload a specific database
        """
        print("Warming up...")
        start = time.perf_counter()
        
        # Load embedding model
        EmbedderSingleton.get()
        
        # Optionally preload database
        if vec_repo_path:
            ChromaDBConnection.get_collection(vec_repo_path)
        
        print(f"Warmup complete in {time.perf_counter()-start:.2f}s")
    
    @staticmethod
    def clear_cache():
        """Clear all caches"""
        ChromaDBConnection.clear()
        print("Caches cleared")