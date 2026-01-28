"""
Optimized Semantic Code Search
==============================

Optimizations:
1. Singleton embedder with GPU optimization
2. ChromaDB connection pooling
3. Query result caching (LRU with TTL)
4. Batch embedding with optimal batch size
5. FP16 inference on GPU/MPS
6. Warmup utilities
"""

import os
import time
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from agno.tools import Toolkit
from dotenv import load_dotenv

# Suppress the Jina model warnings
import warnings
warnings.filterwarnings("ignore", message="Some weights of BertModel were not initialized")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")


@dataclass
class SearchResult:
    """Structured search result"""
    content: str
    score: float
    file_name: str = ""
    chunk_index: int = 0


class OptimizedEmbedder:
    """
    Highly optimized embedding with:
    - Singleton pattern
    - GPU acceleration
    - Batch optimization
    """
    _instance: Optional['OptimizedEmbedder'] = None
    _lock = threading.Lock()
    
    def __init__(
        self, 
        model_id: str = "jinaai/jina-embeddings-v2-base-code",
        use_fp16: bool = True,
        max_seq_length: int = 512
    ):
        print(f"Loading embedding model: {model_id}")
        start = time.perf_counter()
        
        # Detect device
        import torch
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            print(f"   Using Apple Silicon MPS")
        else:
            self.device = "cpu"
            print(f"   Using CPU")
        
        # Load model
        self.model = SentenceTransformer(model_id, device=self.device)
        self.model.max_seq_length = max_seq_length
        
        # Enable FP16 on GPU for ~2x speedup
        if use_fp16 and self.device in ("cuda", "mps"):
            self.model = self.model.half()
            print(f"   FP16 enabled")
        
        # Warmup (compiles kernels, allocates memory)
        _ = self.model.encode("warmup query", normalize_embeddings=True)
        
        elapsed = time.perf_counter() - start
        print(f"Embedder ready in {elapsed:.2f}s")
        
        self._optimal_batch_size = self._find_optimal_batch_size()
    
    def _find_optimal_batch_size(self) -> int:
        """Find optimal batch size for current hardware"""
        if self.device == "cuda":
            import torch
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            if gpu_mem > 20e9:
                return 128
            elif gpu_mem > 12e9:
                return 64
            else:
                return 32
        elif self.device == "mps":
            return 32
        else:
            return 16
    
    @classmethod
    def get_instance(cls) -> 'OptimizedEmbedder':
        """Thread-safe singleton access"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single query"""
        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts"""
        return self.model.encode(
            texts,
            batch_size=self._optimal_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )


class ChromaDBPool:
    """ChromaDB connection pool"""
    _clients: Dict[str, chromadb.ClientAPI] = {}
    _collections: Dict[str, chromadb.Collection] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_collection(cls, db_path: str) -> Tuple[chromadb.Collection, str]:
        """Get collection with auto-detection (thread-safe)"""
        with cls._lock:
            if db_path not in cls._clients:
                print(f"Connecting to ChromaDB: {db_path}")
                cls._clients[db_path] = chromadb.PersistentClient(path=db_path)
            
            client = cls._clients[db_path]
            
            if db_path not in cls._collections:
                collections = client.list_collections()
                if not collections:
                    raise ValueError(f"No collections found in {db_path}")
                
                collection_name = collections[0].name
                cls._collections[db_path] = client.get_collection(collection_name)
                print(f"Collection '{collection_name}' ({cls._collections[db_path].count()} docs)")
            
            return cls._collections[db_path], cls._collections[db_path].name
    
    @classmethod
    def clear(cls):
        with cls._lock:
            cls._clients.clear()
            cls._collections.clear()


class LRUCache:
    """Thread-safe LRU cache with TTL"""
    def __init__(self, max_size: int = 256, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, List[str]]] = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[List[str]]:
        with self._lock:
            if key not in self._cache:
                return None
            
            timestamp, value = self._cache[key]
            
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None
            
            self._cache.move_to_end(key)
            return value
    
    def set(self, key: str, value: List[str]):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
            
            self._cache[key] = (time.time(), value)
    
    def clear(self):
        with self._lock:
            self._cache.clear()


# Global cache
_query_cache = LRUCache(max_size=256, ttl_seconds=3600)


class CodeSearch(Toolkit):
    """
    Optimized semantic code search.
    
    Features:
    - Singleton embedder with FP16
    - Connection pooling
    - LRU caching with TTL
    - Batch query support
    """
    
    def __init__(self, prefetch_db: Optional[str] = None):
        super().__init__(name="semantic_code_search", tools=[self.semantic_code_search])
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        if prefetch_db:
            self._executor.submit(self._warmup, prefetch_db)
    
    def _warmup(self, db_path: str):
        """Background warmup"""
        OptimizedEmbedder.get_instance()
        ChromaDBPool.get_collection(db_path)
    
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
            max_results: Number of results
        
        Returns:
            List of matching code snippets
        """
        start = time.perf_counter()
        vec_repo_path = os.path.abspath(vec_repo_path)
        
        # Check cache
        cache_key = f"{vec_repo_path}:{query}:{max_results}"
        cached = _query_cache.get(cache_key)
        if cached is not None:
            print(f"Cache hit ({(time.perf_counter()-start)*1000:.1f}ms)")
            return cached
        
        # Get resources
        embedder = OptimizedEmbedder.get_instance()
        collection, _ = ChromaDBPool.get_collection(vec_repo_path)
        
        # Embed query
        embed_start = time.perf_counter()
        query_embedding = embedder.embed_single(query)
        embed_time = (time.perf_counter() - embed_start) * 1000
        
        # Search
        search_start = time.perf_counter()
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=max_results,
            include=["documents"]
        )
        search_time = (time.perf_counter() - search_start) * 1000
        
        # Extract results
        contents = results["documents"][0] if results["documents"] else []
        
        # Cache
        _query_cache.set(cache_key, contents)
        
        total = (time.perf_counter() - start) * 1000
        print(f"{len(contents)} results in {total:.1f}ms (embed: {embed_time:.1f}ms, search: {search_time:.1f}ms)")
        
        return contents
    
    def search_batch(
        self,
        vec_repo_path: str,
        queries: List[str],
        max_results: int = 5
    ) -> Dict[str, List[str]]:
        """
        Batch search - optimal for multiple queries.
        
        Args:
            vec_repo_path: Path to vector database
            queries: List of search queries
            max_results: Results per query
        
        Returns:
            Dict mapping each query to its results
        """
        start = time.perf_counter()
        vec_repo_path = os.path.abspath(vec_repo_path)
        
        print(f"Batch search: {len(queries)} queries")
        
        # Check cache
        results = {}
        uncached_queries = []
        
        for query in queries:
            cache_key = f"{vec_repo_path}:{query}:{max_results}"
            cached = _query_cache.get(cache_key)
            if cached is not None:
                results[query] = cached
            else:
                uncached_queries.append(query)
        
        if not uncached_queries:
            print(f"All {len(queries)} queries cached!")
            return results
        
        print(f"   {len(queries) - len(uncached_queries)} cached, {len(uncached_queries)} to search")
        
        # Get resources
        embedder = OptimizedEmbedder.get_instance()
        collection, _ = ChromaDBPool.get_collection(vec_repo_path)
        
        # Batch embed
        embed_start = time.perf_counter()
        embeddings = embedder.embed_batch(uncached_queries)
        embed_time = (time.perf_counter() - embed_start) * 1000
        
        # Batch search
        search_start = time.perf_counter()
        raw_results = collection.query(
            query_embeddings=embeddings.tolist(),
            n_results=max_results,
            include=["documents"]
        )
        search_time = (time.perf_counter() - search_start) * 1000
        
        # Process and cache results
        for i, query in enumerate(uncached_queries):
            contents = raw_results["documents"][i] if raw_results["documents"] else []
            results[query] = contents
            cache_key = f"{vec_repo_path}:{query}:{max_results}"
            _query_cache.set(cache_key, contents)
        
        total = (time.perf_counter() - start) * 1000
        per_query = total / len(queries)
        print(f"Batch done in {total:.1f}ms ({per_query:.1f}ms/query, embed: {embed_time:.1f}ms, search: {search_time:.1f}ms)")
        
        return results
    
    @staticmethod
    def warmup(vec_repo_path: str):
        """Warmup embedder and DB connection"""
        print("Warming up...")
        start = time.perf_counter()
        OptimizedEmbedder.get_instance()
        ChromaDBPool.get_collection(vec_repo_path)
        print(f"Warmup done in {time.perf_counter()-start:.2f}s")
    
    @staticmethod
    def clear_cache():
        """Clear all caches"""
        _query_cache.clear()
        ChromaDBPool.clear()
        print("Caches cleared")
    
    @staticmethod
    def get_stats() -> Dict:
        """Get performance statistics"""
        embedder = OptimizedEmbedder.get_instance()
        return {
            "device": embedder.device,
            "optimal_batch_size": embedder._optimal_batch_size,
            "cache_size": len(_query_cache._cache),
            "cache_max": _query_cache.max_size,
        }


if __name__ == "__main__":
    load_dotenv()
    
    print("=" * 70)
    print("OPTIMIZED CODE SEARCH - BENCHMARK")
    print("=" * 70)
    
    vec_db_path = '../AgnoCodingAgent/Knowledge/vector_db/AgnoCodingAgent'
    vec_db_path = os.path.abspath(vec_db_path)
    
    if not os.path.exists(vec_db_path):
        print(f"DB not found: {vec_db_path}")
        exit(1)
    
    searcher = CodeSearch()
    
    queries = [
        "embedder factory singleton pattern",
        "how to build vector database",
        "chromadb integration",
        "sentence transformer embedding",
        "parallel processing code",
    ]
    
    print("\n" + "=" * 70)
    print("TEST 1: Cold Start (includes model loading)")
    print("=" * 70)
    results = searcher.semantic_code_search(vec_db_path, queries[0])
    print(f"Results: {len(results)}")
    
    print("\n" + "=" * 70)
    print("TEST 2: Warm Single Queries")
    print("=" * 70)
    for q in queries[1:3]:
        results = searcher.semantic_code_search(vec_db_path, q)
    
    print("\n" + "=" * 70)
    print("TEST 3: Cached Query")
    print("=" * 70)
    results = searcher.semantic_code_search(vec_db_path, queries[0])
    
    print("\n" + "=" * 70)
    print("TEST 4: Batch Search (5 queries)")
    print("=" * 70)
    CodeSearch.clear_cache()
    batch_results = searcher.search_batch(vec_db_path, queries)
    for q, r in batch_results.items():
        print(f"   '{q[:30]}...': {len(r)} results")
    
    print("\n" + "=" * 70)
    print("STATS")
    print("=" * 70)
    stats = CodeSearch.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)