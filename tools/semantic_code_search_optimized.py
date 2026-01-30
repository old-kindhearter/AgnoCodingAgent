import os
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
import chromadb
from agno.tools import Toolkit
from embedder_factory import EmbedderSingleton

# Suppress Jina model warnings
import warnings
warnings.filterwarnings("ignore", message="Some weights of BertModel were not initialized")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")


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
        super().__init__(name="semantic_code_search", tools=[self.semantic_code_search, self.semantic_code_batchsearch])
    
    def semantic_code_search( 
        self, 
        vec_repo_path: str, 
        query: str,
        max_results: int = 5
    ) -> List[str]:
        """
        检索本地的代码库
        Args:
            vec_repo_path(str): 要检索本地向量数据库绝对路径
            query(str): 待检索的相关话题
            max_results: int = 5 检索结果的最大数量
        Returns: 
            list[str]: 返回字符串数组,其中包含了所有检索结果的代码内容。
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
    
    def semantic_code_batchsearch(
        self,
        vec_repo_path: str,
        queries: List[str],
        max_results: int = 5
    ) -> Dict[str, List[str]]:
        """
        批量检索本地的代码库
        Args:
            vec_repo_path(str): 要检索本地向量数据库绝对路径
            queries(List[str]): 待检索的相关话题
            max_results: int = 5 检索结果的最大数量
        Returns: 
            Dict[str, List[str]]: 返回包含了所有检索结果的代码内容。
        """
        start = time.perf_counter()
        vec_repo_path = os.path.abspath(vec_repo_path)
        
        # Check cache first
        results = {}
            
        # Get resources
        embedder = EmbedderSingleton.get()
        collection, _ = ChromaDBConnection.get_collection(vec_repo_path)
            
        # Batch embed - this IS faster than individual encoding
        embed_start = time.perf_counter()
        embeddings = embedder.encode_batch(queries)
        embed_time = (time.perf_counter() - embed_start) * 1000
            
        # Batch search
        search_start = time.perf_counter()
        raw_results = collection.query(
            query_embeddings=np.array(embeddings),
            n_results=max_results,
            include=["documents"]
        )
        search_time = (time.perf_counter() - search_start) * 1000
            
        # Process results
        for i, query in enumerate(queries):
            docs = raw_results["documents"][i] if raw_results["documents"] else []
            results[query] = docs
            
        print(f"Batch: {len(queries)} queries, embed={embed_time:.1f}ms, search={search_time:.1f}ms")
        
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