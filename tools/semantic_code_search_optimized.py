import os
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
import chromadb
from agno.tools import Toolkit
from .embedder_factory import EmbedderSingleton

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
    def __init__(self):
        super().__init__(name="semantic_code_search", tools=[self.semantic_code_search, self.semantic_code_batchsearch])
    
    def semantic_code_search( 
        self, 
        vec_repo_path: str, 
        query: str,
        max_results: int = 5
    ) -> List[str]:
        """
        在本地向量数据库中进行语义代码检索，返回最相关的代码片段。
        支持自然语言查询和技术概念搜索。
        
        Args:
            vec_repo_path(str): 本地向量数据库的绝对路径
            query(str): 待检索的查询内容，支持自然语言或技术术语
            max_results(int): 返回的最大结果数量，默认5条，建议范围3-10
        
        Returns:
            list[str]: 返回字符串数组，每个字符串包含：
                    - 文件路径标注 (// File: path/to/file.py)
                    - 代码块序号 (// Part: 0)
                    - 完整的代码内容（尽可能包含完整函数/类）
        使用建议:
        - 使用具体的技术术语能获得更好的结果
        - 适合单一概念的精确查询
        - 如需搜索3+个相关概念，建议使用 semantic_code_batchsearch
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
        批量检索多个相关概念，适合需要同时探索多个方面的场景。
        
        Args:
            vec_repo_path(str): 本地向量数据库的绝对路径
            queries(List[str]): 待检索的查询列表
                            例如: ["身份认证", "权限控制", "会话管理"]
            max_results(int): 每个查询返回的最大结果数，默认5条
        
        Returns:
            Dict[str, List[str]]: 字典，键为原始查询，值为对应的代码片段列表
        
        使用场景:
        - 需要同时搜索3个以上相关概念
        - 从多个角度探索同一功能特性
        - 用户问题包含多个子话题（如"认证和数据库连接如何工作"）
        
        不适用的场景：
        - 单个查询（用 semantic_code_search 更简单）
        - 需要根据前一次结果动态调整查询（用 semantic_code_search 迭代）
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
        
        print(f"Warmup complete in {time.perf_counter()-start:.2f}s")
    
    @staticmethod
    def clear_cache():
        """Clear all caches"""
        ChromaDBConnection.clear()
        print("Caches cleared")