import os
import hashlib
import chromadb
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import Counter
from typing import List, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from agno.tools import Toolkit

import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    """Context manager for timing code blocks"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"⏱️  {name}: {elapsed:.2f}s")
    
@dataclass
class ChunkData:
    doc_id: str
    content: str
    file_name: str
    chunk_index: int
    language: str


def _process_file_worker(args: Tuple[str, str, Language]) -> List[ChunkData]:
    """Standalone worker - must be at module level for pickling"""
    file_path, repo_path, language = args
    chunks = []
    try:
        _, ext = os.path.splitext(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=2000, chunk_overlap=200
        )
        raw_chunks = splitter.create_documents([content])
        raw_chunks = [c.page_content for c in raw_chunks if len(c.page_content) >= 10]
        
        for i, page_content in enumerate(raw_chunks):
            enriched = f"// File: {file_path}\n// Part: {i}\n\n{page_content}"
            chunks.append(ChunkData(
                doc_id=f"{repo_path}:{file_path}:{i}",
                content=enriched,
                file_name=file_path,
                chunk_index=i,
                language=ext,
            ))
    except Exception as e:
        print(f"Error: {file_path}: {e}")
    return chunks


class ParallelCodeVectorStore(Toolkit):
    """
    Option 1: Bypass Agno for build, compatible with Agno for search.
    
    - Build: Direct ChromaDB + SentenceTransformers
    - Search: Agno Knowledge reads the same DB
    """
    
    def __init__(self, num_workers: int = None):
        super().__init__(name="build_vector_base", tools=[self.build_vector_base])
        self.vector_db_dir = "../AgnoCodingAgent/Knowledge/vector_db"
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.model_id = "jinaai/jina-embeddings-v2-base-code"
        self._language = Language.PYTHON
        self._model = None  # Lazy load

    def _get_model(self) -> SentenceTransformer:
        """Lazy load model (singleton within process)"""
        if self._model is None:
            print(f"Loading model: {self.model_id}")
            self._model = SentenceTransformer(self.model_id)
        return self._model

    def _map_ext_to_lang(self, ext: str) -> Language:
        mapping = {
            ".py": Language.PYTHON, ".js": Language.JS, ".ts": Language.JS,
            ".java": Language.JAVA, ".go": Language.GO, ".cpp": Language.CPP,
            ".c": Language.CPP, ".rs": Language.RUST, ".md": Language.MARKDOWN,
        }
        return mapping.get(ext, Language.PYTHON)

    def _get_files(self) -> List[str]:
        """Scan for code files (unchanged from original)"""
        target_extensions = {".py", ".js", ".ts", ".java", ".go", ".cpp", ".c", ".rs", ".md", ".txt"}
        ignore_dirs = {'node_modules', 'dist', 'build', 'venv', '.git', '__pycache__'}
        max_file_size = 100 * 1024
        
        all_files = []
        ext_counter = Counter()
        
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            for file in files:
                _, ext = os.path.splitext(file)
                if ext not in target_extensions or file.startswith('.'):
                    continue
                full_path = os.path.join(root, file)
                try:
                    if os.path.getsize(full_path) > max_file_size:
                        continue
                except OSError:
                    continue
                ext_counter[ext] += 1
                all_files.append(full_path)
        
        if ext_counter:
            most_common_ext, _ = ext_counter.most_common(1)[0]
            self._language = self._map_ext_to_lang(most_common_ext)
        
        print(f"Found {len(all_files)} files (primary: {self._language.value})")
        return all_files

    def _parallel_ast_split(self, files: List[str]) -> List[ChunkData]:
        """Phase 1: Parallel AST splitting"""
        print(f"Phase 1: Splitting with {self.num_workers} workers...")
        
        worker_args = [(f, self.repo_path, self._language) for f in files]
        all_chunks = []
        hash_set = set()
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(_process_file_worker, args): args[0] for args in worker_args}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    print(f"   Processed {completed}/{len(files)} files...")
                for chunk in future.result():
                    content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
                    if content_hash not in hash_set:
                        hash_set.add(content_hash)
                        all_chunks.append(chunk)
        
        print(f"Phase 1 complete: {len(all_chunks)} unique chunks")
        return all_chunks

    def _batch_embed(self, chunks: List[ChunkData]) -> List[List[float]]:
        """Phase 2: Batch embedding"""
        print(f"Phase 2: Embedding {len(chunks)} chunks...")
        
        model = self._get_model()
        texts = [chunk.content for chunk in chunks]
        
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        print(f"Phase 2 complete: {len(embeddings)} embeddings")
        return embeddings.tolist()

    def _direct_insert(self, chunks: List[ChunkData], embeddings: List[List[float]]):
        """Phase 3: Direct ChromaDB insert"""
        print(f"Phase 3: Inserting into ChromaDB...")
        
        # Use ChromaDB directly (not Agno)
        client = chromadb.PersistentClient(path=self.final_path)
        collection = client.get_or_create_collection(
            name=self.repo_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Batch insert to avoid memory issues
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            collection.add(
                ids=[c.doc_id for c in chunks[i:end]],
                embeddings=embeddings[i:end],
                documents=[c.content for c in chunks[i:end]],
                metadatas=[{
                    "file_name": c.file_name,
                    "chunk_index": c.chunk_index,
                    "language": c.language
                } for c in chunks[i:end]]
            )
            print(f"   Inserted {end}/{len(chunks)}")
        
        print(f"Phase 3 complete")

    def build_vector_base(self, repo_path: str) -> str:
        self.repo_path = os.path.abspath(repo_path)
        self.repo_name = os.path.basename(self.repo_path)
        self.final_path = os.path.join(self.vector_db_dir, self.repo_name)
        
        print(f"Building: {self.repo_name}")
        
        if Path(self.final_path).exists() and any(Path(self.final_path).iterdir()):
            print(f"Exists: {self.final_path}")
            return str(self.final_path)
        
        Path(self.final_path).mkdir(parents=True, exist_ok=True)
        
        total_start = time.perf_counter()
        
        # Phase 1: File scanning
        with timer("Phase 1a - File scanning"):
            files = self._get_files()
        
        if not files:
            return "No files found"
        
        # Phase 2: AST splitting
        with timer("Phase 1b - AST splitting (parallel)"):
            chunks = self._parallel_ast_split(files)
        
        if not chunks:
            return "No chunks generated"
        
        # Phase 3: Embedding
        with timer("Phase 2 - Embedding (batch)"):
            embeddings = self._batch_embed(chunks)
        
        # Phase 4: DB insert
        with timer("Phase 3 - ChromaDB insert"):
            self._direct_insert(chunks, embeddings)
        
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTOTAL TIME: {total_elapsed:.2f}s")
        print(f"   Files: {len(files)}, Chunks: {len(chunks)}")
        print(f"   Throughput: {len(chunks)/total_elapsed:.1f} chunks/sec")
        
        return str(self.final_path)