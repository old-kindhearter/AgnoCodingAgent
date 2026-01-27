import os
import hashlib

from pathlib import Path
from collections import Counter
from typing import List

# Agno / Phidata imports
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.document.base import Document

from agno.tools import Toolkit
from dotenv import load_dotenv

# AST Splitting imports
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

try:
    # Try relative import (when used as a module)
    from .embedder_factory import EmbedderFactory
except ImportError:
    # Fall back to absolute import (when run directly)
    from embedder_factory import EmbedderFactory


class CodeVectorStore(Toolkit):
    def __init__(self):
        super().__init__(name="build_vector_base", tools=[self.build_vector_base])

        self.vector_db_dir = "../AgnoCodingAgent/Knowledge/vector_db"  # 直接定死，不要给模型改。建议配置绝对路径。


    # 用来统计代码库的语言，以确定分chunk方式。
    def _map_ext_to_lang(self, ext: str) -> Language:
        """扩展名 -> LangChain Language 枚举映射"""
        mapping = {
            ".py": Language.PYTHON,
            ".js": Language.JS,
            ".jsx": Language.JS,
            ".ts": Language.JS,  # TS 通常兼容 JS 切分逻辑
            ".tsx": Language.JS,
            ".java": Language.JAVA,
            ".go": Language.GO,
            ".cpp": Language.CPP,
            ".c": Language.CPP,
            ".h": Language.CPP,
            ".cs": Language.CSHARP,
            ".rs": Language.RUST,
            ".md": Language.MARKDOWN,
            ".sol": Language.SOL, # Solidity
            ".rb": Language.RUBY,
            ".php": Language.PHP,
        }
        return mapping.get(ext, Language.PYTHON) # 未知类型默认用Python


    def _get_files(self) -> List[str]:
        """[内部方法] 高性能获取仓库代码文件 (O(1) 遍历 + 剪枝 + 熔断)"""
        
        # 1. 使用 Set 进行 O(1) 查找，比 List 快
        # 包含了常见的代码和文档格式
        target_extensions = {
            ".py", ".js", ".ts", ".java", ".go", ".cpp", ".c", ".h", ".rs", 
            ".md", ".txt", ".rst", ".yaml", ".yml", ".json"
        }
        
        # 2. 目录黑名单 - 只要遇到这些目录，直接跳过，绝不进入
        # 注意：这里使用的是目录名，而不是完整路径
        ignore_dirs = {
            'node_modules', 'dist', 'build', 'venv', '.git', '.idea', '.vscode', 
            '__pycache__', 'migrations', 'target', 'bin', 'obj', 'lib', 'vendor', 
            'docker'
        }
        
        # 3. 单文件大小阈值 (例如 100KB)
        # 超过这个大小的代码文件通常是机器生成的
        max_file_size_bytes = 100 * 1024 
        all_files, ext_counter = [], Counter()
        
        # os.walk 只需要遍历一次磁盘
        for root, dirs, files in os.walk(self.repo_path):
            # [关键优化]：原地修改 dirs 列表
            # os.walk 会根据 dirs 列表决定下一步进入哪些子目录。
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            
            for file in files:
                # 1. 扩展名快速检查
                _, ext = os.path.splitext(file)
                if ext not in target_extensions:
                    continue
                
                # 2. 排除隐藏文件
                if file.startswith('.'):
                    continue
                full_path = os.path.join(root, file)
                
                # 3. 文件大小检查
                # 如果文件太大，stat 也是瞬间完成的，比读取内容快得多
                try:
                    file_size = os.path.getsize(full_path)
                    if file_size > max_file_size_bytes:
                        # 可以在这里打印个 log，方便 debug
                        # print(f"Skipping large file: {file} ({file_size/1024:.2f} KB)")
                        continue
                except OSError:
                    continue
                
                ext_counter[ext] += 1
                all_files.append(full_path)
            
        if ext_counter:
            # 获取出现次数最多的后缀，例如: ('.py', 150)
            most_common_ext, count = ext_counter.most_common(1)[0]
            
            # 映射为 Language 枚举
            self._language = self._map_ext_to_lang(most_common_ext)
            
            # 计算占比（可选，用于日志）
            total_valid_files = sum(ext_counter.values())
            ratio = count / total_valid_files
            
            print(f"Language Detection: Main='{self._language.value}' "
                f"(Based on {most_common_ext}: {count}/{total_valid_files} files, {ratio:.1%})")
        
        else:
            print("No valid code files found, defaulting to Python.")
            self._language = Language.PYTHON
                

        print(f"文件扫描完成，共找到 {len(all_files)} 个有效代码文件。")
        return all_files

    def _load_and_split_ast(self) -> List[Document]:
        """核心逻辑：动态 AST 切分 + 上下文注入 + 大块切分策略"""
        files = self._get_files()  # 获取代码文件
        documents = []
        
        print(f"Found {len(files)} files. Starting optimized ingestion...")
        # 统计数据，用于优化、去重
        total_chunks = 0
        hash_set = set()

        for file_path in files:
            try:
                # 1. 获取文件类型
                _, list_ext = os.path.splitext(file_path)
                
                # 2. 读取文件
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # 3. 针对不同语言使用专门的分隔符（比如 Python 优先按 def/class 切，JS 按 function 切）
                current_lang = self._language
                
                # 4. chunk_size: 2000 (约 500-800 行代码)。我们希望尽可能把一个完整的 Class 或函数放在一个块里。
                # chunk_overlap: 200。保持上下文连续性。
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=current_lang,
                    chunk_size=2000, 
                    chunk_overlap=200
                )
                raw_chunks = splitter.create_documents([content])

                # 过滤掉太短的内容，例如空行、注释
                raw_chunks = [raw_chunk.page_content for raw_chunk in raw_chunks if len(raw_chunk.page_content)>=10]
                
                for i, page_content in enumerate(raw_chunks):
                    # 将路径作为上下文注入
                    enriched_content = f"// File: {file_path}\n// Part: {i}\n\n{page_content}"
                    page_hash = hashlib.md5(enriched_content.encode()).hexdigest()
                    if page_hash in hash_set:  # 去重
                        continue  # 跳过
                    else:
                        hash_set.add(page_hash)

                    doc = Document(
                        name=f"{file_path}:{i}",
                        id=f"{self.repo_path}:{file_path}:{i}",
                        content=enriched_content,  # 搜索时，模型能直接看到文件名
                        meta_data={
                            # "file_path": self.repo_path,
                            "file_name": file_path,
                            "chunk_index": i,
                            "language": list_ext,
                        }
                    )
                    documents.append(doc)
                    total_chunks += 1
            
            except Exception as e:
                # 生产环境建议用 logging.warning
                print(f"Error processing {file_path}: {e}")

        print(f"AST Split completed. Total vectors generated: {len(documents)}")
        self.vector_db.insert(self.repo_name, documents)

        return documents

    def _index_exists(self) -> bool:
        """[内部方法] 检查集合中是否已经有数据"""
        if Path(self.final_path).exists():
            file_count = sum(1 for item in Path(self.final_path).iterdir() if item.is_file())
            if file_count != 1:
                print(
                    f"目标路径已存在: {self.final_path}。"
                    f"请删除该目录或选择其他路径。"
                )
                return True  # Path exists, so index exists
        return False  # Path doesn't exist, so index doesn't exist
        

    def build_vector_base(self, repo_path: str) -> str:
        """
        按照语法树规则对本地github仓库的代码进行分chunk，并构建向量知识库
        Args:
            repo_path(str): 本地github仓库的绝对路径
        Returns:
            str: 返回本地github仓库对应的向量化数据库的绝对路径
        """
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        self.repo_path = os.path.abspath(repo_path)
        self.repo_name = os.path.basename(self.repo_path)

        self.final_path = os.path.join(self.vector_db_dir, self.repo_name)

        print(f"Initializing codebase build for: {self.repo_name}")
        
        # ====初始化====
        # 1. 使用中心化的 Embedder Factory（单例模式，避免重复加载）
        print("Initializing embedder...")
        self.embedder = EmbedderFactory.get_embedder()

        # 2. 配置 ChromaDB (自动持久化到文件夹)
        # 先检测数据库是否存在
        if self._index_exists():
            print(f"Knowledge base for '{self.repo_name}' already exists. Skipping build.")
            return str(self.final_path)
        
        # ChromaDB 会在 persist_dir 下创建 sqlite3 文件和二进制索引
        self.vector_db = ChromaDb(
            collection=self.repo_name, 
            embedder=self.embedder,
            persistent_client=True, # 启用持久化客户端
            path=self.final_path # 指定存储路径
        )
        self.vector_db.create()

        # 2. 生成文档
        print("Parsing and splitting code...")
        docs = self._load_and_split_ast()
        
        if not docs:
            print("No documents generated. Check your repo path.")
            return 'No documents generated.'
        return str(self.final_path)


# --- 使用示例 ---
if __name__ == "__main__":
    load_dotenv()
    # 配置
    TARGET_REPO = "/Users/junwei/Personal/gdiist/AgnoCodingAgent" # 替换为你的目标仓库路径
    
    # 初始化
    # 这一步会自动创建 ./local_knowledge_db 文件夹并在里面生成 chroma.sqlite3
    vector_store = CodeVectorStore()
    
    # 核心构建 (第一次运行会跑进度条，第二次运行会直接跳过)
    vector_store.build_vector_base(TARGET_REPO)