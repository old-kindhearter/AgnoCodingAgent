import os
from agno.tools import Toolkit
from agno.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from dotenv import load_dotenv

# Import the centralized embedder factory
try:
    # Try relative import (when used as a module)
    from .embedder_factory import EmbedderFactory
except ImportError:
    # Fall back to absolute import (when run directly)
    from embedder_factory import EmbedderFactory


class CodeSearch(Toolkit):
    def __init__(self):
        super().__init__(name="semantic_code_search", tools=[self.semantic_code_search])


    def semantic_code_search(self, vec_repo_path: str, query: str) -> list[str]:
        """
        检索本地的代码库
        Args:
            vec_repo_path(str): 要检索本地向量数据库绝对路径
            query(str): 待检索的相关话题
        Returns: 
            list[str]: 返回字符串数组，其中包含了所有检索结果的代码内容。
        """
        self.vec_repo_path = os.path.abspath(vec_repo_path)
        self.repo_name = os.path.basename(self.vec_repo_path)

        print(f"Searching in vector database: {self.repo_name}")
        print(f"Database path: {self.vec_repo_path}")
        
        # 使用中心化的 Embedder Factory（单例模式，避免重复加载）
        print("Initializing embedder...")
        self.embedder = EmbedderFactory.get_embedder()

        # 初始化 ChromaDB（连接到已存在的数据库）
        self.vector_db = ChromaDb(
            collection=self.repo_name,
            persistent_client=True,  # 启用持久化客户端
            embedder=self.embedder, 
            path=self.vec_repo_path  # 指定存储路径
        )
        
        # 初始化 Knowledge 对象用于检索
        self.knowledge = Knowledge(
            name="Github Code Database", 
            vector_db=self.vector_db,
            max_results=10,  # 最多返回10个结果
        )

        print(f"🔎 Query: '{query}'")
        print("⏳ Searching...")
        
        # 执行搜索
        search_results = self.knowledge.search(query=query, max_results=5)
        
        # 提取结果内容
        results = [res.content for res in search_results]
        
        print(f"Found {len(results)} relevant code chunks")
        
        return results


if __name__ == "__main__":
    load_dotenv()
    
    # 测试搜索功能
    print("=" * 60)
    print("Testing CodeSearch...")
    print("=" * 60)
    
    test_search = CodeSearch()
    
    # 指向你刚刚创建的向量数据库
    vec_db_path = '/Users/junwei/Personal/gdiist/Knowledge/vector_db/AgnoCodingAgent'
    
    # 测试查询
    test_query = 'how does the embedder factory work'
    
    try:
        results = test_search.semantic_code_search(vec_db_path, test_query)
        
        print("\n" + "=" * 60)
        print("Search Results:")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            # 只显示前200个字符，避免输出过长
            preview = result[:200] + "..." if len(result) > 200 else result
            print(preview)
        
        print("\n" + "=" * 60)
        print(f"Test completed! Found {len(results)} results")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Error: {e}")
        print("=" * 60)
        raise