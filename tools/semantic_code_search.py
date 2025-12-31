import os
from agno.tools import Toolkit
from agno.knowledge.embedder.vllm import VLLMEmbedder
from agno.knowledge.embedder.jina import JinaEmbedder
from agno.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from dotenv import load_dotenv

class CodeSearch(Toolkit):
    def __init__(self):
        super().__init__(name="semantic_code_search", tools=[self.semantic_code_search])
        self.api_key = os.getenv("JINA_API_KEY")
        
        assert self.api_key is not None, "JINA_API_KEY is not set"


    def semantic_code_search(self, vec_repo_path: str, query: str)->list[dict]:
        """
        检索本地的代码库
        Args:
            vec_repo_path(str): 要检索本地向量数据库绝对路径
            query(str): 待检索的相关话题
        Returns: 
            list[dict]: 返回字典数组，其中包含了所有检索结果的相关信息。
        """
        self.vec_repo_path = os.path.abspath(vec_repo_path)
        self.repo_name = os.path.basename(self.vec_repo_path)

        # self.embedder = JinaEmbedder(
        #     id="jina-embeddings-v3",
        #     dimensions=1024,
        #     embedding_type="float",
        #     late_chunking=True,
        #     batch_size=50,
        #     timeout=30.0, 
        #     api_key=self.api_key
        # )
        self.embedder = VLLMEmbedder(
            id='Qwen/Qwen3-Embedding-0.6B',  # 建议配置绝对路径。
            dimensions=1024, 
            batch_size=512
        )

        self.vector_db = ChromaDb(
            collection=self.repo_name,
            persistent_client=True, # 启用持久化客户端
            embedder=self.embedder, 
            path=self.vec_repo_path # 指定存储路径
        )
        
        self.knowledge = Knowledge(
            name="Github Code Database", 
            vector_db=self.vector_db,
            max_results=10, 
        )

        results = [res.content for res in self.knowledge.search(query=query, max_results=5)]
        
        return results


if __name__ == "__main__":
    load_dotenv()
    test = CodeSearch()
    print(test.semantic_code_search('/workspace/ai-test/AgentPractice/Knowledge/vector_db/trl', 'PPO如何实现'))
