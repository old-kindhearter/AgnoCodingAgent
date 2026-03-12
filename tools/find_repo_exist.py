import json
import logging
import os
import requests
from dotenv import load_dotenv
from agno.tools import Toolkit
from agno.utils.log import logger

class FindRepoPath(Toolkit):
    def __init__(self):
        super().__init__(name="find_repo_exist", tools=[self.find_repo_exist])
        self.target_path = '/srv/AgnoCodingAgent/Knowledge/vector_db'

    def find_repo_exist(self, repo_name: str) -> str:
        """
        查找给定父目录下是否已经包含指定名称的 GitHub 仓库，如果存在则返回该仓库的绝对路径，然后使用 [CodeSearchAgent] 进行语义搜索；
        否则返回 None ，让 [RepoAgent] 搜索这个代码库的github url并本地向量化，然后使用 [CodeSearchAgent] 进行语义搜索。
        Args:
            repo_name(str): 待查找的仓库名称

        Returns:
            str: 该仓库的绝对路径的查询结果。
        """
        target_dir = os.path.expanduser(self.target_path)
        repo_path = os.path.join(target_dir, repo_name)
        
        if os.path.isdir(repo_path):
            return repo_path
        else:
            return None

def find_repo_exist(repo_name: str):
    finder = FindRepoPath()
    return finder.find_repo_exist(repo_name)

if __name__ == "__main__":
    load_dotenv()
    
    search_result = str(find_repo_exist("trl"))
    print(search_result)
