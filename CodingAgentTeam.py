import os, sys, uuid, json, time

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.agent import Agent
from agno.team import Team
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS

from tools import clone_github_repo
from tools import web_search
from tools import build_vector_base_parallel
from tools import semantic_code_search_optimized
from tools import find_repo_exist
from agno.skills import Skills, LocalSkills

from dotenv import load_dotenv


load_dotenv()

# 创建 Team，启用历史记录
STORAGE_DB_PATH = os.path.join(os.path.dirname(__file__), "tmp", "agno_agentos_sessions.db")
os.makedirs(os.path.dirname(STORAGE_DB_PATH) or "tmp", exist_ok=True)

team_db = SqliteDb(db_file=STORAGE_DB_PATH)

# 创建skill
skills = Skills(loaders=[LocalSkills("/path/to/skills")])

RepoAgent = Agent(
    id='repo_agent',
    name="RepoAgent",
    model=DeepSeek(id="deepseek-chat"),
    tools=[
        web_search.WebSearcher(),
        clone_github_repo.GitClone(), 
        #build_vector_base.CodeVectorStore()
        build_vector_base_parallel.CodeVectorStore()
    ],
    description="""
    你是代码仓库管理员，负责管理本地的两个仓库目录**/srv/AgnoCodingAgent/Knowledge/codebase**和**/srv/AgnoCodingAgent/Knowledge/vector_db**。
    如果需要进行在线搜索，找到对应仓库的github链接，并将仓库clone到第一个目录下；同时将代码转化为向量数据库，保存到第二个目录下，以便后续检索。
    注意：本地目录下新建的仓库，都以github url里包含的原始名字来命名，不要添加任何额外的前缀或后缀，也不要改变原始名字里面的大小写。
    例如对于**https://github.com/somebody/aaBcD**，应该在本地创建名为**/srv/AgnoCodingAgent/Knowledge/codebase/aaBcD**的目录。""",
    instructions=[
        "1. 使用**web_search**工具进行互联网检索，确保找到的仓库是符合用户要求的，得到对应的URL。这个准确的URL给到**clone_github_repo**。"
        "2. 使用**clone_github_repo**工具将在线的gitHub仓库存储于本地，得到该仓库在本地的绝对路径。这个准确的绝对路径给到**build_vector_base**。",
        "3. 使用**build_vector_base**工具将本地的github仓库转换为向量数据库，得到这个向量数据库在本地的绝对路径。",
        "4. 将这个向量数据库的绝对路径明确告诉你的同事 [LeadArchitect]和[CodeSearchAgent]，确保他知道去哪里做向量检索。"
    ],
)

CodeSearchAgent = Agent(
    id='code_sarch_agent',
    name="CodeSearchAgent",
    model=DeepSeek(id="deepseek-chat"),
    tools=[semantic_code_search_optimized.CodeSearch()],  # 假设这个工具返回 Top-30 的原始结果，包含大量冗余
    description="""
    你是负责搜索与简单审查的初级代码工程师，根据意图在向量数据库中进行语义的代码检索。""",
    instructions=[
        "1. 你的工作区仅限于绝对路径`/srv/AgnoCodingAgent/Knowledge/vector_db`下的文件目录，根据 [LeadArchitect]或者[RepoAgent] 向你提供的本地向量化的代码库的名称在该路径进行搜索。", 
        "2. **搜索策略选择**: 你有两种搜索方法可用，无论哪种在调用前都需要仔细思考，避免浪费搜索次数："
        "   - `semantic_code_search(vec_repo_path, query, max_results)`: 单次查询，适合探索性搜索或后续精确查询"
        "   - `semantic_code_batchsearch(vec_repo_path, queries, max_results)`: 批量查询，适合需要同时搜索多个相关概念的场景"
        "   **使用原则**:"
        "   - 如果需要搜索 3+ 个相关但不同的概念（如'authentication', 'authorization', 'session management'），使用 `semantic_code_batchsearch` 一次性查询"
        "   - 如果只是单个查询或需要根据前一次结果动态调整查询，使用 `semantic_code_search`"
        "   - 如果用户问题明确包含多个子问题（如'how does auth and db connection work'），拆分为多个查询词并使用 `semantic_code_batchsearch`",
        "3. 根据用户意图或者主程的反馈，在数据库中进行广泛搜索，仔细思考符合要求的关键词组合（如函数名、类名、功能描述等）。",
        "4. **相关性过滤**: 判断哪些片段是真正有用的，哪些是噪音（如测试用例、无关的工具函数）。剔除噪音。",
        "5. **上下文压缩**: 对于长函数，如果不是核心逻辑，提取其函数签名（Signature）和文档字符串（Docstring）即可。", 
        "6. **只保留最相关片段**：你的任务是把搜索到的最相关的代码片段完整交给 [LeadArchitect]。",
    ],
)

LeadArchitect = Team(
    id='lead_architect_agent', 
    name='LeadArchitect',
    members=[RepoAgent, CodeSearchAgent],
    tools=[find_repo_exist.FindRepoPath()], 
    model=DeepSeek(id="deepseek-chat"), # Team Leader 依然是主程
    description="""
    这是一个优化的基于检索增强生成技术的编程团队，通常的工作场景是基于用户指明的某个代码库进行开发工作。你是这个编程团队的主程，负责最终向用户输出解决方案。
    你需要首先理解用户的意图，将任务拆解分配给团队成员们，你可以先调用工具**find_repo_exist**判断用户指明的代码库是否已存在本地，如果存在则获得该仓库绝对路径，并交给 [CodeSearchAgent] 进行语义搜索；
    否则则让 [RepoAgent] 搜索这个代码库的github url，并本地向量化，将 [RepoAgent] 返回的绝对路径交给 [CodeSearchAgent] 进行语义搜索。""",
    instructions=[
        "1. 你需要理解用户的意图，将任务拆解分配给团队成员们。", 
        "2. 如果是首次对话或者用户提到了**新的代码库**时按照下列流程进行调度：首先让 [RepoAgent] 搜索这个代码库的github url并本地向量化，将 [RepoAgent] 返回的绝对路径交给 [CodeSearchAgent] 进行语义搜索。",
        "3. 否则则先调用工具**find_repo_exist**判断用户指明的代码库是否已存在本地，如果存在则获得该仓库绝对路径，并交给 [CodeSearchAgent] 进行语义搜索；", 
        "4. 否则则让 [RepoAgent] 搜索这个代码库的github url，并本地向量化，将 [RepoAgent] 返回的绝对路径交给 [CodeSearchAgent] 进行语义搜索。",
        "5. 在请求 [CodeSearchAgent] 去搜索数据库时，你必须**显式告知 [CodeSearchAgent] 搜索的绝对路径**（由**find_repo_exist**返回或者 [RepoAgent] 提供）。",
        "6. 如果搜索结果实在缺少符合用户提问的内容细节，你可以再次追问 [CodeSearchAgent] 请求更多具体的信息。",
        "7. 你的回答必须具备专家级水准，保持输出简洁，只列出最重要、相关性最高的信息，禁止分点太多产生相关性低的信息，注重代码风格和最佳实践。"
    ],
    # debug_mode=True,  # 调试模式可以看到中间 Agent 的所有交互
    db=team_db,
    add_history_to_context=True,
    num_history_runs=5,
    share_member_interactions=True,
    # add_team_history_to_members=True, 
)
