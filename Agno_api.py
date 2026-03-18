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

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="My Team App")

# 配置数据库以保持 session 记忆
db = SqliteDb(db_file="agent.db")

# 创建 Team，启用历史记录
STORAGE_DB_PATH = os.path.join(os.path.dirname(__file__), "tmp", "agno_agentos_sessions.db")
os.makedirs(os.path.dirname(STORAGE_DB_PATH) or "tmp", exist_ok=True)

team_db = SqliteDb(db_file=STORAGE_DB_PATH)

STORAGE_DB_PATH = os.path.join(os.path.dirname(__file__), "tmp", "agno_agentos_sessions.db")
os.makedirs(os.path.dirname(STORAGE_DB_PATH) or "tmp", exist_ok=True)

team_db = SqliteDb(db_file=STORAGE_DB_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown event handler.
    Runs BEFORE the server starts accepting requests.
    """
    # ===== STARTUP =====
    print("=" * 60)
    print("WARMING UP SEARCH ENGINE")
    print("=" * 60)
    
    try:
        # Preload embedding model (500MB, ~5-10 seconds)
        semantic_code_search_optimized.CodeSearch.warmup()
        print("Search engine ready!")
    except Exception as e:
        print(f"Warmup error: {e}")
        print("Model will load on first query.")
    
    print("=" * 60)
    
    yield  # Server runs here
    
    # ===== SHUTDOWN =====
    print("Shutting down...")
    semantic_code_search_optimized.CodeSearch.clear_cache()

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
        "7. **直接返回搜索结果**：你不需要对代码做任何改动，也不需要做总结或者分析。", 
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


@app.post("/continue/v1/chat/completions")
async def continue_chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(content={"error": "No messages provided"}, status_code=400)
    
    model = body.get("model", "agno-team")
    stream = body.get("stream", False)
    
    # 【处理状态冲突】: Continue 是无状态的，它会发来完整的历史。
    # 我们把它的历史拼装成当前 Prompt，然后发给模型。
    if len(messages) > 1:
        history_context = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        user_input = f"【历史对话记录】\n{history_context}\n\n【当前请求】\n{messages[-1]['content']}"
    else:
        user_input = messages[-1]["content"]
    
    # 【核心修复】: 强制每次请求使用全新的 session_id，彻底禁用 Agno 对该接口的记忆，防止上下文套娃爆炸！
    session_id = f"continue-stateless-{uuid.uuid4().hex}"
    user_id = "continue-user"
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    # 处理流式请求
    async def generate():
        # (1) 发送初始头
        init_data = {
            'id': chat_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model,
            'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]
        }
        yield f"data: {json.dumps(init_data)}\n\n"
        # (2) 流式截取
        async for chunk in LeadArchitect.arun(
            input=user_input,
            session_id=session_id,  # 传入随机无状态 Session
            user_id=user_id,
            stream=True,
        ):
            content_to_yield = ""
            # 兼容 Agno 的各种变态返回格式
            if isinstance(chunk, str):
                # 如果是纯文本，直接采纳
                content_to_yield = chunk
            elif hasattr(chunk, 'content'):
                # 如果是对象，检测是否是 RunCompleted 事件
                event_type = str(getattr(chunk, 'event', '')).lower()
                # 【核心修复】：拦截最终的总结输出，防止在页面最底下又把全文重打一遍
                if 'completed' in event_type:
                    continue
                content_to_yield = chunk.content or ""
            if content_to_yield:
                chunk_data = {
                    'id': chat_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model,
                    'choices': [{'index': 0, 'delta': {'content': content_to_yield}, 'finish_reason': None}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # (3) 发送结束标识
        stop_data = {
            'id': chat_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model,
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
        }
        yield f"data: {json.dumps(stop_data)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


agent_os = AgentOS(
    teams=[LeadArchitect],
    base_app=app,
)
app = agent_os.get_app()

# sudo systemctl restart fastapi-plugin.service

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)