"""
Agno Team API - 带记忆的 Continue 插件集成方案

1. 使用 add_history_to_context=True
2. 使用 num_history_runs=N
3. 在 team.run() 中传入 session_id 以实现持久化

使用方法：
    uvicorn Agno_api_with_memory:app --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn
import time
import json
import os
import hashlib
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 引入 Agno 组件
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from dotenv import load_dotenv

from agno.agent import Agent
from agno.team import Team
from agno.models.deepseek import DeepSeek
from agno.models.openrouter import OpenRouter
from agno.db.sqlite import SqliteDb

load_dotenv()

from tools import build_vector_base
from tools import semantic_code_search
from tools import clone_github_repo
from tools import web_search
from tools import build_vector_base_parallel
from tools import semantic_code_search_optimized

load_dotenv()
API_HOST = "0.0.0.0"
API_PORT = 8000

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for startup/shutdown events.
    
    On startup: Downloads and loads the embedding model.
    On shutdown: Clears caches.
    """
    # Startup
    print("=" * 60)
    print("STARTING AGNO API SERVER")
    print("=" * 60)
    
    # Warmup embedding model (no specific DB - that's user-dependent)
    try:
        print("Warming up embedding model...")
        semantic_code_search_optimized.CodeSearch.warmup()  # Just load the model
        print("Search engine ready!")
    except Exception as e:
        print(f"Warmup error: {e}")
        print("Model will load on first query.")
    
    print("=" * 60)
    print(f"API ready at http://{API_HOST}:{API_PORT}")
    print("=" * 60)
    
    yield  # Server runs here
    
    # Shutdown
    print("Shutting down...")
    semantic_code_search_optimized.CodeSearch.clear_cache()


class agno_team():
    def __init__(self):
        # ==============================================================================
        # Role 1: Repo search (运维)
        # 模型: DeepSeek-V3 (极高性价比)
        # 职责: 脏活累活，不涉及复杂推理
        # ==============================================================================
        self.RepoAgent = Agent(
            id='repo_agent',
            name="RepoAgent",
            model=DeepSeek(id="deepseek-chat"),
            tools=[
                web_search.WebSearcher(),
                clone_github_repo.GitClone(), 
                #build_vector_base.CodeVectorStore()
                build_vector_base_parallel.ParallelCodeVectorStore()
            ],
            description="""
            你是代码仓库管理员，负责管理本地的两个仓库目录。
            如果需要进行在线搜索，找到对应仓库的github链接，并将仓库clone到本地。""",
            instructions=[
                "1. 使用**web_search**工具进行互联网检索，确保找到的仓库是符合用户要求的。",
                "2. 使用**clone_github_repo**工具将在线的gitHub仓库存储于本地。",
                "3. 使用**build_vector_base**工具将本地的github仓库转换为向量数据库。",
            ],
            add_history_to_context=True,
        )

        # ==============================================================================
        # Role 2: The Scout (代码侦查员)
        # 模型: DeepSeek-V3 (处理速度快，Cheap)
        # 职责: 广度搜索。
        # ==============================================================================
        self.CodeSearchAgent = Agent(
            id='code_search_agent',
            name="CodeSearchAgent",
            model=DeepSeek(id="deepseek-chat"),
            tools=[semantic_code_search_optimized.CodeSearch()],  # 假设这个工具返回 Top-30 的原始结果，包含大量冗余
            description="""
            你是负责搜索与简单审查的初级代码工程师，根据意图在向量数据库中进行语义的代码检索。""",
            instructions=[
                "1. 你的工作区仅限于 [RepoAgent] 向你提供的本地向量化的代码库的绝对路径。",
                "2. 根据用户意图或者主程的反馈，在数据库中进行广泛搜索。",
                "3. **相关性过滤**: 判断哪些片段是真正有用的，哪些是噪音。",
                "4. **上下文压缩**: 对于长函数，提取其函数签名和文档字符串即可。",
            ],
            add_history_to_context=True,
        )

        # ==============================================================================
        # Team 定义 - 带持久化存储
        # ==============================================================================
        self.LeadArchitect = Team(
            id='lead_architect_agent',
            name='LeadArchitect',
            members=[self.RepoAgent, self.CodeSearchAgent],
            model=DeepSeek(id="deepseek-chat"),
            description="""
            这是一个优化的基于检索增强生成技术的编程团队。你是这个编程团队的主程，负责最终向用户输出解决方案。
            你需要先将任务拆解分配给团队成员们。""",
            instructions=[
                "1. 你需要首先理解用户的意图",
                "2. 如果涉及到了新的代码库，请求 [RepoAgent] 去在线获取新的github仓库。",
                "3. 请求 [CodeSearchAgent] 去搜索数据库。",
                "4. 如果分析师的简报中缺少细节，你可以再次追问 [CodeSearchAgent] 具体的细节。",
                "5. 你的回答必须具备专家级水准，保持简洁，注重代码风格和最佳实践。"
            ],
            # === 关键修复：添加持久化存储 ===
            db=team_db,
            add_history_to_context=True,
            num_history_runs=10,
            share_member_interactions=True,
        )


agno_instance = agno_team()
team_instance = agno_instance.LeadArchitect

# --- 定义 OpenAI 兼容的数据结构 ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "agno-team"
    messages: List[Message]
    stream: bool = False

app = FastAPI(lifespan=lifespan)

# --- 流式响应生成器 ---
def generate_openai_stream(content_generator):
    chat_id = f"chatcmpl-{int(time.time())}"
    
    # 包装成 OpenAI 的 chunk 格式
    try:
        for chunk in content_generator:
            delta_content = chunk if isinstance(chunk, str) else getattr(chunk, "content", str(chunk))
            
            response_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "agno-team",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta_content},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(response_data)}\n\n"
        
        # 发送结束标志
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# --- 实现接口 ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    print(f"收到请求: {request.messages[-1].content[:50]}...")
    
    # 获取用户最新的输入
    user_input = request.messages[-1].content
    
    # === 关键修复：生成稳定的 session_id ===
    # 基于 system prompt 哈希，为每个工作区创建稳定的会话
    system_messages = [m for m in request.messages if m.role == "system"]
    if system_messages:
        system_content = system_messages[0].content[:500]
        session_hash = hashlib.md5(system_content.encode()).hexdigest()[:16]
        session_id = f"continue_{session_hash}"
    else:
        session_id = "continue_default"
    
    user_id = "continue_user"
    
    if request.stream:
        # === 关键修复：传入 session_id 和 user_id ===
        stream_gen = team_instance.run(
            user_input,
            stream=True,
            session_id=session_id,
            user_id=user_id,
        )
        return StreamingResponse(
            generate_openai_stream(stream_gen),
            media_type="text/event-stream",
        )
    else:
        # === 关键修复：传入 session_id 和 user_id ===
        response = team_instance.run(
            user_input,
            stream=False,
            session_id=session_id,
            user_id=user_id,
        )
        content = response.content if hasattr(response, "content") else str(response)
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = semantic_code_search_optimized.CodeSearch.get_stats()
    return {
        "status": "healthy",
        "search_engine": stats
    }

@app.post("/warmup")
async def manual_warmup(db_path: Optional[str] = None):
    """
    Manually trigger warmup.
    
    Args:
        db_path: Optional path to a vector database to preload.
                If not provided, only loads the embedding model.
    """
    try:
        semantic_code_search_optimized.CodeSearch.warmup(db_path)
        return {"status": "success", "db_path": db_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches."""
    semantic_code_search_optimized.CodeSearch.clear_cache()
    return {"status": "success", "message": "Caches cleared"}


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False)
