import uvicorn
import time
import json
import os, sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 引入 Agno 组件
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.models.deepseek import DeepSeek
from agno.team import Team

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import build_vector_base
from tools import semantic_code_search
from tools import clone_github_repo
from tools import web_search
from tools import build_vector_base_parallel

load_dotenv()

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
            你是代码仓库管理员，负责管理本地的两个仓库目录**/workspace/ai-test/AgentPractice/Knowledge/codebase**和**/workspace/ai-test/AgentPractice/Knowledge/vector_db**。
            如果需要进行在线搜索，找到对应仓库的github链接，并将仓库clone到第一个目录下；同时将代码转化为向量数据库，保存到第二个目录下，以便后续检索。
            注意：本地目录下新建的仓库，都以github url里包含的原始名字来命名，不要添加任何额外的前缀或后缀，也不要改变原始名字里面的大小写。
            例如对于**https://github.com/somebody/aaBcD**，应该在本地创建名为**/workspace/ai-test/AgentPractice/Knowledge/codebase/aaBcD**的目录。""",
            instructions=[
                "1. 使用**web_search**工具进行互联网检索，确保找到的仓库是符合用户要求的，得到对应的URL。这个准确的URL给到**clone_github_repo**。"
                "2. 使用**clone_github_repo**工具将在线的gitHub仓库存储于本地，得到该仓库在本地的绝对路径。这个准确的绝对路径给到**build_vector_base**。",
                "3. 使用**build_vector_base**工具将本地的github仓库转换为向量数据库，得到这个向量数据库在本地的绝对路径。",
                "4. 将这个向量数据库的绝对路径明确告诉你的同事 [CodeSearchAgent]，确保他知道去哪里做向量检索。"
            ],
            # show_tool_calls=True
            add_history_to_context=True, 
        )

        # ==============================================================================
        # Role 2: The Scout (代码侦查员)
        # 模型: DeepSeek-V3 (处理速度快，Cheap)
        # 职责: 广度搜索。
        # ==============================================================================
        self.CodeSearchAgent = Agent(
            id='code_sarch_agent',
            name="CodeSearchAgent",
            model=DeepSeek(id="deepseek-chat"),
            tools=[semantic_code_search.CodeSearch()],  # 假设这个工具返回 Top-30 的原始结果，包含大量冗余
            description="""
            你是负责搜索与简单审查的初级代码工程师，根据意图在向量数据库中进行语义的代码检索。""",
            instructions=[
                "1. 你的工作区仅限于 [RepoAgent] 向你提供的本地向量化的代码库的绝对路径，使用这个路径作为搜索参数。", 
                "2. 根据用户意图或者主程的反馈，在数据库中进行广泛搜索，可以尝试多种关键词组合（如函数名、类名、功能描述等）。",
                "3. **相关性过滤**: 针对搜索的query，判断哪些片段是真正有用的，哪些是噪音（如测试用例、无关的工具函数）。剔除噪音。",
                "4. **上下文压缩**: 对于长函数，如果不是核心逻辑，提取其函数签名（Signature）和文档字符串（Docstring）即可。", 
                "5. 不要试图回答问题，不要担心上下文太长，尽可能多收集但只保留最相关片段。你的任务是把搜索到的所有原始代码片段完整交给 [LeadArchitect]。",
            ],
            add_history_to_context=True, 
        )

        self.LeadArchitect = Team(
            id='lead_architect_agent', 
            name='LeadArchitect',
            members=[self.RepoAgent, self.CodeSearchAgent],
            model=DeepSeek(id="deepseek-chat"), # Team Leader 依然是主程
            description="""
            这是一个优化的基于检索增强生成技术的编程团队，通常的工作场景是基于用户指明的某个代码库进行开发工作。你是这个编程团队的主程，负责最终向用户输出解决方案。
            你需要先将任务拆解分配给团队成员们，首次对话或者用户提到了需要引入新的代码库进行开发时按照下列流程进行调度：首先让 [RepoAgent] 搜索这个代码库的github url并本地向量化，然后使用 [CodeSearchAgent] 进行语义搜索，再将结果返回给 [LeadArchitect]。
            在完成了前面的代码本地化之后，按照下列流程进行调度：如果 [LeadArchitect] 需要新的内容，让 [CodeSearchAgent] 进行语义搜索，再将结果返回给 [LeadArchitect]。""",
            instructions=[
                "1. 你需要首先理解用户的意图",
                "2. 如果涉及到了新的代码库，请求 [RepoAgent] 去在线获取新的github仓库。",
                "3. 请求 [CodeSearchAgent] 去搜索数据库。",
                "4. 如果分析师的简报中缺少细节，你可以再次追问 [CodeSearchAgent] 具体的细节。",
                "5. 你的回答必须具备专家级水准，保持简洁，注重代码风格和最佳实践。"
            ],
            # debug_mode=True,  # 调试模式可以看到中间 Agent 的所有交互
            share_member_interactions=True
        )

agno_instance = agno_team()
team_instance = agno_instance.LeadArchitect

# --- 2. 定义 OpenAI 兼容的数据结构 ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "agno-team"
    messages: List[Message]
    stream: bool = False

app = FastAPI()

# --- 3. 流式响应生成器 ---
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

# --- 4. 实现接口 ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    print(f"收到请求: {request.messages[-1].content[:50]}...")
    
    # 获取用户最新的输入
    user_input = request.messages[-1].content
    
    # 注意：Continue 会发送包含了 system prompt 和历史记录的 messages 列表
    # 简单的做法是只取最新一条给 Agno，让 Agno 自身管理记忆（如果配置了 Knowledge/Storage）
    # 或者你需要写逻辑把 messages 转换成 Agno 的 history
    
    if request.stream:
        # 调用 Agno 的流式运行方法
        stream_gen = team_instance.run(user_input, stream=True)
        return StreamingResponse(
            generate_openai_stream(stream_gen), 
            media_type="text/event-stream", 
        )
    else:
        # 非流式处理 (通常 Continue 默认用流式，但为了健壮性保留)
        response = team_instance.run(user_input, stream=False)
        content = response.content if hasattr(response, "content") else str(response)
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
