"""
最基本的记忆持久化测试

使用方法：
    python test_memory_basic.py
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

from agno.agent import Agent
from agno.team import Team
from agno.models.deepseek import DeepSeek
from agno.db.sqlite import SqliteDb

# 清理旧数据库
TEST_DB = "./tmp/test.db"
os.makedirs("./tmp", exist_ok=True)
if os.path.exists(TEST_DB):
    os.remove(TEST_DB)

# 创建 Team
db = SqliteDb(db_file=TEST_DB)

agent = Agent(
    name="TestAgent",
    model=DeepSeek(id="deepseek-chat"),
    role="记住用户告诉你的信息",
)

team = Team(
    name="TestTeam",
    members=[agent],
    model=DeepSeek(id="deepseek-chat"),
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
)

SESSION_ID = "test_session"

# 测试 1: 告诉信息
print("消息 1: 告诉 Team 我的名字")
r1 = team.run("我叫小明，请记住。", session_id=SESSION_ID, stream=False)
print(f"回复: {r1.content}\n")

time.sleep(1)

# 测试 2: 询问信息
print("消息 2: 询问 Team 我的名字")
r2 = team.run("我叫什么名字？", session_id=SESSION_ID, stream=False)
print(f"回复: {r2.content}\n")

# 验证
if "小明" in r2.content:
    print("测试通过：Team 记住了名字")
else:
    print("测试失败：Team 没有记住名字")