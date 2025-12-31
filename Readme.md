Knowledge: 存放了gitclone得到的原始代码库，和向量化的代码库；
tools: 存放了相关工具
    - web_search: 在线检索（包括检索github仓库的url）
    - get_github_resp：将检索到的repo url克隆到本地
    - build_codebase.py: 用于将代码库向量化并存储到本地向量数据库
    - code_base_search.py: 用于检索本地向量数据库中的代码库内容
model: 使用本地embedding
Agno_test.py: agno team的workflow结构，使用```fastapi dev /workspace/ai-test/AgentPractice/Agno_test.py```启动服务，并在https://os.agno.com/访问

需要修改的地方: 
    - 1. 各个工具里面的路径，都使用绝对路径；
    - 2. Agno_test.py里面的repo_search_agent的instructions（line 38），这是为了保证模型不乱去别的地方。