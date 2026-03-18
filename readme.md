## Instruction
Knowledge: 
    codebase: 存放gitclone得到的原始代码库
    vector_db: 存放向量化之后的代码库

model: 使用本地embedding

tools: 存放相关工具
```
    - web_search: 在线检索（包括检索github仓库的url）
    - clone_github_repo：将检索到的repo url克隆到本地
    - build_vector_base.py: 用于将代码库向量化并存储到本地向量数据库
    - semantic_code_search.py: 用于检索本地向量数据库中的代码库内容
```

Agno_AgentOS.py: webui运行agno的team对话功能。使用```fastapi dev /srv/AgnoCodingAgent/Agno_AgentOS.py```启动服务，并在**https://os.agno.com/**访问

Agno_api.py: 使用OpenAI兼容的格式给coding插件提供对话功能，使用```# sudo systemctl restart fastapi-plugin.service```启动服务，接口参考下面的continue插件说明：

~/.continue/config.yaml
```
name: coder-completion-generation
version: 1.1.0
schema: v1
models:
  - name: agno_team
    provider: openai
    model: agno_team
    apiBase: http://10.6.8.87:8000/continue/v1
    apiKey: sk-dummykey
    requestOptions:
        headers: 
          X-Session-ID: continue-session-001
          X-User-ID: dev-user
    roles:
      - apply
      - chat
      - edit
      - autocomplete
    capabilities:
      - tool_use
    chatOptions:
      baseSystemMessage: 你是精通多种计算机编程语言的代码助手，按要求使用中文回答用户问题。回答保持简洁，只选择最有效的方法。
```

## Usage
为了保证模型在路径相关操作上的准确性，在使用前**建议将下列文件中涉及路径的变量修改为绝对路径**。
1. tools/*下除'web_search.py'以外的三个工具
2. Agno_AgentOS.py 和Agno_api.py中关于路径的指令