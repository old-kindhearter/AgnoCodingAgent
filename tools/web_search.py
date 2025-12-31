import json
import logging
import os
import requests
from dotenv import load_dotenv
from agno.tools import Toolkit
from agno.utils.log import logger

class WebSearcher(Toolkit):
    def __init__(self):
        super().__init__(name="websearch_tools", tools=[self.web_search])

        api_key = os.getenv("TAVILY_API_KEY")
        api_url = os.getenv("TAVILY_API_URL")
        assert api_key is not None, "TAVILY_API_KEY is not set"

        self.api_key = api_key
        self.api_url = api_url

    def web_search(self, query: str, count: int = 5) -> str:
        """
        使用 Tavily 搜索引擎检索话题相关内容。
        Args:
            query(str): 待搜索的话题/关键词
            count(int): 搜索结果数量，默认为5
        
        Returns:
            str: 返回格式化后的字符串，包含了所有结果的链接、标题和摘要。
        """
        # Tavily 的 API 参数结构
        payload = json.dumps({
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced", # 使用 advanced 模式以获得更好结果，也可以选 "basic"
            "include_answer": False,    # 是否让 Tavily 直接生成答案，这里我们只要搜索结果
            "include_images": False,
            "include_raw_content": False,
            "max_results": count
        })
        headers = {
            "Content-Type": "application/json"
        }
        try:
            logging.info(f"Searching Tavily for: {query} ...")
            response = requests.request("POST", self.api_url, headers=headers, data=payload)
            response.raise_for_status()
            data = response.json()
            logging.info(f"Gathering the returning...")
            
            # Tavily 只要没有抛出 HTTP 错误，通常就是 200，不需要像 Bocha 那样额外判断 code 字段
            # 但是为了稳健，可以检查 results 列表
            search_results = data.get("results", [])
            
            if not search_results:
                return "未搜索到相关内容。"
            
            formatted_results = ""
            for idx, item in enumerate(search_results):
                title = item.get("title", "无标题")
                content = item.get("content", "无摘要")
                url = item.get("url", "")
                # Tavily 通常不一定每条都在 results 里带详细时间，如果有 published_date 就用，没有就标未知
                date = item.get("published_date", "未知日期")
                
                # 保持和你之前一样的格式化输出
                formatted_results += f"[{idx+1}] 标题: {title}\n    时间: {date}\n    内容: {content}\n    来源: {url}\n\n"
            
            return formatted_results
        except Exception as e:
            error_msg = f"搜索工具调用失败: {str(e)}"
            logger.warning(f"Failed to run Web Search: {e}")
            return error_msg

def web_search(query: str, count: int = 10):
    searcher = WebSearcher()
    return searcher.web_search(query, count)


if __name__ == "__main__":
    load_dotenv()
    
    search_result = str(web_search("AlignKT github", count=5))
    print(type(search_result))
    print(search_result)
