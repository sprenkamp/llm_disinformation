from langchain.tools import DuckDuckGoSearchRun
from duckduckgo_search.exceptions import TimeoutException

def web_search(statement: str, author: str, date:str) -> str:
        search = DuckDuckGoSearchRun(backennd="news", max_results=10)
        #llm = ChatOpenAI(temperature=0, model="gpt-4", timeout=60)
        input_str = f'{statement + "|" + author + "|" + date}'
        try:
              return search.run(input_str)
        except TimeoutException:
              return "no results found due to timeout"
              