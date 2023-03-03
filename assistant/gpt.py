import os
import re
from datetime import datetime, timezone, timedelta

import openai
from langchain import GoogleSearchAPIWrapper
from langchain import LLMChain
from langchain.agents import ConversationalAgent, Tool, AgentExecutor
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAIChat
from langchain.utilities import PythonREPL

openai.api_key = os.getenv("OPENAI_API_KEY")
JST = timezone(timedelta(hours=9))
now = datetime.now(tz=JST)
PREFIX = f"""
以下の質問または応答にできる限りうまく回答してください。
確証が持てない情報は与えられたツールを使って調査するか、Final Answerに回答が不確実な旨を記載してください。

次のツールを利用することもできます:
"""

SUFFIX = f"""現在時刻は{now}です。

--- begin ---
{{history}}
Question: {{input}}
{{agent_scratchpad}}"""

llm = OpenAIChat(temperature=0)
memory = ConversationSummaryBufferMemory(llm=llm, ai_prefix="Final Answer", human_prefix="Question")
tools = [
    Tool(
        "Google Search",  # name of Tool
        GoogleSearchAPIWrapper().run,  # Callable[[str], str]
        "Google検索をつかって情報を取得するためのツールです。汎用的な質問や情報の確からしさを確認するために使えます。入力はGoogleの検索キーワードです。",
    ),
]
prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    # format_instructions=FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "history"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
agent = ConversationalAgent(llm_chain=llm_chain, tools=tools)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)
GPT_PATTERN = re.compile("^gpt\s+((.+))", re.MULTILINE)

def gpt(prompt: str) -> str:
    res = agent_executor({'input': prompt})
    print(res)
    return res['output']

if __name__ == '__main__':
    while True:
        prompt = input()
        if not prompt:
            break
        print(gpt(prompt))