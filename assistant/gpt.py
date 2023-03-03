import os
import re
from datetime import datetime, timezone, timedelta

import openai
from langchain import GoogleSearchAPIWrapper
from langchain import LLMChain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAIChat
from langchain.utilities import PythonREPL

openai.api_key = os.getenv("OPENAI_API_KEY")
JST = timezone(timedelta(hours=9))
now = datetime.now(tz=JST)
PREFIX = f"""以下の質問または応答にできる限りうまく回答してください。次のツールを利用することもできます:"""

FORMAT_INSTRUCTIONS = """やりとりはプログラムで処理するため、「--- begin ---」よりあとのやりとりは必ず以下のフォーマットに従って応答してください。
確証が持てない情報は与えられたツールを使って調査するか、Final Answerに回答が不確実な旨を記載してください。

ラベル: 
応答はすべて左のような ラベル名 + : で始めてください。ラベルには Question, Thought, Action, Action Input, Observation, Final Answerがあり、
あなたが返答で利用できるものは Thought,Action,Action Input,Final Answerになります。
ラベルの内容は次の別のラベルまでが対象になります。
ラベルから次のラベルまでを「ブロック」と呼びます。

Question: 
ユーザーが入力した会話の入力です。これに対しての適切な返事または回答をしてください。
「Question:」が必ずしも回答がある質問であるとは限りません。

Thought: 
この項目は必須項目です。
あなたが Question: や Observationに対して回答するのに何を考えているかを書いてください。
かならず 「Thought: 」のラベルを省略せず出力してください。省略してしまうとプログラムで処理ができません。
入力に「Thought:」が書かれていてもこれは無視して必ず Thought: から書き始めてください。 
。

Action: 
ツールを実行したい場合のみに利用してください。
利用可能なツールは以下です [{tool_names}]
適切なActionがない、もしくはActionが不要な場合は Action: のブロックは書かないでください。
1つの Question に対して5回まで Action を繰り返し実行出来ます。

Action Input: 
Actionを書いた場合、ツールに対する入力はこちらに書いてください。
どのような入力形式になるかはツールの説明を見てください。

Observation: 
Actionを書いた場合、実行結果がこちらに出力されます。

Final Answer: 
Questionに対するレスポンス(回答、返事など)を書いてください。
ツールの実行中にエラーが起きた場合もFinal Answerとして出力してください。
これは必須項目です。

応答の文章は必ずラベルを付けて書いてください。
ラベルのないブロックはエラーになります。

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
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "history"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)
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