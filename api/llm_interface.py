# api/llm_interface.pys
import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

class MockLLMWrapper: # 用于debug，在main.py中的CONFIG中的"use_mock"处设置
    def __init__(self, model_name="mock", temperature=0.0, num_agents=3):
        self.model = model_name
        self.temperature = temperature
        self.num_agents = num_agents

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if "link" in user_prompt.lower():
            # 生成一个固定 pattern：与偶数 agent 建边，自己位置为 0
            link_vector = [1 if (j % 2 == 0 and j != self.extract_agent_id(user_prompt)) else 0
                           for j in range(self.num_agents)]
            link_vector[self.extract_agent_id(user_prompt)] = 0  # 保证不连接自己
            return " ".join(map(str, link_vector))

        if "effort" in user_prompt.lower():
            return "`2`"  # 固定 effort 可换成随机值以测试 parser

        return "0"

    def extract_agent_id(self, prompt_text):
        import re
        match = re.search(r"You are Agent (\d+)", prompt_text)
        return int(match.group(1)) if match else 0

class OpenAIWrapper:
    def __init__(self, model_name="gpt-4o-mini", temperature=0.7):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model_name
        self.temperature = temperature

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        try:
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("[API ERROR]:", e)
            return "0"
