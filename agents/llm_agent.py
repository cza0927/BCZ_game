# agents/llm_agents.py
import os
import numpy as np
from agents.base_agent import BaseAgent
from api.llm_interface import OpenAIWrapper
import re
from typing import List

class LLMAgent(BaseAgent):
    def __init__(self, agent_id, num_agents, model: OpenAIWrapper, alpha=1.0, delta=0.5, c=1.0):
        super().__init__(agent_id, num_agents)
        self.model = model
        self.alpha = alpha
        self.delta = delta
        self.c = c
        self._load_prompts()

        # 注入 agent 参数进 system prompt
        self.system_prompt = self.system_prompt.format(
            alpha=self.alpha,
            delta=self.delta,
            c=self.c
        )

    def _load_prompts(self):
        base_dir = os.path.dirname(__file__)
        prompt_dir = os.path.abspath(os.path.join(base_dir, "../prompt"))

        with open(os.path.join(prompt_dir, "system_prompt.txt"), "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        with open(os.path.join(prompt_dir, "link_body_template.txt"), "r", encoding="utf-8") as f:
            self.link_prompt_template = f.read()

        with open(os.path.join(prompt_dir, "effort_body_template.txt"), "r", encoding="utf-8") as f:
            self.effort_prompt_template = f.read()

    def decide_links(self, observation):
        prompt = self.link_prompt_template.format(
            agent_id=self.id,
            round=observation["round"],
            G_history=observation["G_history"],
            effort_history=observation["effort_history"],
            payoff_history=observation["payoff_history"],
            num_agents=self.num_agents,
        )
        response = self.model.chat(self.system_prompt, prompt)
        print(f"[Agent {self.id}] Link decision reasoning:\n{response}")
        return self.parse_link_response(response)

    def decide_effort(self, observation):
        prompt = self.effort_prompt_template.format(
            agent_id=self.id,
            round=observation["round"],
            G_history=observation["G_history"],
            effort_history=observation["effort_history"],
            payoff_history=observation["payoff_history"],
            num_agents=self.num_agents,
        )
        response = self.model.chat(self.system_prompt, prompt)
        print(f"[Agent {self.id}] Effort decision reasoning:\n{response}")
        return self.parse_effort_response(response)
    
    def parse_link_response(self, response: str) -> List[int]:
        # 去除 markdown 包裹
        cleaned = response.replace("`", "").replace("**", "").strip()
        # 匹配至少 num_agents 个 0/1，允许中间有空格、换行
        match = re.search(rf"\b([01](?:[\s]+[01]){{{self.num_agents - 1}}})\b", cleaned)
        if not match:
            print(f"[ERROR] Cannot parse links from: {response}")
            return [0.0] * self.num_agents
        try:
            return [float(x) for x in match.group(1).strip().split()]
        except Exception:
            print(f"[ERROR] Failed to convert parsed link: {match.group(1)}")
            return [0.0] * self.num_agents

    def parse_effort_response(self, response: str) -> float:
        """
        解析 agent 的 effort 响应，从后向前提取第一个合法的数值（整数或浮点数）。
        """
        # 去除 markdown 格式干扰
        cleaned = response.replace("`", "").replace("**", "").strip()

        # 提取所有数值
        matches = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\b", cleaned)
        if not matches:
            print(f"[ERROR] Cannot parse effort from response: {response}")
            return 0.0

        # 从后往前找第一个合法的 effort 值
        for val in reversed(matches):
            try:
                effort = float(val)
                return effort
            except ValueError:
                continue

        print(f"[ERROR] Failed to convert effort from response: {response}")
        return 0.0