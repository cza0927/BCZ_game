# main.py
# === Import Modules ===
from env.env_core import BCZEnv
from agents.llm_agent import LLMAgent
from api.llm_interface import OpenAIWrapper
import sys
import os
import datetime
import logging

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

# 创建 log 文件名，格式为 output_{时间戳}.txt
def create_log_file():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(log_dir, exist_ok=True)  # 自动创建目录
    log_filename = f"output_{now}.txt"
    log_path = os.path.join(log_dir, log_filename)
    return open(log_path, "w", encoding="utf-8")

def run_simulation(num_agents=3, T=2):
    env = BCZEnv(num_agents=num_agents, delta=0.25)
    model = OpenAIWrapper(model_name="gpt-4o")  # 支持 gpt-3.5-turbo 等
    agents = [LLMAgent(i, num_agents, model) for i in range(num_agents)]
    
    for t in range(T):
        print(f"\n=== ROUND {t+1} ===")

        link_actions = []
        for i, agent in enumerate(agents):
            obs = env.get_observation(agent.id)
            links = agent.decide_links(obs)
            print(f"[Agent {agent.id} link decision]: {links}")
            link_actions.append(links)

        # 环境生成 G，先用 dummy effort 占位
        G, _, _, _ = env.step(link_actions, [0] * num_agents)
        
        # 所有 agent 再做 effort 决策（此时能看到 G）
        effort_actions = []
        for i, agent in enumerate(agents):
            obs = env.get_observation(agent.id)
            effort = agent.decide_effort(obs)
            print(f"[Agent {agent.id} effort decision]: {effort}")
            effort_actions.append(effort)

        # 用真实 effort 重算 step（覆盖前面的 dummy 记录）
        env.G_history[-1] = G
        env.effort_history[-1] = effort_actions
        _, x, pi, W = env.step(link_actions, effort_actions)
        
        # # Step environment and update memory
        # G, x, pi, W = env.step(link_actions, effort_actions)
        for agent in agents:
            agent.update_memory(G, x, pi)

        print(f"Global Welfare W = {W[-1].sum():.2f}")

if __name__ == "__main__":
    log_file = create_log_file()
    tee = Tee(sys.stdout, log_file)
    sys.stdout = sys.stderr = tee

    try:
        run_simulation()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()

