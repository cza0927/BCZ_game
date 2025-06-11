# agents/base_agent.py
class BaseAgent:
    def __init__(self, agent_id, num_agents):
        self.id = agent_id
        self.num_agents = num_agents
        self.memory = []

    def update_memory(self, G, x, pi):
        self.memory.append({
            "G": G,
            "x": x,
            "pi": pi
        })