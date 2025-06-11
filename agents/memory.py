# agents/memory.py
import numpy as np

class AgentMemory:
    def __init__(self, n_agents, max_rounds):
        self.n = n_agents
        self.T = max_rounds
        self.history = {
            "G": [],
            "x": [],
            "pi": []
        }

    def update(self, G, x, pi):
        self.history["G"].append(G)
        self.history["x"].append(x)
        self.history["pi"].append(pi)

    def get_last_k(self, k=1):
        return {
            "G": self.history["G"][-k:],
            "x": self.history["x"][-k:],
            "pi": self.history["pi"][-k:]
        }

    def summarize_state(self):
        return {
            "avg_effort": np.mean(self.history["x"], axis=0) if self.history["x"] else [],
            "avg_payoff": np.mean(self.history["pi"], axis=0) if self.history["pi"] else [],
        }
