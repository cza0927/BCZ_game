# env/env_core.py
import numpy as np

class BCZEnv:
    def __init__(self, num_agents=3, T=1, delta=1, alpha=None, c=1.0):
        self.num_agents = num_agents
        self.T = T
        self.delta = delta
        self.alpha = alpha if alpha is not None else np.ones(num_agents)
        self.c = c

        self.round = 0
        self.G_history = []
        self.effort_history = []
        self.payoff_history = []

    def step(self, link_actions, effort_actions):
        G = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j and link_actions[i][j] and link_actions[j][i]:
                    G[i][j] = 1

        x = np.array(effort_actions)
        pi = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            interaction_term = sum(
                G[i][j] * x[i] * x[j] for j in range(self.num_agents) if j != i
            )
            link_cost = self.c * sum(G[i])
            pi[i] = self.alpha[i] * x[i] - 0.5 * x[i] ** 2 + self.delta * interaction_term - link_cost

        self.G_history.append(G.copy())
        self.effort_history.append(x.copy())
        self.payoff_history.append(pi.copy())
        self.round += 1

        return G, x, pi, np.array(self.payoff_history)

    def compute_links(self, link_actions):
        G = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j and link_actions[i][j] and link_actions[j][i]:
                    G[i][j] = 1
        return G
    
    def get_observation(self, agent_id):
        return {
            "agent_id": agent_id,
            "round": self.round,
            "G_history": [G.tolist() for G in self.G_history],
            "effort_history": [e for e in self.effort_history],
            "payoff_history": [p for p in self.payoff_history]
        }

