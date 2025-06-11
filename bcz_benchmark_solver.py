# bcz_benchmark_solver.py
# === Solve the simultaneous-move BCZ game for optimal efforts given a fixed G ===

import numpy as np
from itertools import combinations
from config import CONFIG

def compute_katz_bonacich_effort(G, alpha, delta):
    I = np.eye(G.shape[0])
    try:
        inv_matrix = np.linalg.inv(I - delta * G)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix [I - delta * G] is singular. Choose smaller delta or modify G.")
    x_star = inv_matrix @ alpha
    return x_star


def compute_payoffs(G, x, alpha, delta, c):
    n = len(x)
    payoffs = np.zeros(n)
    for i in range(n):
        local_effect = delta * sum(G[i, j] * x[i] * x[j] for j in range(n))
        link_cost = c * sum(G[i])
        payoffs[i] = alpha[i] * x[i] - 0.5 * x[i] ** 2 + local_effect - link_cost
    return payoffs


def evaluate_fixed_structure(G, alpha, delta, c):
    x_star = compute_katz_bonacich_effort(G, alpha, delta)
    payoffs = compute_payoffs(G, x_star, alpha, delta, c)
    welfare = np.sum(payoffs)
    return x_star, payoffs, welfare


def brute_force_optimal_structure(num_agents, alpha, delta, c):
    best_G = None
    best_welfare = -np.inf
    best_x = None
    best_payoffs = None

    edge_indices = list(combinations(range(num_agents), 2))
    for i in range(2 ** len(edge_indices)):
        G = np.zeros((num_agents, num_agents))
        for bit, (u, v) in enumerate(edge_indices):
            if (i >> bit) & 1:
                G[u, v] = G[v, u] = 1

        try:
            x_star, payoffs, welfare = evaluate_fixed_structure(G, alpha, delta, c)
        except ValueError:
            continue

        if welfare > best_welfare:
            best_G = G.copy()
            best_x = x_star
            best_payoffs = payoffs
            best_welfare = welfare

    return best_G, best_x, best_payoffs, best_welfare


# === Entry for main.py to call ===
def get_benchmark_from_config(config):
    num_agents = config["num_agents"]
    alpha = config["alpha"]
    delta = config["delta"]
    c = config["c"]
    G_opt, x_opt, pi_opt, W_opt = brute_force_optimal_structure(num_agents, alpha, delta, c)
    return {
        "G_opt": G_opt,
        "x_opt": x_opt,
        "pi_opt": pi_opt,
        "W_opt": W_opt
    }


if __name__ == "__main__":
    result = get_benchmark_from_config(CONFIG)

    print("[Optimal Network Structure G*]:")
    print(result["G_opt"])
    print("[Optimal Efforts x*]:", result["x_opt"])
    print("[Agent Payoffs π*]:", result["pi_opt"])
    print(f"[Global Welfare W*]: {result['W_opt']:.4f}")
