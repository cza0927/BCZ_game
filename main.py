# main.py
# === Import Modules ===
from env.env_core import BCZEnv
from agents.llm_agent import LLMAgent
from api.llm_interface import MockLLMWrapper, OpenAIWrapper
import sys
import os
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from bcz_benchmark_solver import get_benchmark_from_config
from config import CONFIG

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
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"output_{now}.txt"
    log_path = os.path.join(log_dir, log_filename)
    return open(log_path, "w", encoding="utf-8")

# 绘制结果图
def analyze_convergence(G_history, effort_history, payoff_history, alpha=None, delta=0.3, c=1.0, title_prefix="BCZ Game"):
    num_rounds = len(G_history)
    total_links = [np.sum(G) for G in G_history]
    avg_efforts = [np.mean(x) for x in effort_history]
    global_welfare = [np.sum(pi) for pi in payoff_history]
    delta_G = [0.0]
    delta_x = [0.0]

    for t in range(1, num_rounds):
        g_diff = np.abs(np.array(G_history[t]) - np.array(G_history[t - 1])).sum()
        x_diff = np.linalg.norm(np.array(effort_history[t]) - np.array(effort_history[t - 1]))
        delta_G.append(g_diff)
        delta_x.append(x_diff)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    rounds = list(range(num_rounds))

    axs[0, 0].plot(rounds, total_links, label='Total Links')
    axs[0, 0].set_title(f"{title_prefix}: Total Links Over Time")
    axs[0, 0].set_xticks(rounds)

    axs[0, 1].plot(rounds, avg_efforts, label='Avg Effort', color='green')
    axs[0, 1].set_title(f"{title_prefix}: Avg Effort Over Time")
    axs[0, 1].set_xticks(rounds)

    axs[1, 0].plot(rounds, global_welfare, label='Global Welfare', color='purple')
    axs[1, 0].set_title(f"{title_prefix}: Global Welfare Over Time")
    axs[1, 0].set_xticks(rounds)

    axs[1, 1].plot(rounds, delta_G, label='ΔG', color='red')
    axs[1, 1].plot(rounds, delta_x, label='Δx', color='orange')
    axs[1, 1].legend()
    axs[1, 1].set_title(f"{title_prefix}: Change in G and Effort")
    axs[1, 1].set_xticks(rounds)

    for ax in axs.flat:
        ax.set_xlabel('Round')
        ax.grid(True)

    plt.tight_layout()
    fig_path = os.path.join("output", "bcz_convergence_analysis.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"[INFO] Convergence plot saved to: {fig_path}")

# 若收敛，则提前终止
def should_terminate(G_history, welfare_history, threshold_rounds=5, welfare_delta_threshold=1.0):
    import numpy as np
    if len(G_history) < threshold_rounds or len(welfare_history) < threshold_rounds + 1:
        return False

    recent_Gs = G_history[-threshold_rounds:]
    if not all(np.array_equal(recent_Gs[i], recent_Gs[0]) for i in range(1, threshold_rounds)):
        return False

    delta_welfare = np.array(welfare_history[-threshold_rounds:]) - np.array(welfare_history[-threshold_rounds - 1:-1])
    if np.sum(delta_welfare) < welfare_delta_threshold:
        return True

    return False

# 正式模拟BCZ_game
def run_simulation():
    num_agents = CONFIG["num_agents"]
    T = CONFIG["T"]
    alpha = CONFIG["alpha"]
    delta = CONFIG["delta"]
    c = CONFIG["c"]
    model_name = CONFIG["model_name"]
    temperature = CONFIG["temperature"]

    env = BCZEnv(num_agents=num_agents, T=T, delta=delta, alpha=alpha, c=c)
    if CONFIG["use_mock"]:
        model = MockLLMWrapper(
            model_name="mock",
            temperature=CONFIG["temperature"],
            num_agents=CONFIG["num_agents"]
        )
    else:
        model = OpenAIWrapper(
            model_name=CONFIG["model_name"],
            temperature=CONFIG["temperature"]
        )
    
    agents = [
        LLMAgent(i, num_agents, model, alpha=alpha[i], delta=delta, c=c)
        for i in range(num_agents)
    ]

    # 玩T轮
    for t in range(T):
        print(f"\n=== ROUND {t+1} ===")

        # 1. agents 做出 link 决策
        link_actions = []
        for agent in agents:
            obs = env.get_observation(agent.id)
            links = agent.decide_links(obs)
            print(f"[Agent {agent.id} link decision]: {links}")
            link_actions.append(links)

        # 2. 用 compute_links 生成 G，但不调用 step
        G = env.compute_links(link_actions)

        # 3. agents 观察 G 后做 effort 决策
        effort_actions = []
        for agent in agents:
            obs = env.get_observation(agent.id)
            effort = agent.decide_effort(obs)
            print(f"[Agent {agent.id} effort decision]: {effort}")
            effort_actions.append(effort)

        # 4. 执行一次真实 step：记录 effort + payoff
        _, x, pi, W = env.step(link_actions, effort_actions)

        for agent in agents:
            agent.update_memory(G, x, pi)

        print(f"Global Welfare W = {W[-1].sum():.2f}")
        
        # 每轮末尾记录监控指标：连边结构，effort投入，总收益
        G_curr = env.G_history[-1]
        x_curr = env.effort_history[-1]
        pi_curr = env.payoff_history[-1]

        print(f"Round {t+1} link matrix:\n{np.array(G_curr)}")
        print(f"Round {t+1} effort: {x_curr}")
        print(f"Round {t+1} payoffs: {pi_curr}")
        print(f"Round {t+1} total welfare: {np.sum(pi_curr):.2f}")
        
        # 提前终止判定
        if should_terminate(env.G_history, env.payoff_history):
            print(f"⚠️ 提前终止：连续 {5} 轮 G 未变且 welfare 增量低于阈值")
            break

    # 可视化收敛分析
    analyze_convergence(
        G_history=env.G_history,
        effort_history=env.effort_history,
        payoff_history=env.payoff_history,
        alpha=alpha,
        delta=delta,
        c=c,
        title_prefix="BCZ Game Mock Run"
    )
    
    benchmark = get_benchmark_from_config(CONFIG)
    W_llm = np.sum(env.payoff_history[-1])

    print("\n=== BENCHMARK COMPARISON ===")
    print("[Benchmark Optimal G*]:\n", benchmark["G_opt"])
    print("[Benchmark Optimal Efforts x*]:", benchmark["x_opt"])
    print("[Benchmark Payoffs π*]:", benchmark["pi_opt"])
    print(f"[Benchmark Global Welfare W*]: {benchmark['W_opt']:.4f}")
    print(f"[LLM Agent Final Welfare]: {W_llm:.4f}")
    print(f"[Efficiency Ratio]: {W_llm / benchmark['W_opt']:.4f}")
        
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

    