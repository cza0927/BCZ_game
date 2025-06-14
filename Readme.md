# Sequential BCZ\_Game 项目说明文档

本项目基于 Ballester–Calvó-Armengol–Zenou (BCZ) 模型，构建了一个多智能体网络博弈环境。通过结合 LLM 决策代理（如 GPT 系列）与经典博弈求解方法，探索智能体在合作结构与努力选择中的社会推理能力。

## 🎲 BCZ 博弈模型简介

BCZ（Ballester–Calvó-Armengol–Zenou）博弈是一种**网络互依型努力博弈（Effort Game on Networks）**，由 Ballester et al. 在 2006 年提出，用于刻画个体在社会网络中决策努力水平的最优方式。

每个智能体 $i \in \{1, ..., n\}$ 都可选择其努力值 $x_i$，同时与网络中其他个体 $j$ 之间存在连接关系 $g_{ij} \in \{0, 1\}$。

### 收益函数公式（payoff function）

BCZ 博弈中，agent $i$ 的总收益为：

$$
\pi_i = \alpha_i x_i - \frac{1}{2}x_i^2 + \delta \sum_{j \neq i} g_{ij} x_i x_j - c \sum_{j \neq i} g_{ij}
$$

其中：

* $\alpha_i$：个体的私有收益敏感系数（倾向于努力）
* $x_i$：agent 的努力水平
* $g_{ij}$：是否与 agent $j$ 连边（只有当 $i$ 与 $j$ 互相选择对方时，$g_{ij} = 1$）
* $\delta$：互惠强度系数（努力互相促进的程度）
* $c$：每建立一条边的成本

全局 welfare 为所有 agent 收益之和：

$$
W = \sum_i \pi_i
$$

### 博弈目标

每个智能体希望最大化自身收益 $\pi_i$，而研究者可以关心：

* 个体博弈后是否能收敛至稳定网络结构；
* 在有限轮数内是否能逼近理论最优的 $\{g_{ij}^*, x_i^*\}$；
* LLM agent 是否具备 “社会推理” 能力，在交互中收敛到高效率结构。


## 📁 项目结构

```
BCZ_Game/
├── agents/                   # 智能体模块（抽象类 + LLM代理 + 记忆系统）
│   ├── base_agent.py         # Agent 抽象类定义
│   ├── llm_agent.py          # 基于 LLM 的 Agent 实现
│   └── memory.py             # Agent 的历史观测记忆机制
│
├── api/                      # OpenAI API 接口封装
│   └── llm_interface.py      # LLM 模型的统一调用接口（支持Mock和OpenAI）
│
├── env/                      # 环境定义模块
│   └── env_core.py           # BCZ 游戏环境定义，包含 payoff 计算与状态转移
│
├── prompt/                   # 决策提示模板
│   ├── system_prompt.txt     # Agent的背景设定和规则
│   ├── link_body_template.txt# 连边决策模板
│   └── effort_body_template.txt # 努力决策模板
│
├── output/                   # 输出文件夹（日志与可视化图像）
│   ├── output_*.txt          # 每次运行的详细日志记录
│   └── bcz_convergence_analysis.png # 可视化的收敛趋势图
│
├── config.py                 # 全局参数配置（alpha, delta, c, num_agents等）
├── main.py                   # 主程序入口，运行多轮博弈仿真
├── bcz_benchmark_solver.py   # 使用经典 BCZ 方法求理论最优结构和effort
├── requirements.txt          # 所需 Python 库（建议使用虚拟环境）
└── README.md                 # 本文件
```

---

## ⚙️ 运行方式

### 1. 安装环境

建议使用 `venv` 或 `conda` 创建独立环境：

```bash
pip install -r requirements.txt
```

### 2. 修改参数配置

在 `config.py` 中配置仿真参数：

```python
CONFIG = {
    "use_mock": True,             # 是否使用 mock LLM（无需API）
    "num_agents": 3,
    "T": 5,                       # 总轮数
    "alpha": np.array([1.0, 1.0, 1.0]),
    "delta": 0.5,
    "c": 1.0,                    # 连边成本
    "model_name": "gpt-4o-mini",
    "temperature": 0.7
}
```

若要利用LLM的api进行模拟，请将"use_mock"设置为False，同时在.env中输入api

### 3. 启动仿真运行

```bash
python main.py
```

程序将：

* 多轮模拟 agent 连边与 effort 决策
* 保存日志到 `output/` 文件夹
* 输出收敛趋势图 `bcz_convergence_analysis.png`
* 自动调用 `bcz_benchmark_solver.py` 对比理论最优结构和 welfare

---

## 📈 输出内容

每次运行后将输出：

* 每一轮 agent 的 link 决策、effort 决策、payoff
* 全局 welfare 变化趋势
* LLM agent 收敛表现 vs 理论最优表现（benchmark）
* 可视化结果存储于：`output/bcz_convergence_analysis.png`

---

## 🧪 项目支持

* ✅ 支持 LLM agent 的社会博弈推理评估
* ✅ 模拟连边结构演化与 effort 策略学习
* ✅ 可切换 OpenAI 与 Mock 模式，方便 Debug
* ✅ 引入 `c` 成本后，支持 benchmark 最优结构求解

## 🧪 后续拓展

* 引入异质化 `alpha_i`, `delta_ij`，建模不对称合作关系
* 加入 reputation、惩罚机制，测试策略演化
* 批量实验并评估不同模型或策略稳定性

---

如有问题或需求，欢迎随时联系维护者，或在 issue 区提出建议！
