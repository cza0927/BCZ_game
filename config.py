# config.py
import numpy as np

CONFIG = {
    "use_mock": True,
    "num_agents": 3,
    "T": 3,
    "alpha": np.array([1.0, 1.0, 1.0]),
    "delta": 0.5,
    "c": 1.0,
    "model_name": "gpt-4o-mini",
    "temperature": 0.7
}