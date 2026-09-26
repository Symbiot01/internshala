"""Temperature scaling (Guo, Pleiss, Sun, Weinberger, ICML 2017).

One scalar T is fit by minimizing binary NLL on calibration-half logits.
Probabilities used by the decision rule are sigmoid(logit / T).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return T > 0 that minimizes NLL. T=1 leaves the model unchanged."""
    logit_t = torch.tensor(logits, dtype=torch.float64)
    label_t = torch.tensor(labels, dtype=torch.float64)
    log_t = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_t).clamp(min=1e-3, max=100.0)
        loss = F.binary_cross_entropy_with_logits(logit_t / temperature, label_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).clamp(min=1e-3, max=100.0).item())


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.clip(logits.astype(np.float64) / max(temperature, 1e-3), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float64)
