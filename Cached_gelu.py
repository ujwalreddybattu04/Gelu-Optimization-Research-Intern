import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import time


# ---- CachedGELU Implementation ----
class CachedGELU(nn.Module):
    def __init__(self, x_min=-100.0, x_max=100.0, N=50000):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.N = N
        self.step = (x_max - x_min) / (N - 1)
        self.inv_step = 1.0 / self.step
        x_table = torch.linspace(x_min, x_max, N)
        y_table = 0.5 * x_table * (1.0 + torch.erf(x_table / torch.sqrt(torch.tensor(2.0))))
        slope = torch.diff(y_table, append=y_table[-1].unsqueeze(0))
        self.register_buffer("y_table", y_table)
        self.register_buffer("slope", slope)

    def forward(self, x):
        x_clamped = torch.clamp(x, self.x_min, self.x_max)
        idx_f = (x_clamped - self.x_min) * self.inv_step
        idx = idx_f.long().clamp(0, self.N - 1)
        frac = idx_f - idx.float()
        y_val = self.y_table[idx]
        m_val = self.slope[idx]
        approx = y_val + frac * m_val
        gelu_exact = 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))
        return torch.where((x < self.x_min) | (x > self.x_max), gelu_exact, approx)