import torch
from torch import nn

class LocalReponseNormalization(nn.Module):
    def __init__(self, k=2, n=5, alpha=10e-4, beta=0.75):
        super(LocalReponseNormalization, self).__init__()
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        x_squared = x**2
        x_cumsum = nn.functional.pad(x_squared.cumsum(dim=1), (0, 0, 0, 0, 1, 0)) # Add padding to the front of the channels (before the first channel)

        indices = torch.arange(0, x.size(1), device=x.device)
        lower_bounds = (indices - self.n // 2).clamp(min=0)
        upper_bounds = (indices + self.n // 2).clamp(max=x.size(1) - 1)

        expanded_lower_bounds = lower_bounds.view(1, x.size(1), 1, 1).expand_as(x)
        expanded_upper_bounds = upper_bounds.view(1, x.size(1), 1, 1).expand_as(x)

        # Sum(i to j) = S[j] - S[i - 1], however since we've added padding to handle the S[-1] case, S[j] = S[j+1] & S[i-1] = S[i]
        start = x_cumsum.gather(dim=1, index=expanded_lower_bounds)
        end = x_cumsum.gather(dim=1, index=expanded_upper_bounds + 1)
        sum_squared = end - start

        b = x / (self.k + self.alpha * sum_squared) ** self.beta

        return b