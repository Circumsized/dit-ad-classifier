"""Small, independently testable domain-alignment building blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL loss between source and target feature covariances."""

    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be [N,D] with the same D")
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.new_zeros(())
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / (source.shape[0] - 1)
    target_cov = target_centered.T @ target_centered / (target.shape[0] - 1)
    return (source_cov - target_cov).pow(2).mean()


def mmd_rbf_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Multi-kernel RBF maximum mean discrepancy."""

    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be [N,D] with the same D")
    xx = torch.cdist(source, source).pow(2)
    yy = torch.cdist(target, target).pow(2)
    xy = torch.cdist(source, target).pow(2)
    loss = source.new_zeros(())
    for bandwidth in bandwidths:
        gamma = 1.0 / (2.0 * bandwidth * bandwidth)
        loss = loss + torch.exp(-gamma * xx).mean() + torch.exp(-gamma * yy).mean()
        loss = loss - 2.0 * torch.exp(-gamma * xy).mean()
    return loss / len(bandwidths)


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.strength * grad_output, None


def gradient_reverse(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(x, float(strength))


class DomainDiscriminator(nn.Module):
    """Site discriminator for optional DANN training."""

    def __init__(self, feature_dim: int, n_domains: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, features: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        return self.network(gradient_reverse(features, strength))
