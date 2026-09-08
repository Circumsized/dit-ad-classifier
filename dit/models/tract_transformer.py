"""A shape-safe Transformer for AFQ tract profiles."""

from __future__ import annotations

import torch
from torch import nn


class TractTransformer(nn.Module):
    """Encode along-tract profiles and model relationships between tracts.

    Input shape is always ``[batch, tract, point, metric]``.  Missing values
    may be passed as NaNs or described by ``valid_mask``.  The model returns
    raw logits; callers apply softmax only for reporting probabilities.
    """

    def __init__(
        self,
        *,
        n_tracts: int,
        n_points: int,
        n_metrics: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        feedforward_dim: int = 192,
        dropout: float = 0.2,
        pooling: str = "cls",
        include_covariates: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if n_tracts < 1 or n_points < 2 or n_metrics < 1 or n_classes < 2:
            raise ValueError("invalid model dimensions")
        if pooling not in {"cls", "mean", "attention"}:
            raise ValueError("pooling must be cls, mean or attention")

        self.n_tracts = n_tracts
        self.n_points = n_points
        self.n_metrics = n_metrics
        self.n_classes = n_classes
        self.d_model = d_model
        self.pooling = pooling
        self.include_covariates = include_covariates

        self.point_encoder = nn.Sequential(
            nn.Conv1d(n_metrics, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.point_position = nn.Parameter(torch.zeros(1, n_points, d_model))
        self.point_score = nn.Linear(d_model, 1)
        self.tract_embedding = nn.Parameter(torch.zeros(1, n_tracts, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.readout_score = nn.Linear(d_model, 1)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.covariate_encoder = (
            nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.LayerNorm(16), nn.Dropout(dropout))
            if include_covariates
            else None
        )
        classifier_dim = d_model + (16 if include_covariates else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, n_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.point_position, std=0.02)
        nn.init.normal_(self.tract_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        covariates: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x, valid_mask=valid_mask)
        if self.include_covariates:
            if covariates is None:
                raise ValueError("covariates with shape [B,2] are required")
            if covariates.ndim != 2 or covariates.shape != (x.shape[0], 2):
                raise ValueError(f"expected covariates [B,2], got {tuple(covariates.shape)}")
            features = torch.cat((features, self.covariate_encoder(covariates.float())), dim=-1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits

    def forward_features(
        self, x: torch.Tensor, *, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,T,P,M], got {tuple(x.shape)}")
        batch, tracts, points, metrics = x.shape
        if tracts != self.n_tracts or points != self.n_points or metrics != self.n_metrics:
            raise ValueError(
                f"expected [B,{self.n_tracts},{self.n_points},{self.n_metrics}], "
                f"got {tuple(x.shape)}"
            )
        if valid_mask is None:
            valid_mask = torch.isfinite(x)
        if valid_mask.shape != x.shape:
            raise ValueError("valid_mask must match x")
        valid_mask = valid_mask.bool()
        point_valid = valid_mask.any(dim=-1)
        tract_valid = point_valid.any(dim=-1)
        safe_x = torch.where(valid_mask, x, torch.zeros_like(x)).float()

        encoded = self.point_encoder(
            safe_x.reshape(batch * tracts, points, metrics).transpose(1, 2)
        ).transpose(1, 2)
        encoded = encoded + self.point_position
        flat_valid = point_valid.reshape(batch * tracts, points)
        scores = self.point_score(encoded).squeeze(-1)
        scores = scores.masked_fill(~flat_valid, torch.finfo(scores.dtype).min)
        all_missing = ~flat_valid.any(dim=1)
        scores = torch.where(all_missing[:, None], torch.zeros_like(scores), scores)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(flat_valid, weights, torch.zeros_like(weights))
        tract_tokens = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        tract_tokens = tract_tokens.reshape(batch, tracts, self.d_model)
        tract_tokens = tract_tokens + self.tract_embedding
        tract_tokens = torch.where(tract_valid.unsqueeze(-1), tract_tokens, torch.zeros_like(tract_tokens))

        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, tract_tokens), dim=1)
        padding_mask = torch.cat(
            (torch.zeros((batch, 1), dtype=torch.bool, device=x.device), ~tract_valid), dim=1
        )
        encoded_tracts = self.encoder(tokens, src_key_padding_mask=padding_mask)

        if self.pooling == "cls":
            return encoded_tracts[:, 0]
        valid_encoded = encoded_tracts[:, 1:]
        if self.pooling == "mean":
            denominator = tract_valid.sum(dim=1, keepdim=True).clamp_min(1)
            return (valid_encoded * tract_valid.unsqueeze(-1)).sum(dim=1) / denominator
        scores = self.readout_score(valid_encoded).squeeze(-1)
        scores = scores.masked_fill(~tract_valid, torch.finfo(scores.dtype).min)
        no_tracts = ~tract_valid.any(dim=1)
        scores = torch.where(no_tracts[:, None], torch.zeros_like(scores), scores)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(tract_valid, weights, torch.zeros_like(weights))
        return torch.sum(valid_encoded * weights.unsqueeze(-1), dim=1)
