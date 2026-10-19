"""Variable-band spectral Transformer encoder."""

from __future__ import annotations

import torch
from torch import nn

from .tokens import SpectralTokenizer


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class VariableBandEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.08,
        pe_dim: int = 32,
        patch_size: int = 1,
    ):
        super().__init__()
        self.patch_size = max(int(patch_size), 1)
        self.tokenizer = SpectralTokenizer(dim=dim, pe_dim=pe_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls, std=0.02)

    @staticmethod
    def _pool(x: torch.Tensor, patch: int, kind: str = "mean") -> torch.Tensor:
        if patch <= 1:
            return x
        b, n = x.shape[:2]
        pad_n = (patch - n % patch) % patch
        if pad_n:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_n) if x.dim() == 3 else (0, pad_n))
            n = x.shape[1]
        x = x.view(b, n // patch, patch, *x.shape[2:])
        if kind == "max":
            return x.max(dim=2).values
        return x.mean(dim=2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p = self.patch_size
        r = self._pool(batch["reflectance"], p)
        wl = self._pool(batch["wavelength"], p)
        fw = self._pool(batch["fwhm"], p)
        sg = self._pool(batch["sigma"], p)
        va = self._pool(batch["valid"], p)
        tok = self.tokenizer(r, wl, fw, sg, va)
        bsz = tok.shape[0]
        x = torch.cat([self.cls.expand(bsz, -1, -1), tok], dim=1)
        pad = va < 0.5
        pad = torch.cat([torch.zeros(bsz, 1, dtype=torch.bool, device=tok.device), pad], dim=1)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad)
        x = self.norm(x)
        return {"tokens": x[:, 1:], "cls": x[:, 0], "full": x}
