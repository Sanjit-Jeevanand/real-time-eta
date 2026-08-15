"""Multi-head quantile network -- step 2's comparator, and fix 3.

One network, three heads, one joint pinball loss. The `ordered` flag is what makes
this both things at once:

* `ordered=False` -- three independent heads. A neural comparator to LightGBM that
  can cross exactly like the boosters do.
* `ordered=True` -- fix 3. Heads emit a base and two softplus increments, so the
  outputs are non-decreasing *by construction*. There is no crossing to measure and
  no post-hoc repair step, because the parameterisation cannot express a crossing.

The ordering is structural, not a penalty: nothing is being traded off, and no
tolerance can be tuned wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl
    from numpy.typing import NDArray

    from eta.types import Quantile

__all__ = ["MultiHeadQuantileNet", "pick_device", "pinball_loss_torch"]

log = get_logger(__name__)

TARGET = "total_time_s"


def pick_device(prefer: str = "auto") -> str:
    """MPS on the M4 when it is there, CPU otherwise."""
    import torch

    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pinball_loss_torch(pred: Any, target: Any, alphas: Any) -> Any:
    """Joint pinball loss: mean over rows and over quantile levels.

    `pred` is (batch, n_alphas); `target` is (batch, 1) and broadcasts across heads.
    Summing the per-level losses is what makes this one objective rather than three
    models sharing a trunk by coincidence.
    """
    import torch

    delta = target - pred
    return torch.mean(torch.maximum(alphas * delta, (alphas - 1.0) * delta))


@dataclass(slots=True)
class MultiHeadQuantileNet:
    """MLP trunk with one head per quantile level."""

    alphas: tuple[Quantile, ...]
    features: Sequence[str]
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 8192
    max_epochs: int = 12
    patience: int = 3
    ordered: bool = False
    device: str = "auto"
    name: str = "nn_quantile"
    _model: Any = None
    _mu: Any = None
    _sigma: Any = None
    _y_scale: float = 1.0
    _device: str = "cpu"

    def _build(self, n_in: int) -> Any:
        import torch
        from torch import nn

        n_out = len(self.alphas)
        ordered = self.ordered

        class Net(nn.Module):
            def __init__(self, widths: tuple[int, ...], p_drop: float) -> None:
                super().__init__()
                layers: list[nn.Module] = []
                prev = n_in
                for width in widths:
                    layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(p_drop)]
                    prev = width
                self.trunk = nn.Sequential(*layers)
                self.head = nn.Linear(prev, n_out)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                raw: torch.Tensor = self.head(self.trunk(x))
                if not ordered:
                    return raw
                # base + cumulative softplus increments: non-decreasing by construction.
                base = raw[:, :1]
                steps = nn.functional.softplus(raw[:, 1:])
                return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)

        return Net(self.hidden, self.dropout)

    def fit(self, train: pl.DataFrame, seed: int, valid: pl.DataFrame | None = None) -> None:
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        self._device = pick_device(self.device)

        x = train.select(self.features).to_numpy().astype(np.float32)
        y = train[TARGET].to_numpy().astype(np.float32)

        # Standardise inputs and scale the target; pinball on raw seconds is badly
        # conditioned and the net spends its first epochs learning the intercept.
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self._mu = x.mean(axis=0, keepdims=True)
        self._sigma = x.std(axis=0, keepdims=True)
        self._sigma[self._sigma < 1e-6] = 1.0
        self._y_scale = float(np.median(y)) or 1.0

        model = self._build(x.shape[1]).to(self._device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        alphas_t = torch.tensor(
            [float(a) for a in self.alphas], dtype=torch.float32, device=self._device
        ).reshape(1, -1)

        xt = torch.from_numpy((x - self._mu) / self._sigma)
        yt = torch.from_numpy(y / self._y_scale).reshape(-1, 1)

        vxt = vyt = None
        if valid is not None:
            vx = valid.select(self.features).to_numpy().astype(np.float32)
            vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
            vy = valid[TARGET].to_numpy().astype(np.float32)
            vxt = torch.from_numpy((vx - self._mu) / self._sigma).to(self._device)
            vyt = torch.from_numpy(vy / self._y_scale).reshape(-1, 1).to(self._device)

        n = xt.shape[0]
        best = float("inf")
        best_state: dict[str, Any] | None = None
        stale = 0

        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(n)
            total = 0.0
            batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = xt[idx].to(self._device)
                yb = yt[idx].to(self._device)
                opt.zero_grad()
                loss = pinball_loss_torch(model(xb), yb, alphas_t)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu())
                batches += 1

            train_loss = total / max(batches, 1)
            if vxt is None or vyt is None:
                log.info("nn_epoch", epoch=epoch, train_pinball=round(train_loss, 5))
                continue

            model.eval()
            with torch.no_grad():
                chunks = [
                    model(vxt[s : s + self.batch_size])
                    for s in range(0, vxt.shape[0], self.batch_size)
                ]
                val_loss = float(pinball_loss_torch(torch.cat(chunks), vyt, alphas_t).cpu())
            log.info(
                "nn_epoch",
                epoch=epoch,
                train_pinball=round(train_loss, 5),
                val_pinball=round(val_loss, 5),
            )
            if val_loss < best - 1e-6:
                best = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    log.info("nn_early_stopped", epoch=epoch, best_val_pinball=round(best, 5))
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        log.info(
            "nn_fitted",
            seed=seed,
            ordered=self.ordered,
            device=self._device,
            rows=train.height,
            best_val_pinball=(round(best, 5) if best < float("inf") else None),
        )

    def predict_matrix(self, frame: pl.DataFrame) -> NDArray[np.float64]:
        import torch

        if self._model is None:
            msg = f"{self.name} has not been fitted"
            raise RuntimeError(msg)
        x = frame.select(self.features).to_numpy().astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xt = torch.from_numpy((x - self._mu) / self._sigma)
        out: list[NDArray[np.float64]] = []
        with torch.no_grad():
            for start in range(0, xt.shape[0], self.batch_size):
                chunk = xt[start : start + self.batch_size].to(self._device)
                out.append(self._model(chunk).cpu().numpy().astype(np.float64))
        return np.vstack(out) * self._y_scale

    def predict(self, frame: pl.DataFrame, alpha: Quantile) -> NDArray[np.float64]:
        return self.predict_matrix(frame)[:, self.alphas.index(alpha)]
