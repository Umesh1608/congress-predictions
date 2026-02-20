"""FT-Transformer predictor for tabular + GNN embedding features.

Uses rtdl_revisiting_models.FTTransformer — the official implementation from
"Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021).
Designed to work with tabular features augmented by GNN embeddings.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.ml.models.base import BasePredictor

logger = logging.getLogger(__name__)


def _get_ft_transformer():
    """Lazy import FTTransformer."""
    try:
        from rtdl_revisiting_models import FTTransformer
    except ImportError:
        raise ImportError(
            "rtdl-revisiting-models is required for FTTransformerPredictor. "
            "Install with: pip install rtdl-revisiting-models"
        )
    return FTTransformer


class FTTransformerPredictor(BasePredictor):
    """FT-Transformer classifier for trade profitability prediction.

    Architecture: Feature Tokenizer + Transformer encoder for tabular data.
    Supports continuous features (no categorical features in our pipeline).
    """

    model_name = "ft_transformer"

    def __init__(
        self,
        n_blocks: int = 3,
        d_block: int = 192,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 15,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self._train_mean: np.ndarray | None = None
        self._train_std: np.ndarray | None = None
        self._n_features: int = 0

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using training statistics."""
        if self._train_mean is None or self._train_std is None:
            return X
        return (X - self._train_mean) / (self._train_std + 1e-8)

    def _build_model(self, n_features: int) -> Any:
        """Create FTTransformer model."""
        FTTransformer = _get_ft_transformer()
        return FTTransformer(
            n_cont_features=n_features,
            cat_cardinalities=[],
            d_out=2,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train FT-Transformer with early stopping on val AUC."""
        self._n_features = X_train.shape[1]

        # Compute normalization stats from training data
        self._train_mean = np.nanmean(X_train, axis=0)
        self._train_std = np.nanstd(X_train, axis=0)

        X_train_norm = self._normalize(np.nan_to_num(X_train, nan=0.0))

        self.model = self._build_model(self._n_features)
        self.model = self.model.to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        # Prepare tensors
        X_t = torch.tensor(X_train_norm, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        w_t = (
            torch.tensor(sample_weight, dtype=torch.float32)
            if sample_weight is not None
            else None
        )

        if X_val is not None and y_val is not None:
            X_val_norm = self._normalize(np.nan_to_num(X_val, nan=0.0))
            X_v = torch.tensor(X_val_norm, dtype=torch.float32).to(self.device)
            y_v = torch.tensor(y_val, dtype=torch.long).to(self.device)
        else:
            X_v = y_v = None

        best_val_auc = 0.0
        best_state = None
        patience_counter = 0
        n = len(X_t)

        for epoch in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i: i + self.batch_size]
                xb = X_t[idx].to(self.device)
                yb = y_t[idx].to(self.device)

                logits = self.model(xb, None)
                if w_t is not None:
                    wb = w_t[idx].to(self.device)
                    loss = torch.nn.functional.cross_entropy(logits, yb, reduction="none")
                    loss = (loss * wb).mean()
                else:
                    loss = torch.nn.functional.cross_entropy(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            if X_v is not None and y_v is not None and (epoch + 1) % 5 == 0:
                from sklearn.metrics import roc_auc_score

                self.model.eval()
                with torch.no_grad():
                    val_proba = self._predict_proba_tensor(X_v)
                try:
                    val_auc = roc_auc_score(y_v.cpu().numpy(), val_proba[:, 1].cpu().numpy())
                except ValueError:
                    val_auc = 0.5

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 20 == 0:
                    logger.info(
                        "FT-Transformer epoch %d/%d: loss=%.4f, val_auc=%.4f (best=%.4f)",
                        epoch + 1, self.max_epochs,
                        epoch_loss / max(n_batches, 1), val_auc, best_val_auc,
                    )

                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d (best val_auc=%.4f)", epoch + 1, best_val_auc)
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        # Compute final metrics
        from src.ml.evaluation import evaluate_classifier

        if X_val is not None and y_val is not None:
            val_preds = self.predict(X_val)
            val_proba = self.predict_proba(X_val)
            metrics = evaluate_classifier(y_val, val_preds, val_proba)
        else:
            train_preds = self.predict(X_train)
            train_proba = self.predict_proba(X_train)
            metrics = evaluate_classifier(y_train, train_preds, train_proba)

        logger.info("FTTransformerPredictor trained: %s", metrics)
        return metrics

    def _predict_proba_tensor(self, X: Tensor) -> Tensor:
        """Internal: compute class probabilities from tensor input."""
        self.model.eval()
        all_proba = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = X[i: i + self.batch_size]
                logits = self.model(xb, None)
                proba = torch.softmax(logits, dim=1)
                all_proba.append(proba)
        return torch.cat(all_proba, dim=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0 or 1)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class 1 (profitable)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X_norm = self._normalize(np.nan_to_num(X, nan=0.0))
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        proba = self._predict_proba_tensor(X_t)
        return proba[:, 1].cpu().numpy()

    def save(self, path: str) -> None:
        """Save model state, normalization stats, and config."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "state_dict": self.model.state_dict() if self.model else None,
            "train_mean": self._train_mean,
            "train_std": self._train_std,
            "n_features": self._n_features,
            "feature_columns": self.feature_columns,
            "config": {
                "n_blocks": self.n_blocks,
                "d_block": self.d_block,
                "attention_n_heads": self.attention_n_heads,
                "attention_dropout": self.attention_dropout,
                "ffn_dropout": self.ffn_dropout,
                "residual_dropout": self.residual_dropout,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load model from saved state."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        self._train_mean = data["train_mean"]
        self._train_std = data["train_std"]
        self._n_features = data["n_features"]
        self.feature_columns = data.get("feature_columns", [])

        config = data.get("config", {})
        self.n_blocks = config.get("n_blocks", 3)
        self.d_block = config.get("d_block", 192)
        self.attention_n_heads = config.get("attention_n_heads", 8)
        self.attention_dropout = config.get("attention_dropout", 0.2)
        self.ffn_dropout = config.get("ffn_dropout", 0.1)
        self.residual_dropout = config.get("residual_dropout", 0.0)

        if data["state_dict"] is not None and self._n_features > 0:
            self.model = self._build_model(self._n_features)
            self.model.load_state_dict(data["state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()
