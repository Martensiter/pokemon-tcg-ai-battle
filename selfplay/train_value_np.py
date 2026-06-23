"""Train the value network in pure numpy and export weights.npz (no torch).

A torch-free counterpart to ``train_value.py`` for machines where torch is
unavailable -- e.g. the aarch64 collector device (SwitchBot AI Hub). The value
net is a tiny MLP (FEATURE_DIM->...->1), so a hand-rolled numpy MLP with Adam +
BCE-with-logits trains it in seconds on CPU.

Output is byte-identical in layout to ``train_value.py``: ``W1/b1 ... Wk/bk`` with
``W`` shaped ``(in, out)`` and a sigmoid applied at inference. ``agent/value_net.py``
loads it unchanged.

  uv run python selfplay/train_value_np.py --data selfplay/data_collected_all.npz \
      --out agent/weights.npz --epochs 60
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bce_with_logits(z: np.ndarray, y: np.ndarray) -> float:
    """Numerically stable mean binary cross-entropy from logits."""
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def train_mlp(X: np.ndarray, y: np.ndarray, hidden: list[int],
              epochs: int = 60, lr: float = 1e-3, batch: int = 512,
              val_frac: float = 0.1, weight_decay: float = 1e-5,
              seed: int = 0, verbose: bool = True):
    """Train an MLP [in]+hidden+[1] with Adam + BCE. Returns (weights, metrics).

    ``weights`` is the export dict ``{"W1":..,"b1":..,...}`` ready for
    ``np.savez``; ``metrics`` holds final ``val_loss``/``val_acc``.
    """
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)
    n, d = X.shape

    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    n_val = int(n * val_frac)
    Xva, yva = X[:n_val], y[:n_val]
    Xtr, ytr = X[n_val:], y[n_val:]
    if len(Xtr) == 0:                      # tiny datasets: train on everything
        Xtr, ytr, Xva, yva = X, y, X, y

    dims = [d] + list(hidden) + [1]
    # He initialisation for the ReLU stack; small init for the output layer.
    Ws, bs = [], []
    for i in range(len(dims) - 1):
        scale = np.sqrt(2.0 / dims[i]) if i < len(dims) - 2 else np.sqrt(1.0 / dims[i])
        Ws.append((rng.standard_normal((dims[i], dims[i + 1])) * scale).astype(np.float32))
        bs.append(np.zeros((dims[i + 1],), np.float32))

    # Adam state
    mW = [np.zeros_like(W) for W in Ws]; vW = [np.zeros_like(W) for W in Ws]
    mb = [np.zeros_like(b) for b in bs]; vb = [np.zeros_like(b) for b in bs]
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    L = len(Ws)

    def forward(xb):
        acts = [xb]
        pre = []
        h = xb
        for k in range(L):
            z = h @ Ws[k] + bs[k]
            pre.append(z)
            h = np.maximum(z, 0.0) if k < L - 1 else z   # last layer: logits
            acts.append(h)
        return acts, pre

    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), batch):
            bidx = idx[s:s + batch]
            xb, yb = Xtr[bidx], ytr[bidx]
            acts, pre = forward(xb)
            logits = acts[-1]
            # dL/dlogits for BCE-with-logits = sigmoid(z) - y, averaged over batch
            g = (_sigmoid(logits) - yb) / len(xb)
            t += 1
            for k in reversed(range(L)):
                gW = acts[k].T @ g + weight_decay * Ws[k]
                gb = g.sum(axis=0)
                # Adam update
                mW[k] = b1 * mW[k] + (1 - b1) * gW
                vW[k] = b2 * vW[k] + (1 - b2) * (gW * gW)
                mb[k] = b1 * mb[k] + (1 - b1) * gb
                vb[k] = b2 * vb[k] + (1 - b2) * (gb * gb)
                mWh = mW[k] / (1 - b1 ** t); vWh = vW[k] / (1 - b2 ** t)
                mbh = mb[k] / (1 - b1 ** t); vbh = vb[k] / (1 - b2 ** t)
                Ws[k] -= lr * mWh / (np.sqrt(vWh) + eps)
                bs[k] -= lr * mbh / (np.sqrt(vbh) + eps)
                if k > 0:
                    g = g @ Ws[k].T
                    g = g * (pre[k - 1] > 0)            # ReLU gradient
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            zva = Xva
            for k in range(L):
                zva = zva @ Ws[k] + bs[k]
                if k < L - 1:
                    zva = np.maximum(zva, 0.0)
            vl = _bce_with_logits(zva, yva)
            va = float(np.mean((_sigmoid(zva) > 0.5) == (yva > 0.5)))
            if verbose:
                print(f"epoch {ep+1:>3}: val_loss={vl:.4f} val_acc={va:.3f}", flush=True)

    # final metrics + export
    zva = Xva
    for k in range(L):
        zva = zva @ Ws[k] + bs[k]
        if k < L - 1:
            zva = np.maximum(zva, 0.0)
    metrics = {"val_loss": _bce_with_logits(zva, yva),
               "val_acc": float(np.mean((_sigmoid(zva) > 0.5) == (yva > 0.5)))}
    weights = {}
    for k in range(L):
        weights[f"W{k+1}"] = Ws[k].astype(np.float32)   # (in, out) — matches train_value.py
        weights[f"b{k+1}"] = bs[k].astype(np.float32)
    return weights, metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join(ROOT, "selfplay", "data.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "agent", "weights.npz"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    d = np.load(args.data)
    X, y = d["X"], d["y"]
    if len(y) == 0:
        raise SystemExit("no training data")
    print(f"training on {len(y)} states (dim={X.shape[1]}, hidden={args.hidden})")
    weights, metrics = train_mlp(X, y, args.hidden, epochs=args.epochs, lr=args.lr,
                                 batch=args.batch, val_frac=args.val_frac, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **weights)
    dims = [X.shape[1]] + list(args.hidden) + [1]
    print(f"saved {args.out} (dims={dims}); "
          f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
