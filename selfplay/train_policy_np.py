"""Train a behavioral-cloning POLICY net in pure numpy (no torch, no engine).

Counterpart to ``train_value_np.py`` for the distillation track. The records
(from ``collector.policy_extract``) are grouped: each decision has a variable set
of options, one of which the expert chose. The net scores each option from
``[state_features (32) | option_features]`` and we train a per-decision **softmax
cross-entropy** to the chosen option (a pointer/ranking objective, not a fixed
classifier -- option counts vary per decision).

  python selfplay/train_policy_np.py --data policy_data.npz --out agent/policy.npz

Export layout mirrors the value net (``W1/b1 ... Wk/bk``, ``W`` shaped (in,out),
ReLU between layers, a single linear score at the top), so a numpy inference path
can load it the same way. Integrating it into the agent is a separate, later step.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _offsets(group: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(group)]).astype(np.int64)


def _grouped_softmax_ce(scores: np.ndarray, group: np.ndarray, chosen: np.ndarray):
    """Mean per-decision softmax CE + top-1 accuracy + per-option gradient.

    ``scores`` is (M,) over all options; ``group`` (G,) sizes; ``chosen`` (G,)
    index within each group. Returns ``(loss, acc, dscores)``.
    """
    off = _offsets(group)
    G = len(group)
    dscores = np.zeros_like(scores)
    loss = 0.0
    correct = 0
    for g in range(G):
        a, b = off[g], off[g + 1]
        s = scores[a:b]
        s = s - s.max()
        e = np.exp(s)
        p = e / e.sum()
        c = int(chosen[g])
        loss += -np.log(p[c] + 1e-12)
        dscores[a:b] = p
        dscores[a + c] -= 1.0
        if int(np.argmax(p)) == c:
            correct += 1
    if G:
        loss /= G
        dscores /= G
    return loss, (correct / G if G else 0.0), dscores


def train_policy(state: np.ndarray, opt: np.ndarray, group: np.ndarray,
                 chosen: np.ndarray, hidden: list[int], epochs: int = 60,
                 lr: float = 1e-3, val_frac: float = 0.1, weight_decay: float = 1e-5,
                 seed: int = 0, verbose: bool = True):
    """Train an option-scoring MLP with grouped softmax CE (Adam). -> (weights, metrics)."""
    rng = np.random.default_rng(seed)
    state = state.astype(np.float32)
    opt = opt.astype(np.float32)
    group = group.astype(np.int64)
    chosen = chosen.astype(np.int64)

    # Per-option input = [state (repeated per group) | option features].
    state_rep = np.repeat(state, group, axis=0)
    X = np.concatenate([state_rep, opt], axis=1).astype(np.float32)
    d = X.shape[1]

    # Split by DECISION (group), not by option row.
    G = len(group)
    perm = rng.permutation(G)
    n_val = int(G * val_frac)
    val_g, tr_g = perm[:n_val], perm[n_val:]
    if len(tr_g) == 0:
        tr_g = perm
        val_g = perm

    off = _offsets(group)

    def slice_rows(gids):
        rows = np.concatenate([np.arange(off[g], off[g + 1]) for g in gids]) if len(gids) else np.array([], int)
        return rows

    def subset(gids):
        gids = np.sort(gids)
        rows = slice_rows(gids)
        return X[rows], group[gids], chosen[gids]

    Xtr, gtr, ctr = subset(tr_g)
    Xva, gva, cva = subset(val_g)

    dims = [d] + list(hidden) + [1]
    Ws, bs = [], []
    for i in range(len(dims) - 1):
        scale = np.sqrt(2.0 / dims[i]) if i < len(dims) - 2 else np.sqrt(1.0 / dims[i])
        Ws.append((rng.standard_normal((dims[i], dims[i + 1])) * scale).astype(np.float32))
        bs.append(np.zeros((dims[i + 1],), np.float32))
    mW = [np.zeros_like(W) for W in Ws]; vW = [np.zeros_like(W) for W in Ws]
    mb = [np.zeros_like(b) for b in bs]; vb = [np.zeros_like(b) for b in bs]
    b1, b2, eps = 0.9, 0.999, 1e-8
    L = len(Ws)
    t = 0

    def forward(xb):
        acts = [xb]; pre = []
        h = xb
        for k in range(L):
            z = h @ Ws[k] + bs[k]
            pre.append(z)
            h = np.maximum(z, 0) if k < L - 1 else z
            acts.append(h)
        return acts[-1][:, 0], acts, pre

    def backward(acts, pre, dscore):
        grads_W = [None] * L; grads_b = [None] * L
        dh = dscore.reshape(-1, 1)
        for k in range(L - 1, -1, -1):
            if k < L - 1:
                dh = dh * (pre[k] > 0)
            grads_W[k] = acts[k].T @ dh + weight_decay * Ws[k]
            grads_b[k] = dh.sum(axis=0)
            dh = dh @ Ws[k].T
        return grads_W, grads_b

    for ep in range(epochs):
        scores, acts, pre = forward(Xtr)
        loss, acc, dscores = _grouped_softmax_ce(scores, gtr, ctr)
        gW, gb = backward(acts, pre, dscores)
        t += 1
        for k in range(L):
            for g_, m_, v_, p_ in ((gW[k], mW, vW, Ws), (gb[k], mb, vb, bs)):
                m_[k] = b1 * m_[k] + (1 - b1) * g_
                v_[k] = b2 * v_[k] + (1 - b2) * (g_ * g_)
                mhat = m_[k] / (1 - b1 ** t)
                vhat = v_[k] / (1 - b2 ** t)
                p_[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            vs, _, _ = forward(Xva)
            vloss, vacc, _ = _grouped_softmax_ce(vs, gva, cva)
            print(f"epoch {ep:3d}: train_loss={loss:.4f} acc={acc:.3f} | "
                  f"val_loss={vloss:.4f} val_acc={vacc:.3f}")

    vs, _, _ = forward(Xva)
    vloss, vacc, _ = _grouped_softmax_ce(vs, gva, cva)
    weights = {}
    for k in range(L):
        weights[f"W{k+1}"] = Ws[k]
        weights[f"b{k+1}"] = bs[k]
    metrics = {"val_loss": float(vloss), "val_acc": float(vacc),
               "decisions": int(G), "options": int(len(X)), "in_dim": int(d)}
    return weights, metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="policy npz (state/opt/group/chosen)")
    ap.add_argument("--out", default=os.path.join(ROOT, "agent", "policy.npz"))
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args(argv)

    d = np.load(args.data)
    if len(d["group"]) == 0:
        print("no decisions in data")
        return 1
    weights, metrics = train_policy(d["state"], d["opt"], d["group"], d["chosen"],
                                    args.hidden, epochs=args.epochs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **weights)
    print(f"saved {args.out}  val_acc={metrics['val_acc']:.3f} "
          f"(decisions={metrics['decisions']}, in_dim={metrics['in_dim']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
