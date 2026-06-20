"""Train the value network with torch and export numpy weights for inference.

Input: selfplay/data.npz (X, y) from gen_data.py. Output: agent/weights.npz with
W1/b1 ... Wk/bk that agent/value_net.py executes in pure numpy. A small MLP with
sigmoid output trained with BCE to predict P(to-move player wins).
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "selfplay", "data.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "agent", "weights.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    d = np.load(args.data)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32).unsqueeze(1)
    n = X.shape[0]
    assert n > 0, "no training data"

    # shuffle + split
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    X, y = X[perm], y[perm]
    n_val = int(n * args.val_frac)
    Xtr, ytr = X[n_val:], y[n_val:]
    Xva, yva = X[:n_val], y[:n_val]

    dims = [X.shape[1]] + args.hidden + [1]
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    net = nn.Sequential(*layers)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    def evaluate(Xs, ys):
        net.eval()
        with torch.no_grad():
            logits = net(Xs)
            loss = lossf(logits, ys).item()
            acc = ((torch.sigmoid(logits) > 0.5).float() == ys).float().mean().item()
        return loss, acc

    for ep in range(args.epochs):
        net.train()
        idx = torch.randperm(Xtr.shape[0])
        for s in range(0, Xtr.shape[0], args.batch):
            b = idx[s:s + args.batch]
            opt.zero_grad()
            loss = lossf(net(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            vl, va = evaluate(Xva, yva) if n_val else evaluate(Xtr, ytr)
            print(f"epoch {ep+1:>3}: val_loss={vl:.4f} val_acc={va:.3f}")

    # export to numpy npz (sigmoid applied at inference)
    out = {}
    li = 0
    for m in net:
        if isinstance(m, nn.Linear):
            li += 1
            out[f"W{li}"] = m.weight.detach().numpy().T.astype(np.float32)  # (in, out)
            out[f"b{li}"] = m.bias.detach().numpy().astype(np.float32)
    np.savez(args.out, **out)
    vl, va = evaluate(Xva, yva) if n_val else evaluate(Xtr, ytr)
    print(f"saved {args.out} ({li} layers, dims={dims}); final val_loss={vl:.4f} val_acc={va:.3f}")


if __name__ == "__main__":
    main()
