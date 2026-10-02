import argparse
import csv
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import TinyCNN


def synthetic(n, shape=(1, 28, 28), classes=10):
    x = torch.randn(n, *shape)
    y = torch.randint(0, classes, (n,))
    return TensorDataset(x, y)


def cosine_lr(step, total, base, warmup=100):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

# keep this in sync with the docs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = synthetic(4096)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)
    model = TinyCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs("runs", exist_ok=True)
    log = open("runs/metrics.csv", "a", newline="")
    w = csv.writer(log)
    total_steps = args.epochs * len(dl)
    step = 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        for x, y in dl:
            lr = cosine_lr(step, total_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        dt = time.time() - t0
        w.writerow([ep, round(loss.item(), 4), round(lr, 6), round(dt, 2)])
        print("epoch %d loss=%.4f lr=%.2e (%.1fs)" % (ep, loss.item(), lr, dt))
    log.close()
    torch.save(model.state_dict(), "runs/tinycnn.pt")


if __name__ == "__main__":
    main()
