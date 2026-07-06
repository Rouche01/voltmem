"""
Volatility-Adjusted EWC vs Baseline EWC — v2 (shared weights, honest test)
============================================================================

Fixes from v1:
  1. SINGLE shared head/trunk for both domains -> stable and volatile tasks
     genuinely compete for the same weights (no dedicated capacity to escape
     interference). This makes forgetting an actual risk for both methods.

  2. Volatility is estimated BEFORE any gradient step / EWC penalty is
     applied (pre-update loss on the new task using only knowledge from
     prior tasks of that domain) -> measures the domain's true drift, not
     an artifact of how hard EWC was clamping the weights.

  3. EWC penalty terms are now scored per STORED PAST TASK, weighted by
     that task's domain's volatility -> stable-domain memories get strongly
     protected, volatile-domain memories get weakly protected (since they're
     expected to go stale anyway, protecting them hard just wastes capacity
     and blocks adaptation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

DEVICE = "cpu"

# ----------------------------------------------------------------------------
# 1. Synthetic task generator (single shared 2-class head)
# ----------------------------------------------------------------------------
# Stable domain: decision boundary barely moves task to task (small jitter).
# Volatile domain: decision boundary relocates a lot each task (large jitter).
# Both domains use the SAME two output classes -> genuine competition for
# the same head + trunk weights.

def make_task(domain, n=300, seed=0):
    rng = np.random.RandomState(seed)
    if domain == "stable":
        base_a = np.array([-2.0, 0.0])
        base_b = np.array([2.0, 0.0])
        jitter = 0.15
        ca = base_a + rng.randn(2) * jitter
        cb = base_b + rng.randn(2) * jitter
    else:  # volatile
        ca = rng.randn(2) * 3.0
        cb = rng.randn(2) * 3.0
        while np.linalg.norm(ca - cb) < 2.0:
            cb = rng.randn(2) * 3.0

    xa = ca + rng.randn(n // 2, 2) * 0.6
    xb = cb + rng.randn(n // 2, 2) * 0.6
    X = np.vstack([xa, xb]).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    return torch.tensor(X), torch.tensor(y)


# ----------------------------------------------------------------------------
# 2. Model: fully shared trunk + single shared head
# ----------------------------------------------------------------------------

class SharedNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# 3. EWC bookkeeping
# ----------------------------------------------------------------------------

def compute_fisher(model, X, n_samples=300):
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    n_samples = min(n_samples, X.shape[0])
    idx = torch.randperm(X.shape[0])[:n_samples]
    for i in idx:
        model.zero_grad()
        out = model(X[i:i+1])
        logp = F.log_softmax(out, dim=1)
        sampled = torch.multinomial(logp.exp(), 1).item()
        loss = -logp[0, sampled]
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] /= n_samples
    return fisher


def ewc_penalty(model, memory):
    """
    memory: list of dicts {fisher, star, weight} — weight is the
    domain-volatility-derived protection strength for that stored task.
    """
    if not memory:
        return torch.tensor(0.0)
    penalty = 0.0
    for item in memory:
        fisher, star, w = item["fisher"], item["star"], item["weight"]
        for n, p in model.named_parameters():
            penalty = penalty + w * (fisher[n] * (p - star[n]) ** 2).sum()
    return penalty


def pre_update_loss(model, X, y):
    """Loss on a fresh task BEFORE any training on it — the mismatch signal,
    measured independent of EWC strength."""
    model.eval()
    with torch.no_grad():
        out = model(X)
        loss = F.cross_entropy(out, y)
    return loss.item()


def train_task(model, X, y, memory, lr=0.03, epochs=15, ewc_lambda=3000.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X)
        ce = F.cross_entropy(out, y, label_smoothing=0.1)
        penalty = ewc_penalty(model, memory)
        loss = ce + ewc_lambda * penalty
        loss.backward()
        opt.step()
    return model


def eval_task(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        pred = out.argmax(dim=1)
        return (pred == y).float().mean().item()


# ----------------------------------------------------------------------------
# 4. Continual learning run
# ----------------------------------------------------------------------------

def run_experiment(mode, n_tasks=12, gamma=2.0, beta=0.5, seed=0):
    """
    mode: 'baseline' (every stored task protected equally) or
          'volatility' (protection per stored task scaled by that task's
          domain volatility, estimated from pre-update loss)
    """
    torch.manual_seed(seed)
    model = SharedNet()
    memory = []  # list of {fisher, star, weight, domain}

    # running volatility estimate per domain, seeded at neutral
    volatility = {"stable": 1.0, "volatile": 1.0}

    stable_tasks, volatile_tasks = [], []

    for t in range(n_tasks):
        domain = "stable" if t % 2 == 0 else "volatile"
        X, y = make_task(domain, seed=seed * 100 + t)

        if domain == "stable":
            stable_tasks.append((X, y))
        else:
            volatile_tasks.append((X, y))

        # ---- measure mismatch BEFORE training (independent of EWC) ----
        mismatch = pre_update_loss(model, X, y)
        volatility[domain] = beta * volatility[domain] + (1 - beta) * mismatch

        # ---- train, using current memory with current protection weights ----
        train_task(model, X, y, memory)

        # ---- consolidate this task into memory ----
        fisher = compute_fisher(model, X)
        star = {n: p.detach().clone() for n, p in model.named_parameters()}

        if mode == "baseline":
            w = 1.0
        else:  # volatility-adjusted: stable memories protected hard,
               # volatile memories protected weakly
            w = 1.0 / (volatility[domain] ** gamma)
            w = float(np.clip(w, 0.05, 20.0))  # keep numerically sane

        memory.append({"fisher": fisher, "star": star, "weight": w, "domain": domain})

    early_stable = np.mean([eval_task(model, X, y) for X, y in stable_tasks[:2]])
    late_volatile = np.mean([eval_task(model, X, y) for X, y in volatile_tasks[-2:]])
    avg_stable = np.mean([eval_task(model, X, y) for X, y in stable_tasks])
    avg_volatile = np.mean([eval_task(model, X, y) for X, y in volatile_tasks])

    return {
        "early_stable_retention": early_stable,
        "late_volatile_adaptation": late_volatile,
        "avg_stable_acc": avg_stable,
        "avg_volatile_acc": avg_volatile,
        "final_volatility": dict(volatility),
    }


# ----------------------------------------------------------------------------
# 5. Run comparison
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    N_TASKS = 12
    N_RUNS = 8

    results = {"baseline": [], "volatility": []}
    for run in range(N_RUNS):
        for mode in ["baseline", "volatility"]:
            res = run_experiment(mode, n_tasks=N_TASKS, seed=run)
            results[mode].append(res)

    print("=" * 72)
    print(f"v2 — shared-weight test, averaged over {N_RUNS} runs, {N_TASKS} tasks")
    print("=" * 72)

    for mode in ["baseline", "volatility"]:
        rs = results[mode]
        early_stable = np.mean([r["early_stable_retention"] for r in rs])
        late_volatile = np.mean([r["late_volatile_adaptation"] for r in rs])
        avg_stable = np.mean([r["avg_stable_acc"] for r in rs])
        avg_volatile = np.mean([r["avg_volatile_acc"] for r in rs])
        print(f"\n--- Mode: {mode.upper()} ---")
        print(f"  Early-task STABLE retention (forgetting test): {early_stable:.3f}")
        print(f"  Late-task VOLATILE adaptation:                 {late_volatile:.3f}")
        print(f"  Avg accuracy across ALL stable tasks:          {avg_stable:.3f}")
        print(f"  Avg accuracy across ALL volatile tasks:        {avg_volatile:.3f}")

    print("\nLearned volatility estimates (volatility-mode, run 0):")
    for d, v in results["volatility"][0]["final_volatility"].items():
        print(f"  {d:10s}: {v:.3f}")

    print("\n" + "=" * 72)
    print("Prediction: volatility-adjusted should beat baseline on BOTH")
    print("early-stable retention AND late-volatile adaptation, since it")
    print("spends protection budget where it's warranted instead of uniformly.")
    print("=" * 72)
