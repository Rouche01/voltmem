"""
Volatility-Adjusted EWC vs Baseline EWC — Continual Learning Demo
===================================================================

Synthetic continual-learning benchmark (avoids needing to download MNIST):

We simulate a stream of tasks split into two domain types:
  - STABLE domain tasks: underlying data distribution barely shifts across tasks
    (low volatility) -> SHOULD be protected hard against forgetting.
  - VOLATILE domain tasks: underlying data distribution shifts a lot from
    task to task (high volatility) -> SHOULD adapt fast, low protection.

Each "task" is a small classification problem (Gaussian blobs in 2D feature
space, fed through a small shared MLP with two parameter groups: one feeding
the "stable" output head, one feeding the "volatile" output head).

We compare:
  1. Baseline EWC      - uniform Fisher-weighted penalty across all params
  2. Volatility-EWC    - per-group penalty scaled by a running volatility
                         estimate (high volatility group -> less protection)

Metric: after training on all tasks sequentially, measure accuracy retained
on EARLY stable-domain tasks (forgetting) and accuracy reached on LATE
volatile-domain tasks (adaptability).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

DEVICE = "cpu"

# ----------------------------------------------------------------------------
# 1. Synthetic task generator
# ----------------------------------------------------------------------------
# Each task is binary classification on 2D points drawn from two Gaussian
# blobs. "Stable" domain tasks reuse nearly the same blob centers every time
# (low volatility, the true decision boundary barely moves).
# "Volatile" domain tasks redraw blob centers randomly each task (high
# volatility, the true decision boundary moves a lot).

def make_task(domain, n=300, seed=0):
    rng = np.random.RandomState(seed)
    if domain == "stable":
        # Centers barely jitter around a fixed point across tasks
        base_a = np.array([-2.0, 0.0])
        base_b = np.array([2.0, 0.0])
        jitter = 0.15
        ca = base_a + rng.randn(2) * jitter
        cb = base_b + rng.randn(2) * jitter
    else:  # volatile
        # Centers are placed almost anywhere each task -> boundary moves a lot
        ca = rng.randn(2) * 3.0
        cb = rng.randn(2) * 3.0
        # ensure some separation so task is learnable
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
# 2. Model: shared trunk + two "domain group" heads
# ----------------------------------------------------------------------------
# Parameter groups (for EWC bookkeeping):
#   group 'shared' : trunk layers (low-level feature extraction)
#   group 'stable' : head used only for stable-domain tasks
#   group 'volatile': head used only for volatile-domain tasks

class DomainNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_stable = nn.Linear(hidden, 2)
        self.head_volatile = nn.Linear(hidden, 2)

    def forward(self, x, domain):
        z = self.trunk(x)
        if domain == "stable":
            return self.head_stable(z)
        else:
            return self.head_volatile(z)

    def named_param_groups(self):
        """Map each parameter to a bookkeeping group name."""
        groups = {}
        for n, p in self.named_parameters():
            if n.startswith("trunk"):
                groups[n] = "shared"
            elif n.startswith("head_stable"):
                groups[n] = "stable"
            elif n.startswith("head_volatile"):
                groups[n] = "volatile"
        return groups


# ----------------------------------------------------------------------------
# 3. EWC bookkeeping
# ----------------------------------------------------------------------------

def compute_fisher(model, X, y, domain, n_samples=300):
    """Empirical Fisher information (diagonal) via squared gradients of NLL."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    n_samples = min(n_samples, X.shape[0])
    idx = torch.randperm(X.shape[0])[:n_samples]
    for i in idx:
        model.zero_grad()
        out = model(X[i:i+1], domain)
        logp = F.log_softmax(out, dim=1)
        # sample label from model's own predictive distribution (true empirical Fisher)
        sampled = torch.multinomial(logp.exp(), 1).item()
        loss = -logp[0, sampled]
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] /= n_samples
    return fisher


def ewc_penalty(model, fisher_list, star_list, group_scale, param_groups):
    """
    fisher_list / star_list: lists across previously-audited tasks, each a
    dict {param_name: tensor}.
    group_scale: dict {group_name: scale_factor}  (1.0 for baseline EWC,
                 1/volatility^gamma for volatility-adjusted EWC)
    """
    if not fisher_list:
        return torch.tensor(0.0)
    penalty = 0.0
    for fisher, star in zip(fisher_list, star_list):
        for n, p in model.named_parameters():
            g = param_groups[n]
            scale = group_scale.get(g, 1.0)
            penalty = penalty + scale * (fisher[n] * (p - star[n]) ** 2).sum()
    return penalty


# ----------------------------------------------------------------------------
# 4. Training loop for one continual-learning run
# ----------------------------------------------------------------------------

def train_task(model, X, y, domain, fisher_list, star_list, group_scale,
                param_groups, lr=0.05, epochs=60, ewc_lambda=400.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X, domain)
        ce = F.cross_entropy(out, y)
        penalty = ewc_penalty(model, fisher_list, star_list, group_scale, param_groups)
        loss = ce + ewc_lambda * penalty
        loss.backward()
        opt.step()
    return model


def eval_task(model, X, y, domain):
    model.eval()
    with torch.no_grad():
        out = model(X, domain)
        pred = out.argmax(dim=1)
        return (pred == y).float().mean().item()


def run_experiment(mode, n_tasks=10, gamma=2.0):
    """
    mode: 'baseline' (uniform EWC) or 'volatility' (volatility-scaled EWC)
    Task stream: alternates stable / volatile domain tasks.
    """
    model = DomainNet()
    param_groups = model.named_param_groups()

    fisher_list, star_list = [], []
    volatility = {"shared": 1.0, "stable": 1.0, "volatile": 1.0}  # running EMA
    beta = 0.6  # EMA decay for volatility tracking

    stable_tasks = []   # store (X,y) for stable tasks to test forgetting later
    volatile_tasks = []

    prev_star = {n: p.detach().clone() for n, p in model.named_parameters()}

    for t in range(n_tasks):
        domain = "stable" if t % 2 == 0 else "volatile"
        X, y = make_task(domain, seed=t)

        if domain == "stable":
            stable_tasks.append((X, y))
        else:
            volatile_tasks.append((X, y))

        # ---- set group_scale based on mode ----
        if mode == "baseline":
            group_scale = {"shared": 1.0, "stable": 1.0, "volatile": 1.0}
        else:  # volatility-adjusted
            group_scale = {
                g: 1.0 / (volatility[g] ** gamma) for g in volatility
            }

        # ---- train on this task with EWC penalty from previous tasks ----
        train_task(model, X, y, domain, fisher_list, star_list, group_scale, param_groups)

        # ---- update volatility estimate: how much did params actually move ----
        with torch.no_grad():
            group_delta = {"shared": 0.0, "stable": 0.0, "volatile": 0.0}
            group_count = {"shared": 0, "stable": 0, "volatile": 0}
            for n, p in model.named_parameters():
                g = param_groups[n]
                delta = (p.detach() - prev_star[n]).abs().mean().item()
                group_delta[g] += delta
                group_count[g] += 1
            for g in volatility:
                if group_count[g] > 0:
                    observed = group_delta[g] / group_count[g]
                    volatility[g] = beta * volatility[g] + (1 - beta) * (observed * 10 + 0.05)
            prev_star = {n: p.detach().clone() for n, p in model.named_parameters()}

        # ---- consolidate: compute Fisher + store star params for this task ----
        fisher = compute_fisher(model, X, y, domain)
        star = {n: p.detach().clone() for n, p in model.named_parameters()}
        fisher_list.append(fisher)
        star_list.append(star)

    # ---- final evaluation ----
    early_stable_acc = np.mean([eval_task(model, X, y, "stable") for X, y in stable_tasks[:2]])
    late_volatile_acc = np.mean([eval_task(model, X, y, "volatile") for X, y in volatile_tasks[-2:]])
    all_stable_acc = np.mean([eval_task(model, X, y, "stable") for X, y in stable_tasks])
    all_volatile_acc = np.mean([eval_task(model, X, y, "volatile") for X, y in volatile_tasks])

    return {
        "early_stable_retention": early_stable_acc,
        "late_volatile_adaptation": late_volatile_acc,
        "avg_stable_acc": all_stable_acc,
        "avg_volatile_acc": all_volatile_acc,
        "final_volatility_estimates": dict(volatility),
    }


# ----------------------------------------------------------------------------
# 5. Run both modes and compare
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    N_TASKS = 12
    N_RUNS = 5  # average over multiple seeds for stability

    results = {"baseline": [], "volatility": []}

    for run in range(N_RUNS):
        torch.manual_seed(run)
        random.seed(run)
        np.random.seed(run)
        for mode in ["baseline", "volatility"]:
            torch.manual_seed(run)  # reset model init identically per mode
            res = run_experiment(mode, n_tasks=N_TASKS)
            results[mode].append(res)

    print("=" * 70)
    print(f"Results averaged over {N_RUNS} runs, {N_TASKS} tasks (alternating stable/volatile)")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print("Prediction being tested:")
    print("  Volatility-adjusted EWC should show HIGHER early-stable retention")
    print("  (better protection where it's warranted) AND HIGHER late-volatile")
    print("  adaptation (less wasted protection where domain genuinely drifts),")
    print("  compared to uniform baseline EWC.")
    print("=" * 70)

    b = results["baseline"]
    v = results["volatility"]
    print("\nFinal volatility estimates learned (volatility-mode, last run):")
    for g, val in v[-1]["final_volatility_estimates"].items():
        print(f"  {g:10s}: {val:.3f}")
