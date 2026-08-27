"""
ipd.evolution
=============

**Evolutionary dynamics — the basis of H5.**

Two population-level dynamics are compared, paired on the same payoff
matrix Pi.

1. Replicator equation (RE)
----------------------------
Standard individual-level selection: type i's frequency x_i grows
when its fitness exceeds the population mean,

    dx_i/dt = x_i * ((Pi x)_i - x' Pi x),

discrete update x_i <- x_i * f_i / f-bar with f = Pi x. Known limit:
whenever T > R > P > S, defectors dominate cooperators — pure
individual-level selection does not evolve cooperation in a PD.

2. Optimal replicator equation (ORE)
-------------------------------------
Bravetti & Padilla (2018), "An optimal strategy to solve the
Prisoner's Dilemma", Sci. Rep. 8:1948. Selection also operates at the
**competing-group level** (slow timescale): the RE is extended to an
optimal-control problem maximising the terminal mean fitness
g(x(tau)) = x(tau)' Pi x(tau), yielding an extended system with a
co-state p:

    forward   dx_a/dt = x_a * (p_a - <p>)      x(0) = x0
    backward  dp_a/dt = <p>*p_a - p_a^2 / 2    p(tau) = grad g(x(tau))
    terminal  p_a(tau) = ((Pi + Pi') x(tau))_a

The co-state carries "the large reward the group will finally share"
**backwards in time**; where p_C > p_D even selfish individuals
cooperate. The boundary-value problem is solved by a
forward-backward sweep (FBSM), generalised naturally to K types so
RE and ORE trajectories pair on the same Pi.

H5 operationalisation: "does HalloReg survive across generations" —
integrate RE and ORE from each initial composition x0 (= counts/30)
and test whether the terminal HalloReg frequency clears the survival
threshold,

    survival(x0) = 1[x_halloreg(tau) > threshold],

with threshold 1/30 (one individual in a 30-agent population).
Comparing the occupancy of the survival basin across types shows who
survives from the wider set of initial conditions.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# ==================================================================== RE
def replicator_trajectory(Pi: np.ndarray, x0: Sequence[float],
                          steps: int = 600) -> np.ndarray:
    """
    Trajectory of the discrete replicator update, shape
    (steps + 1, k).

    Fitness is shifted by the minimum to stay positive (payoffs can
    be non-positive in principle); replicator dynamics are invariant
    to affine payoff transformations, so the shift changes nothing
    qualitative.
    """
    P = np.asarray(Pi, dtype=float)
    shift = max(0.0, -P.min()) + 1e-6
    x = np.clip(np.asarray(x0, dtype=float), 1e-12, None)
    x = x / x.sum()
    traj = np.empty((steps + 1, len(x)))
    traj[0] = x
    for t in range(steps):
        f = P @ x + shift
        fbar = float(x @ f)
        x = x * f / max(fbar, 1e-12)
        x = np.clip(x, 1e-15, None)
        x = x / x.sum()
        traj[t + 1] = x
    return traj


def replicator_ends_batch(Pi: np.ndarray, X0: np.ndarray,
                          steps: int = 600) -> np.ndarray:
    """
    Terminal RE compositions for many initial points, (N, k) — for
    basin analysis. Fully vectorised over the initial-point axis;
    only time remains sequential.
    """
    P = np.asarray(Pi, dtype=float)
    shift = max(0.0, -P.min()) + 1e-6
    X = np.clip(np.asarray(X0, dtype=float), 1e-12, None)
    X = X / X.sum(axis=1, keepdims=True)
    for _ in range(steps):
        F = X @ P.T + shift                       # (N, k)
        fbar = np.sum(X * F, axis=1, keepdims=True)
        X = X * F / np.maximum(fbar, 1e-12)
        X = np.clip(X, 1e-15, None)
        X = X / X.sum(axis=1, keepdims=True)
    return X


# ==================================================================== ORE
def _grad_g(Pi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gradient of the terminal mean fitness g(x) = x'Pi x:
    grad g = (Pi + Pi')x — the terminal co-state."""
    return (Pi + Pi.T) @ x


def ore_trajectory(Pi: np.ndarray, x0: Sequence[float], tau: float = 1.0,
                   steps: int = 400, sweeps: int = 60, relax: float = 0.5,
                   tol: float = 1e-7) -> Dict:
    """
    K-type ORE trajectory via forward-backward sweeps.

    Returns x (steps+1, k), p (steps+1, k), x_end (k,), converged,
    resid
    """
    P = np.asarray(Pi, dtype=float)
    x0 = np.clip(np.asarray(x0, dtype=float), 1e-12, None)
    x0 = x0 / x0.sum()
    k = len(x0)
    dt = tau / steps

    # The co-state equation dp/dt = <p>p - p^2/2 is Riccati-type and
    # can blow up in finite time; clamp at a payoff-scale multiple
    # for numerical stability. Basin verdicts depend only on the sign
    # structure of the terminal composition, so the clamp changes no
    # qualitative conclusion.
    p_bound = 20.0 * max(1.0,
                         float(np.max(np.abs(_grad_g(P, np.ones(k) / k)))),
                         float(np.max(np.abs(P))))

    x = np.tile(x0, (steps + 1, 1))
    p = np.tile(_grad_g(P, x0), (steps + 1, 1))
    prev = None
    resid = np.inf

    for _ in range(sweeps):
        # --- forward: state x ---
        x[0] = x0
        for n in range(steps):
            pm = float(x[n] @ p[n])
            xn = np.clip(x[n] + dt * (x[n] * (p[n] - pm)), 1e-12, None)
            x[n + 1] = xn / xn.sum()
        # --- backward: co-state p (from the terminal condition) ---
        p_new = p.copy()
        p_new[steps] = np.clip(_grad_g(P, x[steps]), -p_bound, p_bound)
        for n in range(steps, 0, -1):
            pm = float(x[n] @ p_new[n])
            dpdt = pm * p_new[n] - 0.5 * p_new[n] ** 2
            p_new[n - 1] = np.clip(p_new[n] - dt * dpdt, -p_bound, p_bound)
        # --- relaxation: damp oscillations ---
        p = (1.0 - relax) * p + relax * p_new
        if prev is not None:
            resid = float(np.max(np.abs(p - prev)))
            if resid < tol:
                break
        prev = p.copy()

    return {"x": x, "p": p, "x_end": x[steps].copy(),
            "converged": bool(resid < tol), "resid": float(resid)}


def ore_ends_batch(Pi: np.ndarray, X0: np.ndarray, tau: float = 1.0,
                   steps: int = 220, sweeps: int = 30,
                   relax: float = 0.5) -> np.ndarray:
    """
    Terminal ORE compositions for many initial points, (N, k) —
    FBSM fully vectorised over initial points; state and co-state
    live as (steps+1, N, k) tensors, only time integrates
    sequentially.
    """
    P = np.asarray(Pi, dtype=float)
    X0 = np.clip(np.asarray(X0, dtype=float), 1e-12, None)
    X0 = X0 / X0.sum(axis=1, keepdims=True)
    N, k = X0.shape
    dt = tau / steps
    G = P + P.T
    p_bound = 20.0 * max(1.0, float(np.max(np.abs(G @ (np.ones(k) / k)))),
                         float(np.max(np.abs(P))))

    x = np.tile(X0, (steps + 1, 1, 1))                       # (S+1, N, k)
    p = np.tile((X0 @ G.T)[None, :, :], (steps + 1, 1, 1))   # (S+1, N, k)

    for _ in range(sweeps):
        x[0] = X0
        for n in range(steps):
            pm = np.sum(x[n] * p[n], axis=1, keepdims=True)
            xn = np.clip(x[n] + dt * (x[n] * (p[n] - pm)), 1e-12, None)
            x[n + 1] = xn / xn.sum(axis=1, keepdims=True)
        p_new = p.copy()
        p_new[steps] = np.clip(x[steps] @ G.T, -p_bound, p_bound)
        for n in range(steps, 0, -1):
            pm = np.sum(x[n] * p_new[n], axis=1, keepdims=True)
            dpdt = pm * p_new[n] - 0.5 * p_new[n] ** 2
            p_new[n - 1] = np.clip(p_new[n] - dt * dpdt, -p_bound, p_bound)
        p = (1.0 - relax) * p + relax * p_new
    return x[steps].copy()


# ==================================================================== Survival metrics
def survival_fraction(ends: np.ndarray, idx: int,
                      threshold: float = 1.0 / 30.0) -> float:
    """
    **Survival-basin occupancy** of type idx over a set of terminal
    compositions; the default threshold 1/30 is one individual in a
    30-agent population.
    """
    return float(np.mean(np.asarray(ends)[:, idx] > threshold))


def mean_terminal_frequency(ends: np.ndarray, idx: int) -> float:
    """Mean terminal frequency of type idx."""
    return float(np.mean(np.asarray(ends)[:, idx]))


def dominance_fraction(ends: np.ndarray, idx: int) -> float:
    """Fraction of initial points whose terminal composition is
    dominated by type idx."""
    E = np.asarray(ends)
    return float(np.mean(np.argmax(E, axis=1) == idx))


def terminal_cc(CCm: np.ndarray, ends: np.ndarray) -> float:
    """
    Mean **behavioural** mutual-cooperation rate of the terminal
    compositions, CC(x) = x' CCm x. Derived directly from observed CC
    behaviour rather than type labels, avoiding arbitrary
    thresholding and label-behaviour dissociation (cooperative-label
    types that actually defect in deadlock regimes).
    """
    E = np.asarray(ends)
    return float(np.mean(np.einsum("ni,ij,nj->n", E, np.asarray(CCm), E)))
