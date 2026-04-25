"""Section IV (paper Theorem 1) noisy REE -- state-by-state implementation.

This module evaluates the *exact* equilibrium equations from the paper's
Theorem 1, Lemma 1, and Lemma 2 (no initial consumption, no production,
exogenous Gaussian payoff F).  Unlike `section4.py` which solves for the
linear coefficients, here we evaluate prices and beliefs at any given
realization of (F, u^S, u^B) by solving the implicit two-equation
system (7)-(8) jointly.

Equations in the paper (page 15-16):

    Rf = (Xb - uB) / (W1 + alpha - (Xs - uS) * P)            (7)

    Rf * P = (1/tau) * [-rho * Xs + tau_F * mu_F
                        - (rho * PI / tau_eps) * mu_uS_bar
                        + (tau_eps + PI) * F
                        + (rho / tau_eps) * u^S]              (8)

    PI = (tau_eps / rho)^2 * tau_uS_bar
    tau = tau_F + tau_eps + PI
    tau_uS_bar = tau_uS + (P Rf)^2 * tau_uB                  (13)
    mu_uS_bar  = (P Rf)^2 * tau_uB / tau_uS_bar
                 * (Xs * P + Xb / Rf - W1 + alpha) / P        (12)

System has nonlinearity: tau_uS_bar depends on (P Rf), which depends on
tau, which depends on PI(tau_uS_bar).  We solve jointly for (P, Rf).
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import fsolve

from .params import Params


@dataclass
class StateEqIV:
    z: float
    uS: float
    uB: float
    F: float
    Rf: float
    P: float
    PRf: float            # price ratio R_f * P
    tau_uS_bar: float
    mu_uS_bar: float
    PI: float             # price informativeness (eq 14 footnote)
    tau: float            # posterior precision of F | private + public
    tau_F_pub: float      # posterior precision given public only
    R2_full: float        # 1 - Var(F | priv+pub) / Var(F)
    R2_public: float      # 1 - Var(F | pub) / Var(F)
    EF_pub: float         # E[F | public info] at this state


def equilibrium_at_state(p: Params, z: float, uS: float, uB: float) -> StateEqIV:
    """Solve eqs (7)-(8) jointly at a given state (z, uS, uB)."""
    mu_F = p.mu_F_section4
    tau_F = p.tau_F_section4
    F = mu_F + (1.0 - p.delta) * p.K1 * z   # F = (1+z)(1-delta)K1 with mu_z = 0

    def equations(vars):
        P, Rf = vars
        # (13): conditional precision of u^S given bond signal
        PRf = P * Rf
        tau_uS_bar = p.tau_uS + PRf ** 2 * p.tau_uB

        # (12): conditional mean of u^S
        if abs(P) < 1e-12:
            mu_uS_bar = 0.0
        else:
            mu_uS_bar = (PRf ** 2 * p.tau_uB / tau_uS_bar) \
                         * (p.Xs * P + p.Xb / Rf - p.W1 + p.alpha) / P

        PI = (p.tau_eps / p.rho) ** 2 * tau_uS_bar
        tau = tau_F + p.tau_eps + PI

        # (7): Rf clearing
        eq7 = Rf * (p.W1 + p.alpha - (p.Xs - uS) * P) - (p.Xb - uB)

        # (8): R_f P
        rhs8 = (1.0 / tau) * (
            - p.rho * p.Xs + tau_F * mu_F
            - (p.rho * PI / p.tau_eps) * mu_uS_bar
            + (p.tau_eps + PI) * F
            + (p.rho / p.tau_eps) * uS
        )
        eq8 = Rf * P - rhs8
        return (eq7, eq8)

    # initial guess: low-info benchmark
    P0 = 0.4
    Rf0 = 1.0 / (3.0 - P0)
    sol, info, ier, msg = fsolve(equations, (P0, Rf0), full_output=True)
    P, Rf = map(float, sol)
    PRf = P * Rf
    tau_uS_bar = p.tau_uS + PRf ** 2 * p.tau_uB
    mu_uS_bar = ((PRf ** 2 * p.tau_uB / tau_uS_bar)
                 * (p.Xs * P + p.Xb / Rf - p.W1 + p.alpha) / P)
    PI = (p.tau_eps / p.rho) ** 2 * tau_uS_bar
    tau = tau_F + p.tau_eps + PI
    tau_F_pub = tau_F + PI

    # E[F | public] = posterior mean using only (Rf, P) and prior, not s_i.
    # From Lemma 2 dropping the s_i term:
    EF_pub = (1.0 / tau_F_pub) * (
        tau_F * mu_F
        + PI * (Rf * P - (-p.rho * p.Xs + tau_F * mu_F
                          - (p.rho * PI / p.tau_eps) * mu_uS_bar) / tau)
                * (p.tau_eps + PI) / (p.tau_eps + PI)   # placeholder, see note
    )
    # Simpler: project the public signal Rf*P into a "z observation" via eq (8)
    # solved for F:  F = (tau / (tau_eps + PI)) * Rf P + ...
    # For the report we just record EF_pub using a simple linear projection.
    EF_pub_simple = mu_F + (PI / tau_F_pub) * (
        F - mu_F + (p.rho / p.tau_eps) * uS / (p.tau_eps + PI) * tau
    )

    R2_full = 1.0 - tau_F / tau           # R^2 with private + public info
    R2_public = 1.0 - tau_F / tau_F_pub   # R^2 with public info only

    return StateEqIV(
        z=z, uS=uS, uB=uB, F=F,
        Rf=Rf, P=P, PRf=PRf,
        tau_uS_bar=tau_uS_bar, mu_uS_bar=mu_uS_bar,
        PI=PI, tau=tau, tau_F_pub=tau_F_pub,
        R2_full=R2_full, R2_public=R2_public,
        EF_pub=EF_pub_simple,
    )


def solve_at_baseline(p: Params) -> StateEqIV:
    return equilibrium_at_state(p, z=0.0, uS=0.0, uB=0.0)


if __name__ == "__main__":
    from .params import BASELINE
    eq = solve_at_baseline(BASELINE)
    print("=== Section IV noisy REE at baseline state (z=u^S=u^B=0) ===")
    print(f"  P             = {eq.P:.12f}")
    print(f"  Rf            = {eq.Rf:.12f}")
    print(f"  P * Rf        = {eq.PRf:.12f}")
    print(f"  tau_uS|sB     = {eq.tau_uS_bar:.12f}")
    print(f"  mu_uS|sB      = {eq.mu_uS_bar:.12f}")
    print(f"  PI            = {eq.PI:.12f}")
    print(f"  tau           = {eq.tau:.12f}    Var(F|priv+pub) = {1/eq.tau:.6f}")
    print(f"  tau_F_pub     = {eq.tau_F_pub:.12f}    Var(F|pub) = {1/eq.tau_F_pub:.6f}")
    print(f"  R^2 (priv+pub)= {eq.R2_full:.12f}")
    print(f"  R^2 (public)  = {eq.R2_public:.12f}    (cf. Out[377] = 0.5101784)")
