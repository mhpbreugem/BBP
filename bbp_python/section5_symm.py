"""Section V model in the *symmetric-information* limit.

All investors share the same information set (no private signals,
no learning from prices), but the firm's payoff F is genuinely random
because productivity z ~ N(0, 1/tau_z) is uncertain.  This adds a real
risk premium to the stock price relative to the deterministic limit
of `section5_det`.

CARA + Gaussian payoff yields a closed-form equilibrium:

  F | (info) ~ N(mu_F, sigma_F^2),  with mu_F = (K_1 + I), sigma_F^2 = (K_1+I)^2 / tau_z
  C2 = X * F + Y    ~  N(X mu_F + Y, X^2 sigma_F^2)

Investor FOC for X^S (CARA-Gaussian portfolio choice):
  X^S = (mu_F - PX * Rf) / (rho * sigma_F^2)        (eq A9 generalised)

Investor FOC for C_1 (CARA Euler equation):
  exp(-rho C_1) = beta * Rf * E[exp(-rho C_2)]
                = beta * Rf * exp(-rho E[C_2] + (rho^2 / 2) Var(C_2))

Firm FOC: I = (K_1 / kappa) * E[z | info] = 0  at the mean state.

Market clearing pins X = Xs = 1 (and Y = Xb = 1 with alpha = 0).
The remaining unknowns (Rf, PX, C_1) are determined by the three
conditions above plus the initial-period budget constraint.
"""

from dataclasses import dataclass
import math
from scipy.optimize import fsolve

from .params import Params


@dataclass
class EqV_Symm:
    Rf: float
    PY: float
    PX: float
    I: float
    X: float
    Y: float
    C1: float
    EC2: float
    VarC2: float
    mu_F: float
    var_F: float
    risk_premium: float    # rho * Var(F)
    residual: tuple


def solve_section5_symmetric(p: Params) -> EqV_Symm:
    """Solve symmetric-information version of Section V at the mean state."""
    # Mean state: E[z|info] = 0, so optimal investment is zero.
    I = 0.0

    # Stock payoff distribution (mean state).
    capital = p.K1 * (1.0 - p.delta) + I
    mu_F = (1.0 + p.mu_z) * capital - p.kappa * I ** 2 / (2.0 * p.K1)
    var_F = capital ** 2 / p.tau_z          # Var = (K1+I)^2 Var(z)

    # Walrasian quantities at u^S = u^B = 0 with alpha = 0:
    X = p.Xs           # = 1
    Y = p.Xb           # = 1

    W0 = p.W1
    rho, beta = p.rho, p.beta

    risk_prem = rho * var_F * X        # the appropriate risk premium since X = 1

    def equations(vars):
        Rf, PX, C1 = vars
        PY = 1.0 / Rf

        # 1. Stock-demand FOC  X = (mu_F - PX Rf) / (rho * var_F)
        eq_X = X * rho * var_F - (mu_F - PX * Rf)

        # 2. Initial-budget constraint
        #    C1 + PX X + PY Y = W0   (with X = Y = 1 and alpha = 0)
        eq_budget = C1 + PX * X + PY * Y - W0

        # 3. Euler equation
        #    exp(-rho C1) = beta Rf exp(-rho EC2 + rho^2/2 VarC2)
        EC2 = X * mu_F + Y
        VarC2 = X ** 2 * var_F
        lhs = -rho * C1
        rhs = math.log(beta * Rf) - rho * EC2 + 0.5 * rho ** 2 * VarC2
        eq_euler = lhs - rhs

        return (eq_X, eq_budget, eq_euler)

    Rf0, PX0, C10 = 1.0, 0.5, 1.5
    sol, info, ier, msg = fsolve(equations, (Rf0, PX0, C10), full_output=True)
    Rf, PX, C1 = map(float, sol)
    PY = 1.0 / Rf
    EC2 = X * mu_F + Y
    VarC2 = X ** 2 * var_F
    return EqV_Symm(
        Rf=Rf, PY=PY, PX=PX, I=I,
        X=X, Y=Y, C1=C1,
        EC2=EC2, VarC2=VarC2,
        mu_F=mu_F, var_F=var_F,
        risk_premium=risk_prem,
        residual=tuple(equations((Rf, PX, C1))),
    )


if __name__ == "__main__":
    from .params import BASELINE
    eq = solve_section5_symmetric(BASELINE)
    print(f"=== Section V (symmetric-info noisy, mean state) ===")
    print(f"Rf            = {eq.Rf:.12f}    PY = {eq.PY:.12f}")
    print(f"PX            = {eq.PX:.12f}")
    print(f"I             = {eq.I:.12f}")
    print(f"X, Y          = {eq.X:.6f}, {eq.Y:.6f}")
    print(f"C1            = {eq.C1:.12f}")
    print(f"E[C2]         = {eq.EC2:.12f}")
    print(f"Var(C2)       = {eq.VarC2:.12f}")
    print(f"mu_F, var_F   = {eq.mu_F:.6f}, {eq.var_F:.6f}")
    print(f"risk premium  = {eq.risk_premium:.6f}   (rho * Var(F) at X=1)")
    print(f"residual      = {max(abs(r) for r in eq.residual):.3e}")
