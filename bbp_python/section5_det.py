"""Section V (full model) deterministic single-state limit.

This module solves the BBP Section V equilibrium at the *unconditional
mean state* (z = u^S = u^B = 0) and with the integrations collapsed to
a single quadrature node (NS = NPI = 1).  In this limit:

  * z = 0, so productivity is deterministic and equal to 1.
  * eps_i = 0, so the private signal carries no extra information.
  * The conjectured prices, consumption, and investment are constants.
  * No learning takes place; the bond signal carries no information.

The point of this module is to validate the model structure end-to-end
and produce a baseline number that we can later cross-check against the
full grid-integrated solver.  It does *not* yet target Out[377] (which
requires Gauss-Hermite quadrature) but matches Mathematica's symbolic
setup line-for-line.

Equations at the single-node deterministic limit
------------------------------------------------
Define R_f = 1/PY (gross interest rate) and let PX = stock price.

  D1 = 1 * (K_1 - I)                                 (period-1 dividend)
  D2 = (K_1 (1 - delta) + I) - kappa I^2 / (2 K_1)    (period-2 dividend at z = 0)

  W0 = e0 = W1     (since X_0 = Y_0 = 0 and the e_0 term equals W̄_1)

  FOCX (single-node):  D2 = PX * R_f
  FOCC1:               exp(-rho * C1) = beta * R_f * exp(-rho * C2)
  ADX:                 XSg = X
  ADY:                 YSg * PY = W0 - PX * XSg - C1
  MCX:                 X = Xs - theta_X = 1
  MCY:                 YSg = Xb - theta_Y - alpha / PY = 1
  FOCI:                I = (K_1 / kappa) * E[z | info] = 0    (mean state)

Substituting MCX, MCY, FOCI, ADX, ADY into the remaining FOCX and FOCC1:

  D2 = PX * R_f                                     (1)
  exp(-rho * C1) = beta * R_f * exp(-rho * C2)       (2)
  C2 = (W0 - PX * X - C1) / PY + X * D2
     = (W0 - PX - C1) / PY + D2                      (X = 1)

With FOCX (D2 = PX * R_f = PX / PY), C2 simplifies to (W0 - C1) / PY.

The last equation FOCC1 then determines (PY, C1) given that PX = PY * D2
and that ADY pins YSg.  We close the system numerically.
"""

from dataclasses import dataclass
import math
from scipy.optimize import brentq

from .params import Params


@dataclass
class EqV_Det:
    Rf: float
    PY: float       # bond price 1/Rf
    PX: float       # stock price
    I: float
    X: float        # investor stock holding
    Y: float        # investor bond holding
    C1: float       # investor initial consumption
    C2: float       # investor terminal consumption
    XSg: float
    YSg: float
    D1: float
    D2: float
    W0: float
    residual: float


def solve_section5_deterministic(p: Params) -> EqV_Det:
    """Solve the Section V model at the deterministic single-node limit."""
    # 1. Investment is zero at the mean state (E[z|info] = 0).
    I = 0.0

    # 2. Cashflows reduce as follows:
    D1 = 1.0 * (p.K1 - I)
    D2 = (p.K1 * (1.0 - p.delta) + I) - p.kappa * I ** 2 / (2.0 * p.K1)

    # 3. Initial wealth -- all-bond endowment of size W1.
    W0 = p.W1

    # 4. Market clearing pins X = 1, YSg = 1.
    X = p.Xs            # = 1 with theta_X = 0
    YSg = p.Xb          # alpha = theta_Y = 0

    # 5. Reduce FOCC1 to a one-dimensional equation in PY.
    #    With FOCX -> PX = D2 * PY and X = 1:
    #       C2 = (W0 - PX) / PY + D2 - C1 / PY
    #          = (W0 - D2 * PY - C1) / PY + D2
    #          = (W0 - C1) / PY                     (the D2 terms cancel)
    #    FOCC1: exp(-rho C1) = beta * (1/PY) * exp(-rho * (W0 - C1) / PY)
    #
    #    ADY: YSg * PY = W0 - PX * XSg - C1
    #         1 * PY  = W0 - PX - C1     (XSg = X = 1)
    #         PY      = W0 - D2 * PY - C1
    #         C1      = W0 - PY * (1 + D2)
    #    Substituting C1 into FOCC1 yields a single equation in PY.

    rho, beta = p.rho, p.beta

    def residual(PY: float) -> float:
        C1 = W0 - PY * (1.0 + D2)
        C2 = (W0 - C1) / PY
        # FOCC1: -rho C1 = ln(beta) - ln(PY) - rho C2
        return -rho * C1 - (math.log(beta) - math.log(PY) - rho * C2)

    # Bracket: PY in (small, W0/(1+D2)) so that C1 > 0
    upper = W0 / (1.0 + D2) - 1e-9
    PY = brentq(residual, 1e-4, upper, xtol=1e-14)

    Rf = 1.0 / PY
    PX = D2 * PY
    C1 = W0 - PY * (1.0 + D2)
    C2 = (W0 - C1) / PY
    Y = (W0 - PX * X - C1) / PY
    XSg = X
    res = residual(PY)

    return EqV_Det(
        Rf=Rf, PY=PY, PX=PX, I=I,
        X=X, Y=Y, C1=C1, C2=C2,
        XSg=XSg, YSg=YSg,
        D1=D1, D2=D2, W0=W0,
        residual=res,
    )


if __name__ == "__main__":
    from .params import BASELINE
    eq = solve_section5_deterministic(BASELINE)
    print(f"=== Section V (deterministic single-state limit) ===")
    print(f"Rf            = {eq.Rf:.12f}    (PY = {eq.PY:.12f})")
    print(f"PX            = {eq.PX:.12f}")
    print(f"I             = {eq.I:.12f}")
    print(f"X (stock hold)= {eq.X:.12f}")
    print(f"Y (bond hold )= {eq.Y:.12f}")
    print(f"C1            = {eq.C1:.12f}")
    print(f"C2            = {eq.C2:.12f}")
    print(f"D1, D2        = {eq.D1:.12f}, {eq.D2:.12f}")
    print(f"residual FOCC1= {eq.residual:.3e}")
