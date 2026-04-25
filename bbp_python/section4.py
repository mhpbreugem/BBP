"""Section IV (simplified illustration): exogenous Gaussian payoff F,
no initial consumption, no production, CARA investors, bond and stock
markets clear, alpha-piece in noise-trader bond demand.

Closed-form derivation follows Appendix B of Breugem-Buss-Peress (2022),
equations (A2)-(A13).

Conjectures
-----------
    bond market clearing :  0       = b0 + b1 * uS + b2 * uB                  (A2)
    stock market clearing:  Rf P    = a0 + a1 * F  + a2 * uS                   (A2)

Coefficients (A10, A12, A13)
----------------------------
    b1 = P Rf
    b2 = 1
    b0 = Rf * (W1 + alpha) - Xs * P * Rf - Xb           (rewriting (A10))

    tau_uS_bar = tau_uS + (b2 / b1)**2 * tau_uB        (A3)
    mu_uS_bar  = (1 / tau_uS_bar) * (b1**2 / b2**2) * tau_uB * (- b0 / b1)
               = (tau_uB / tau_uS_bar) * (- b0 / b1)
               (= 0 in equilibrium because b0 = 0; see below)

    tau = tau_F + tau_eps + (a1 / a2)**2 * tau_uS_bar    (A4)

    a0 = (tau_F / tau) * mu_F + (tau_eps * tau_uS_bar) / (rho * tau) * mu_uS_bar
         - (rho / tau) * Xs                                                     (A12)
    a1 = (tau_eps + PI) / tau                                                   (A12)
    a2 = -(tau_eps + PI) * rho / (tau * tau_eps)                                (A13)

    where PI = price-informativeness = (tau_eps / rho**2) * tau_uS_bar
    (substituting tau_uS_bar for tau_uS|sB in the closed form).
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import fsolve

from .params import Params


@dataclass
class EqIV:
    Rf: float
    P_const: float          # constant part of P (= a0 / Rf)
    P_F: float              # coefficient of F in P (= a1 / Rf)
    P_uS: float             # coefficient of uS in P (= a2 / Rf)

    a0: float
    a1: float
    a2: float
    b0: float
    b1: float
    b2: float

    tau_uS_bar: float
    mu_uS_bar: float
    tau: float
    PI: float               # price informativeness (a1 / a2)^2 * tau_uS_bar


def solve_section4(p: Params) -> EqIV:
    """Solve the Section IV simplified equilibrium.

    Two unknowns close the system: (Rf, P_bar) where P_bar is the
    *expected* (constant part) stock price.  Given them we have b1 = Rf
    P_bar, b2 = 1, b0 follows, the conditional moments of u^S | s_B
    follow, then a0, a1, a2 follow.  The two market-clearing identities
    we have to enforce are:

        (1) b0 = Rf * (W1 + alpha) - Xs * P_bar * Rf - Xb        (already used)
        (2) the constant terms in the conjectured stock price must match,
            i.e.  Rf * P_bar = a0   (eq A2 evaluated at F = mu_F, u^S = 0).

    We close (1) by *imposing* b0 = 0 (which holds in equilibrium iff
    the deterministic part of the bond signal vanishes at u^S = u^B = 0
    -- equivalent to picking Rf so that aggregate desired bond holdings
    equal Xb at the mean state).  That gives Rf as a function of P_bar:

        Rf = Xb / (W1 + alpha - Xs * P_bar).

    The remaining equation (2) determines P_bar.
    """
    mu_F = p.mu_F_section4
    tau_F = p.tau_F_section4

    def residual(Pbar: float) -> float:
        # Rf from b0 = 0 equation (eq A10 with constant terms set to zero).
        Rf = p.Xb / (p.W1 + p.alpha - p.Xs * Pbar)

        b1 = Pbar * Rf
        b2 = 1.0
        b0 = 0.0  # by construction

        tau_uS_bar = p.tau_uS + (b2 / b1) ** 2 * p.tau_uB
        mu_uS_bar = (p.tau_uB / tau_uS_bar) * (b2 / b1) * (- b0 / b1) * b1
        # simplifies to 0 when b0 = 0

        # Solve for tau, a1, a2 given the consistency PI = (a1/a2)^2 tau_uS_bar
        # but a1, a2 themselves depend on tau.  Closed form:
        #   tau = tau_F + tau_eps + (rho^2 / tau_eps) * tau_uS_bar^(-1) * tau_uS_bar^2 / 1
        # Use the algebra: from (a1/a2) = - tau_eps / rho (eqs A12, A13) we get
        #   PI = (tau_eps / rho)^2 * tau_uS_bar / ?   -- let us derive cleanly below.
        # From A12, A13:  a2 = - rho / tau_eps * a1, so (a1/a2)^2 = (tau_eps/rho)^2.
        # Therefore the price-informativeness term in (A4) is
        #   (a1/a2)^2 * tau_uS_bar = (tau_eps^2 / rho^2) * tau_uS_bar.
        PI = (p.tau_eps ** 2 / p.rho ** 2) * tau_uS_bar
        tau = tau_F + p.tau_eps + PI

        a1 = (p.tau_eps + PI) / tau
        a2 = - (p.tau_eps + PI) * p.rho / (tau * p.tau_eps)
        a0 = ((tau_F / tau) * mu_F
              + (p.tau_eps * tau_uS_bar) / (p.rho * tau) * mu_uS_bar
              - (p.rho / tau) * p.Xs)

        return Rf * Pbar - (a0 + a1 * mu_F)  # at uS = 0 by ergodicity

    Pbar0 = 0.5
    Pbar = float(fsolve(residual, Pbar0, full_output=False)[0])

    # Re-evaluate at the solution to populate all fields
    Rf = p.Xb / (p.W1 + p.alpha - p.Xs * Pbar)
    b1 = Pbar * Rf
    b2 = 1.0
    b0 = 0.0
    tau_uS_bar = p.tau_uS + (b2 / b1) ** 2 * p.tau_uB
    mu_uS_bar = 0.0
    PI = (p.tau_eps ** 2 / p.rho ** 2) * tau_uS_bar
    tau = tau_F + p.tau_eps + PI
    a1 = (p.tau_eps + PI) / tau
    a2 = - (p.tau_eps + PI) * p.rho / (tau * p.tau_eps)
    a0 = ((tau_F / tau) * mu_F
          + (p.tau_eps * tau_uS_bar) / (p.rho * tau) * mu_uS_bar
          - (p.rho / tau) * p.Xs)

    return EqIV(
        Rf=Rf,
        P_const=a0 / Rf,
        P_F=a1 / Rf,
        P_uS=a2 / Rf,
        a0=a0, a1=a1, a2=a2,
        b0=b0, b1=b1, b2=b2,
        tau_uS_bar=tau_uS_bar,
        mu_uS_bar=mu_uS_bar,
        tau=tau,
        PI=PI,
    )


def public_info_R2(p: Params, eq: EqIV) -> float:
    """Average R^2 of the stock payoff F given public info {Rf, P}.

    In Section IV simplified, the posterior precision conditional on
    public info only (no private signal) is:

        tau_F_pub = tau_F + (a1/a2)^2 * tau_uS_bar = tau_F + PI

    R^2 = 1 - Var(F | pub) / Var(F) = 1 - tau_F / tau_F_pub.
    """
    tau_F = p.tau_F_section4
    tau_F_pub = tau_F + eq.PI
    return 1.0 - tau_F / tau_F_pub


def expected_posterior_mean(p: Params, eq: EqIV) -> float:
    """E_0[ E[F | Rf, P] ] = mu_F (by tower property)."""
    return p.mu_F_section4


if __name__ == "__main__":
    from .params import BASELINE
    eq = solve_section4(BASELINE)
    print(f"=== Section IV (simplified) closed-form equilibrium ===")
    print(f"Rf            = {eq.Rf:.10f}")
    print(f"P (constant)  = {eq.P_const:.10f}")
    print(f"P coef on F   = {eq.P_F:.10f}")
    print(f"P coef on uS  = {eq.P_uS:.10f}")
    print(f"a0, a1, a2    = {eq.a0:.10f}, {eq.a1:.10f}, {eq.a2:.10f}")
    print(f"b0, b1, b2    = {eq.b0:.10f}, {eq.b1:.10f}, {eq.b2:.10f}")
    print(f"tau           = {eq.tau:.10f}")
    print(f"tau_uS_bar    = {eq.tau_uS_bar:.10f}")
    print(f"PI            = {eq.PI:.10f}")
    print(f"R2_public     = {public_info_R2(BASELINE, eq):.10f}")
