"""Grid-integrated quantities for Section IV.

Out[377] in the Mathematica notebook is

    E0[ R^2(F | public info) ]
        = 1 - E[ Var(F | Rf, P) ] / Var(F)

where the expectation is over (z, u^S, u^B), each independent Gaussian
with the baseline precisions.  The notebook uses a 43 x 3 x 15
Gauss-Hermite tensor grid; we replicate that here for the Section IV
model (no initial consumption, no production).  Section V will
introduce extra dependencies via aggregate consumption and investment;
this grid is the same format and serves as a stepping-stone.
"""

import numpy as np

from .params import Params
from .section4_noisy import equilibrium_at_state


def gauss_hermite_normal(n: int):
    """Gauss-Hermite nodes and weights for a standard normal:
    integrates int f(x) (1/sqrt(2 pi)) exp(-x^2/2) dx."""
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    # transform from physicist's Hermite to probabilist's
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    return nodes, weights


def grid_R2_public(p: Params, n_z: int = 43, n_uS: int = 3, n_uB: int = 15):
    """Compute the expected R^2 of the stock payoff F given public info,
    averaged over the joint distribution of (z, u^S, u^B)."""
    # Standard-normal Gauss-Hermite, then scale by 1/sqrt(precision)
    z_nodes, z_w = gauss_hermite_normal(n_z)
    uS_nodes, uS_w = gauss_hermite_normal(n_uS)
    uB_nodes, uB_w = gauss_hermite_normal(n_uB)

    z_nodes = z_nodes / np.sqrt(p.tau_z)
    uS_nodes = uS_nodes / np.sqrt(p.tau_uS)
    uB_nodes = uB_nodes / np.sqrt(p.tau_uB)

    tau_F = p.tau_F_section4
    var_F = 1.0 / tau_F

    Evar = 0.0
    EF_pub = 0.0
    weight_sum = 0.0

    n_skipped = 0
    for i, z in enumerate(z_nodes):
        for j, uS in enumerate(uS_nodes):
            for k, uB in enumerate(uB_nodes):
                w = z_w[i] * uS_w[j] * uB_w[k]
                try:
                    eq = equilibrium_at_state(p, z=z, uS=uS, uB=uB)
                    var_pub = 1.0 / eq.tau_F_pub
                    Evar += w * var_pub
                    EF_pub += w * eq.EF_pub
                    weight_sum += w
                except Exception:
                    n_skipped += 1

    R2 = 1.0 - Evar / var_F
    return {
        "R2_public_grid": R2,
        "E[Var(F|pub)]": Evar,
        "Var(F)": var_F,
        "weight_sum": weight_sum,
        "skipped": n_skipped,
        "EF_pub_avg": EF_pub,
    }


if __name__ == "__main__":
    from .params import BASELINE
    out = grid_R2_public(BASELINE, n_z=43, n_uS=3, n_uB=15)
    print("=== Section IV grid-integrated public R^2 (43x3x15 Gauss-Hermite) ===")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k:<22s} = {v:.10f}")
        else:
            print(f"  {k:<22s} = {v}")
    print(f"  -- target Out[377] (Section V) = 0.5101784082068117")
