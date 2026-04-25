"""Run the Python translation and report alongside the Mathematica
reference numbers extracted from `1 Baseline (1).nb`.

The Mathematica notebook implements the *full Section V* model with
initial consumption + production + nonlinear bond signal, and reports
expectations averaged over a 43 x 3 x 15 Gauss-Hermite grid in
(z, u^S, u^B).  The closed-form Section IV (this file) is only a
partial benchmark -- agreement on Out[*] is therefore not expected at
this stage; the script prints both sets of numbers side-by-side so the
gap is explicit.
"""

from .params import BASELINE
from .section4 import solve_section4, public_info_R2
from .section5_det import solve_section5_deterministic


def main() -> None:
    eq4 = solve_section4(BASELINE)
    eq5 = solve_section5_deterministic(BASELINE)

    print("=" * 72)
    print("BBP (Breugem-Buss-Peress 2022) -- Python translation, status report")
    print("=" * 72)
    print()
    print("Section IV (simplified, exogenous Gaussian payoff) closed-form:")
    print(f"  Rf            = {eq4.Rf:.12f}")
    print(f"  E[P]          = {eq4.P_const + eq4.P_F * BASELINE.mu_F_section4:.12f}")
    print(f"  a0, a1, a2    = {eq4.a0:.6f}, {eq4.a1:.6f}, {eq4.a2:.6f}")
    print(f"  b0, b1, b2    = {eq4.b0:.6f}, {eq4.b1:.6f}, {eq4.b2:.6f}")
    print(f"  R^2 (public)  = {public_info_R2(BASELINE, eq4):.12f}")
    print()
    print("Section V (full model, NS=Npi=1 deterministic mean-state limit):")
    print(f"  Rf            = {eq5.Rf:.12f}    PY = {eq5.PY:.12f}")
    print(f"  PX            = {eq5.PX:.12f}")
    print(f"  I             = {eq5.I:.12f}")
    print(f"  X, Y          = {eq5.X:.6f}, {eq5.Y:.6f}")
    print(f"  C1, C2        = {eq5.C1:.12f}, {eq5.C2:.12f}")
    print(f"  FOC residual  = {eq5.residual:.3e}")

    print()
    print("Mathematica reference values (1 Baseline (1).nb -- full Section V):")
    refs = [
        ("Out[354]  Solving error",          "4.551977556537516e-15"),
        ("Out[377]  E0R2 (public R^2)",      "0.5101784082068117"),
        ("Out[378]  E0[E[Pi|Omega^U]]",      "1.037531029785874"),
        ("Out[385]  Private-info quantity",  "6.878616163917778"),
        ("Out[397]  Production qty 1",       "0.0074031982195399415"),
        ("Out[398]  Production qty 2",       "0.008162852807050758"),
        ("Out[400]  Production qty 4",       "0.04583474041260338"),
        ("Out[435]  Consumption qty 1",      "1.823268337170934"),
        ("Out[444]  Consumption qty 2",      "0.2681904635570172"),
        ("Out[470]  ER vs Rf slope",         "-0.10865917344723486"),
    ]
    for name, val in refs:
        print(f"  {name:<32s} = {val}")

    print()
    print("Notes:")
    print("  * Section IV here drops initial consumption AND production,")
    print("    so the R^2 above does not target Out[377] directly.")
    print("  * Reproducing Out[377] requires the Section V solver (next step).")


if __name__ == "__main__":
    main()
