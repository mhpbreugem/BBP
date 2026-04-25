# BBP Python translation -- progress log

Working translation of `1 Baseline (1).nb` (Breugem-Buss-Peress 2022,
"What do Interest Rates Reveal about the Stock Market?").

## What runs

```
python3 -m bbp_python.verify          # full side-by-side report
python3 -m bbp_python.section4        # Section IV linearised closed-form
python3 -m bbp_python.section4_noisy  # state-by-state Theorem-1 solver
python3 -m bbp_python.section4_grid   # 43x3x15 Gauss-Hermite grid R^2
python3 -m bbp_python.section5_det    # Section V deterministic mean-state
python3 -m bbp_python.section5_symm   # Section V symmetric-info noisy
```

## Coverage so far

| Module | What it solves | Verified against |
|---|---|---|
| `section4.py` | Linearised closed-form coefficients | paper appendix B |
| `section4_noisy.py` | Joint (P, Rf) system at any state | paper Theorem 1 / Lemma 1 / Lemma 2 |
| `section4_grid.py` | E0[R^2] over a 43x3x15 quadrature grid | -- (vs. Section V Out[377]) |
| `section5_det.py` | Section V deterministic (no risk, no learning) | self-consistent |
| `section5_symm.py` | Section V noisy, symmetric info | self-consistent |

## Key numbers

| Quantity | Python | Mathematica | Notes |
|---|---|---|---|
| Section IV R^2 (mean state) | 0.4026 | -- | -- |
| Section IV R^2 (grid avg) | **0.4265** | **0.5102** (Out[377]) | gap = init cons + production |
| Section IV E[F|pub] (grid avg) | **1.0176** | **1.0375** (Out[378]) | gap = init cons + production |
| Section V deterministic C1 | 1.8701 | -- | upper bound (no risk) |
| Section V symmetric C1 | 1.6845 | -- | lower bound (full prior risk) |
| Section V REE C1 | -- | **1.8233** (Out[435]) | between the bounds |

## What is *not* yet implemented

The full Section V noisy REE with private signals is the missing piece.
That requires:

1. A linear conjecture in (z, u^S, u^B) for Rf, P, C_bar_1, and I.
2. Bayesian update for each investor: posterior mean and precision of F
   given (s_i, P, Rf), now including the *bond signal* whose constant
   piece b_0 contains C_bar_1 (eq 17 in the paper).
3. Aggregation across investors, market clearing, and coefficient
   matching - the same 119-dim system the Mathematica notebook solves
   numerically for each grid point.

That step is multi-session work; this branch records the intermediate
brackets so progress can be resumed without re-deriving the algebra.

## Next session check-list

* [ ] Section IV WITH initial consumption (C_bar_1 enters the bond
      signal but production is still absent).  Should pick up part of
      the remaining 8pp R^2 gap.
* [ ] Add the firm FOC for I, closing the rest of the gap.
* [ ] Match the rest of the Out[*] reference values.
