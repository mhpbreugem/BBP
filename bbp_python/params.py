"""Baseline parameter values from Breugem, Buss, Peress (2022),
'What do Interest Rates Reveal about the Stock Market?'
Footnote text on p. 22 (Section IV graphs):

    beta = 0.95, rho = 4, tau_z = 2.5^2, tau_eps = 1^2,
    Xs = 1, tau_uS = 5^2, Xb = 1, tau_uB = 10^2,
    alpha = 0, W1 = 3, K1 = 1, kappa = 5, delta = 0
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    beta: float = 0.95
    rho: float = 4.0

    tau_z: float = 2.5 ** 2
    tau_eps: float = 1.0 ** 2
    tau_uS: float = 5.0 ** 2
    tau_uB: float = 10.0 ** 2

    Xs: float = 1.0
    Xb: float = 1.0
    alpha: float = 0.0

    W1: float = 3.0
    K1: float = 1.0
    kappa: float = 5.0
    delta: float = 0.0

    mu_z: float = 0.0

    @property
    def mu_F_section4(self) -> float:
        """For the Section IV simplified illustration the firm output is
        exogenous Gaussian.  Mean is implied by the production setup at
        I = 0:  F = (1 + z)(1 - delta) K1 = K1 (with mu_z = 0, delta = 0)."""
        return (1.0 + self.mu_z) * (1.0 - self.delta) * self.K1

    @property
    def tau_F_section4(self) -> float:
        """Var(F) = K1^2 (1 - delta)^2 / tau_z, hence tau_F = tau_z / (K1 (1-delta))^2."""
        return self.tau_z / ((1.0 - self.delta) * self.K1) ** 2


BASELINE = Params()
