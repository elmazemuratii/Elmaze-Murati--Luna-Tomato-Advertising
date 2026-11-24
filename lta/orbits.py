"""CR3BP propagation (scipy, no heyoka needed) used only for the 3D visualisation."""
import numpy as np
from scipy.integrate import solve_ivp


def cr3bp_rhs(mu):
    def f(_, s):
        x, y, z, vx, vy, vz = s
        r1 = ((x + mu) ** 2 + y ** 2 + z ** 2) ** 1.5
        r2 = ((x - 1 + mu) ** 2 + y ** 2 + z ** 2) ** 1.5
        ax = 2 * vy + x - (1 - mu) * (x + mu) / r1 - mu * (x - 1 + mu) / r2
        ay = -2 * vx + y - (1 - mu) * y / r1 - mu * y / r2
        az = -(1 - mu) * z / r1 - mu * z / r2
        return [vx, vy, vz, ax, ay, az]
    return f


def propagate(prob, fam, i, n=64):
    """positions (n,3) of one full orbit starting at the occultation state (t=0)."""
    row = prob.db[fam][i]
    T = 1.0 / prob.f1[fam][i]                 # orbital period (nd) == row[6]
    sol = solve_ivp(cr3bp_rhs(prob.mu), (0, T), row[:6], method="DOP853",
                    t_eval=np.linspace(0, T, n, endpoint=False), rtol=1e-11, atol=1e-13)
    return sol.y[:3].T
