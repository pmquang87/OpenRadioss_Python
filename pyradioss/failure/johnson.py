"""
Johnson–Cook failure criterion (/FAIL/JOHNSON).

Fortran origin: ``engine/source/materials/fail/johnson_cook/fail_johnson_s.F``
(solids) and ``fail_johnson_c.F`` (shells); reader
``starter/source/materials/fail/johnson_cook/hm_read_fail_johnson.F``.

Theory (Johnson & Cook, Eng. Fracture Mech. 21 (1985) 31-48)
------------------------------------------------------------
Ductile fracture by accumulation of plastic strain, with a failure
strain that depends on the **stress triaxiality**
sigma* = sigma_m / sigma_vm (mean over von Mises stress — the classic
observation that hydrostatic tension accelerates void growth) and on the
strain rate:

    eps_f = ( D1 + D2 * exp(D3 * sigma*) ) * ( 1 + D4 * ln(rate/rate_0) )

    D += d_eps_p / eps_f         (linear damage accumulation)

The point breaks at D >= 1. The thermal term (1 + D5*T*) of the full
model is not ported (no thermal solution in this port — same deliberate
cut as the LAW2 temperature term).

Conventions matching the original:
* sigma* uses the CURRENT stress state, tension positive. Note D3 is
  usually NEGATIVE for metals in this convention (failure strain drops
  with triaxiality); some references flip the sign of both.
* the rate factor is clamped at 1 below the reference rate (no rate
  term unless D4 > 0 and the rate exceeds eps_dot_0);
* no damage grows without plastic flow (d_eps_p = 0 -> D frozen), so
  the criterion is inert on elastic materials — the Starter warns.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def _rate_factor(fail, deps, dt, dev_from_6):
    """1 + D4*ln(rate/rate0), clamped at 1 below the reference rate."""
    D4 = fail.params["D4"]
    if D4 == 0.0:
        return 1.0
    if dev_from_6:
        tr3 = (deps[:, 0] + deps[:, 1] + deps[:, 2]) / 3.0
        ee = (deps[:, 0] - tr3) ** 2 + (deps[:, 1] - tr3) ** 2 \
            + (deps[:, 2] - tr3) ** 2 \
            + 0.5 * (deps[:, 3] ** 2 + deps[:, 4] ** 2 + deps[:, 5] ** 2)
    else:  # plane stress: thickness strain from incompressibility
        dzz = -(deps[:, 0] + deps[:, 1]) * 0.5
        tr3 = (deps[:, 0] + deps[:, 1] + dzz) / 3.0
        ee = (deps[:, 0] - tr3) ** 2 + (deps[:, 1] - tr3) ** 2 \
            + (dzz - tr3) ** 2 + 0.5 * deps[:, 2] ** 2
    rate = np.sqrt((2.0 / 3.0) * ee) / max(dt, _TINY)
    r = np.maximum(rate / fail.params["eps_dot_0"], 1.0)
    return 1.0 + D4 * np.log(r)


def solid_step(fail, sig, d_epsp, deps, dt, dama):
    """3-D damage step. sig (m, 6) Voigt; returns the broken mask."""
    p = fail.params
    sm = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    s0, s1, s2 = sig[:, 0] - sm, sig[:, 1] - sm, sig[:, 2] - sm
    vm = np.sqrt(1.5 * (s0 ** 2 + s1 ** 2 + s2 ** 2)
                 + 3.0 * (sig[:, 3] ** 2 + sig[:, 4] ** 2 + sig[:, 5] ** 2))
    triax = sm / np.maximum(vm, _TINY)
    eps_f = (p["D1"] + p["D2"] * np.exp(p["D3"] * triax)) \
        * _rate_factor(fail, deps, dt, True)
    dama += np.maximum(d_epsp, 0.0) / np.maximum(eps_f, _TINY)
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama):
    """Plane-stress damage step for one layer. sig (m, 3) = [xx, yy, xy]."""
    p = fail.params
    sm = (sig[:, 0] + sig[:, 1]) / 3.0            # sigma_zz = 0
    vm = np.sqrt(sig[:, 0] ** 2 - sig[:, 0] * sig[:, 1] + sig[:, 1] ** 2
                 + 3.0 * sig[:, 2] ** 2)
    triax = sm / np.maximum(vm, _TINY)
    eps_f = (p["D1"] + p["D2"] * np.exp(p["D3"] * triax)) \
        * _rate_factor(fail, deps, dt, False)
    dama += np.maximum(d_epsp, 0.0) / np.maximum(eps_f, _TINY)
    return dama >= 1.0
