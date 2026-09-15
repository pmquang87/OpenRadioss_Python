"""
Forming Limit Diagram failure model (/FAIL/FLD).

Fortran origin: ``starter/source/materials/fail/fld/hm_read_fail_fld.F`` and
``engine/source/materials/fail/fld/fail_fld_c.F``.

Physics:
The Forming Limit Diagram (FLD) criterion predicts onset of localized necking /
fracture in sheet metal shells based on the minor and major in-plane principal strains
(eps_min, eps_maj).

Principal strains:
    S1 = 0.5 * (eps_xx + eps_yy)
    S2 = 0.5 * (eps_xx - eps_yy)
    Q  = sqrt(S2^2 + (0.5 * eps_xy)^2)
    eps_maj = S1 + Q
    eps_min = S1 - Q

Limit major strain EM is evaluated from the FLD curve function f(eps_min):
- If Istrain == 0 (true strain):
    EM = f(eps_min)
- If Istrain == 1 (engineering strain):
    eps_min_eng = exp(eps_min) - 1
    EM_eng = f(eps_min_eng)
    EM = ln(1 + EM_eng)

Damage ratio:
    D = eps_maj / max(EM, 1e-20)
    dama = max(dama, D)

Failure condition:
    Point breaks when dama >= 1.0 (unless Ifail_sh == 4, where damage is monitored
    without element deletion).
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Solid step: FLD is a shell/plane-stress criterion; no-op for solids."""
    return np.zeros(len(dama), dtype=bool)


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Plane-stress FLD damage step for one layer of a shell slice.

    Parameters
    ----------
    fail : FailureModel
        The /FAIL/FLD failure model instance.
    sig : np.ndarray (m, 3)
        Plane-stress tensor [sig_xx, sig_yy, sig_xy].
    d_epsp : np.ndarray (m,) or float
        Plastic strain increment of the cycle.
    deps : np.ndarray (m, 3)
        In-plane strain increment [deps_xx, deps_yy, deps_xy].
    dt : float
        Current time step.
    dama : np.ndarray (m,)
        Persistent damage array (modified in place).
    tstar : np.ndarray (m,) or None
        Homologous temperature (unused by FLD).
    eps_tot : np.ndarray (m, 3) or None
        Total in-plane strain tensor [eps_xx, eps_yy, eps_xy]. If None,
        approximated from deps or plastic strain.
    """
    p = fail.params
    func = p.get("function")
    if func is None:
        return dama >= 1.0

    if eps_tot is not None:
        e = eps_tot
    else:
        # Fallback if total strain is not explicitly supplied
        e = deps if deps.ndim == 2 else deps[None, :]

    eps_xx = e[:, 0]
    eps_yy = e[:, 1]
    eps_xy = e[:, 2]

    e12 = 0.5 * eps_xy
    s1 = 0.5 * (eps_xx + eps_yy)
    s2 = 0.5 * (eps_xx - eps_yy)
    q = np.sqrt(s2 ** 2 + e12 ** 2)

    emaj = s1 + q
    emin = s1 - q

    # Ensure emin <= emaj
    swap = emin > emaj
    if np.any(swap):
        emaj_s = emaj.copy()
        emaj = np.where(swap, emin, emaj)
        emin = np.where(swap, emaj_s, emin)

    istrain = p.get("istrain", 0)
    if istrain == 1:
        # Engineering strain input: convert true minor strain to engineering
        emin_eng = np.exp(np.clip(emin, -100.0, 100.0)) - 1.0
        em_eng = np.interp(emin_eng, func.x, func.y)
        em = np.log(np.maximum(em_eng + 1.0, _TINY))
    else:
        # True strain input
        em = np.interp(emin, func.x, func.y)

    dam = emaj / np.maximum(em, _TINY)
    np.maximum(dama, dam, out=dama)

    ifail_sh = p.get("ifail_sh", getattr(fail, "ifail_sh", 1))
    if ifail_sh == 4:
        # Ifail_sh = 4: calculation only, no element deletion
        return np.zeros(len(dama), dtype=bool)

    return dama >= 1.0
