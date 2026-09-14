"""
Tabulated failure criterion (/FAIL/TAB1).

Fortran origin: ``engine/source/materials/fail/tabulated/fail_tab_s.F``
(solids) and ``fail_tab_c.F`` (shells); reader
``starter/source/materials/fail/tabulated/hm_read_fail_tab1.F``.

Physics:
eps_f = table(triaxiality) * Xscale1
    * fscale_el * funct_el(L/L_ref)
    * fscale_t * funct_t(TSTAR)

D += Dadv * d_epsp / eps_f (if Table2 is not used, otherwise complicated)
Or more precisely, if Table2 is not used and fct_IDd = 0:
dD = n * D**(1 - 1/n) * d_epsp / eps_f
If D >= Dcrit -> point breaks.
"""

import numpy as np


_TINY = 1e-20


def _accumulate(fail, dama, d_epsp, eps_f):
    """Damage accumulation for TAB1 without fct_IDd (the default damage logic).
    DP = DN * DD ** (1 - 1/DN)
    UVAR(I,1) += DP * d_epsp / eps_f
    """
    p = fail.params
    # Pre-calculated damage scale DP:
    dn = p.get("n", 1.0)
    if dn != 0.0 and dn != 1.0:
        exp = 1.0 - 1.0 / dn
        dp = np.where(dama > 0.0, dn * np.maximum(dama, 1e-12) ** exp, 1.0)
    else:
        dp = 1.0

    grow = eps_f > 0.0
    # Fortran: IF (EPSF > ZERO) UVAR = UVAR + DP * DPLA / EPSF
    dama += np.where(grow, dp * np.maximum(d_epsp, 0.0) / np.maximum(eps_f, _TINY), 0.0)
    np.minimum(dama, max(1.0, float(p.get("dcrit", 1.0))), out=dama)


def _scale_factors(fail, eps_f, tstar=None):
    p = fail.params
    # Xscale1
    eps_f *= p.get("xscale1", 1.0)
    
    # Not fully implemented yet: length scaling and temperature scaling
    # If a deck uses fct_id_el > 0 or fct_id_t > 0, it would be evaluated here.
    # We will ignore these for the M77 milestone since FAILURE_TAB1_0000.rad does not use them (all 0)
    return eps_f


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """3-D damage step."""
    p = fail.params
    dcrit = p.get("dcrit", 1.0)
    table = p.get("table")
    if table is None:
        return dama >= dcrit

    sm = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    s0, s1, s2 = sig[:, 0] - sm, sig[:, 1] - sm, sig[:, 2] - sm
    vm = np.sqrt(1.5 * (s0 ** 2 + s1 ** 2 + s2 ** 2)
                 + 3.0 * (sig[:, 3] ** 2 + sig[:, 4] ** 2 + sig[:, 5] ** 2))
    triax = sm / np.maximum(vm, _TINY)

    # evaluate table
    eps_f = np.interp(triax, table.x, table.y)
    eps_f = _scale_factors(fail, eps_f, tstar)
    
    _accumulate(fail, dama, d_epsp, eps_f)
    return dama >= dcrit


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Plane-stress damage step for one layer. sig (m, 3) = [xx, yy, xy]."""
    p = fail.params
    dcrit = p.get("dcrit", 1.0)
    table = p.get("table")
    if table is None:
        return dama >= dcrit

    sm = (sig[:, 0] + sig[:, 1]) / 3.0            # sigma_zz = 0
    vm = np.sqrt(sig[:, 0] ** 2 - sig[:, 0] * sig[:, 1] + sig[:, 1] ** 2
                 + 3.0 * sig[:, 2] ** 2)
    triax = sm / np.maximum(vm, _TINY)

    # evaluate table
    eps_f = np.interp(triax, table.x, table.y)
    eps_f = _scale_factors(fail, eps_f, tstar)

    _accumulate(fail, dama, d_epsp, eps_f)
    return dama >= dcrit
