"""
Bi-quadratic failure criterion (/FAIL/BIQUAD).

Fortran origin: ``engine/source/materials/fail/biquad/fail_biquad_s.F``
(solids) / ``fail_biquad_c.F`` (shells); reader
``starter/source/materials/fail/biquad/hm_read_fail_biquad.F``.

Theory
------
Like Johnson–Cook failure, plastic strain accumulates damage weighted by
a triaxiality-dependent failure strain,

    D += d_eps_p / eps_f(sigma*),      break at D >= 1,

but eps_f(sigma*) is given by TWO parabolas fitted through five
calibration points at the canonical triaxialities of standard coupon
tests:

    sigma* :  -1/3        0        1/3        2/3         1
    eps_f  :   c1        c2        c3         c4         c5
            (compression) (shear) (uniaxial) (plane strain) (biaxial)

    parabola 1 through (-1/3, c1), (0, c2), (1/3, c3)   used sigma* <= 1/3
    parabola 2 through (1/3, c3), (2/3, c4), (1, c5)    used sigma* >  1/3

The two meet at the uniaxial point c3, so eps_f is continuous. Outside
[-1/3, 1] each parabola extrapolates (Radioss does the same); a small
positive floor guards against a parabola dipping through zero far outside
its fitted range.

Port simplifications: the built-in material presets (M-flag = 1..99,
which derive c1..c5 from c3 alone) and the S-flag / Inst_start necking
options are not ported — give the five coefficients explicitly.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-20
_FLOOR = 1e-6      # eps_f floor way outside the fitted range


def _parabola(x0, y0, x1, y1, x2, y2):
    """Coefficients (a, b, c) of y = a x^2 + b x + c through 3 points."""
    # Lagrange form condensed for the equally-spaced points used here
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) \
        / denom
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1
         + x0 * x1 * (x0 - x1) * y2) / denom
    return a, b, c


def fit(params: dict) -> None:
    """Pre-compute the parabola coefficients from c1..c5 (Starter side)."""
    c1, c2, c3, c4, c5 = (params[k] for k in ("c1", "c2", "c3", "c4", "c5"))
    params["plow"] = _parabola(-1.0 / 3.0, c1, 0.0, c2, 1.0 / 3.0, c3)
    params["phigh"] = _parabola(1.0 / 3.0, c3, 2.0 / 3.0, c4, 1.0, c5)


def eps_f(fail, triax: np.ndarray) -> np.ndarray:
    """Failure strain at the given triaxiality (vectorized)."""
    al, bl, cl = fail.params["plow"]
    ah, bh, ch = fail.params["phigh"]
    low = triax <= 1.0 / 3.0
    e = np.where(low,
                 al * triax ** 2 + bl * triax + cl,
                 ah * triax ** 2 + bh * triax + ch)
    return np.maximum(e, _FLOOR)


def solid_step(fail, sig, d_epsp, deps, dt, dama):
    """3-D damage step (deps/dt unused: no rate term in BIQUAD)."""
    sm = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    s0, s1, s2 = sig[:, 0] - sm, sig[:, 1] - sm, sig[:, 2] - sm
    vm = np.sqrt(1.5 * (s0 ** 2 + s1 ** 2 + s2 ** 2)
                 + 3.0 * (sig[:, 3] ** 2 + sig[:, 4] ** 2 + sig[:, 5] ** 2))
    triax = sm / np.maximum(vm, _TINY)
    dama += np.maximum(d_epsp, 0.0) / eps_f(fail, triax)
    return dama >= 1.0


def shell_step(fail, sig, d_epsp, deps, dt, dama):
    """Plane-stress damage step for one layer."""
    sm = (sig[:, 0] + sig[:, 1]) / 3.0
    vm = np.sqrt(sig[:, 0] ** 2 - sig[:, 0] * sig[:, 1] + sig[:, 1] ** 2
                 + 3.0 * sig[:, 2] ** 2)
    triax = sm / np.maximum(vm, _TINY)
    dama += np.maximum(d_epsp, 0.0) / eps_f(fail, triax)
    return dama >= 1.0
