r"""
LAW52 — Gurson-Tvergaard-Needleman (GTN) porous metal plasticity material model
(/MAT/LAW52, /MAT/GURSON, /MAT/PLAS_GURS).

Upstream Fortran origin:
------------------------
- ``engine/source/materials/mat/mat052/sigeps52.F`` (3D solid continuum kernel)
- ``engine/source/materials/mat/mat052/sigeps52c.F`` (2D shell plane-stress kernel)
- ``starter/source/materials/mat/mat052/hm_read_mat52.F`` (parameter initialization)

Theory and Constitutive Equations:
----------------------------------
The GTN model represents ductile failure of metals through microvoid nucleation,
growth, and coalescence in an elasto-plastic matrix.

1. Gurson-Tvergaard-Needleman Yield Criterion:
   .. math::
       \Phi(q, P, \sigma_M, f^*) = \left(\frac{q}{\sigma_M}\right)^2
       + 2 q_1 f^* \cosh\left(\frac{3 q_2 P}{2 \sigma_M}\right)
       - (1 + q_3 f^{*2}) = 0

   where:
   - :math:`q = \sigma_{\mathrm{eq}} = \sqrt{\frac{3}{2}\mathbf{s}:\mathbf{s}}` is the von Mises equivalent stress;
   - :math:`P = \frac{1}{3}\mathrm{tr}(\boldsymbol{\sigma})` is the hydrostatic pressure (positive in tension);
   - :math:`\sigma_M` is the equivalent flow stress of the unvoided matrix material;
   - :math:`f^*` is the effective void volume fraction;
   - :math:`q_1, q_2, q_3` are Tvergaard constitutive fitting parameters (typically :math:`q_1=1.5, q_2=1.0, q_3=q_1^2=2.25`).

2. Effective Void Volume Fraction :math:`f^*` (Tvergaard-Needleman Coalescence):
   .. math::
       f^* = \begin{cases}
           f & \text{if } f \le f_c \\
           f_c + \frac{f_u - f_c}{f_F - f_c}(f - f_c) & \text{if } f > f_c
       \end{cases}
   with ultimate void volume fraction :math:`f_u = 1 / q_1`.

3. Evolution of Void Volume Fraction:
   .. math::
       \dot{f} = \dot{f}_g + \dot{f}_n
   - Void growth (conservation of mass):
     .. math::
         \Delta f_g = (1 - f) \mathrm{tr}(\Delta\boldsymbol{\varepsilon}^p)
   - Void nucleation (Chu & Needleman 1980 plastic-strain-controlled Gaussian model):
     .. math::
         \Delta f_n = \frac{f_N}{s_N \sqrt{2\pi}} \exp\left(-\frac{1}{2}
         \left(\frac{\varepsilon_M - \varepsilon_N}{s_N}\right)^2\right) \Delta\varepsilon_M

4. Matrix Plastic Work Equivalence:
   .. math::
       (1 - f) \sigma_M \Delta\varepsilon_M = \boldsymbol{\sigma} : \Delta\boldsymbol{\varepsilon}^p

5. Matrix Flow Stress and Hardening:
   Power-law hardening with Cowper-Symonds strain-rate sensitivity:
   .. math::
       \sigma_M = (A + B \varepsilon_M^n) \left[1 + \left(\frac{\dot{\varepsilon}_M}{C}\right)^{1/P}\right]

6. Cutting-Plane / Newton-Raphson Return Mapping:
   Matches ``sigeps52.F`` and ``sigeps52c.F`` cutting-plane algorithm.

7. Complete Ductile Rupture and Element Deletion:
   When :math:`f^* \ge f_u` or :math:`f \ge f_F` or under cavitation tension :math:`VA \le 0`,
   load-carrying capacity is lost: stresses are zeroed and ``off = 0.0``.

8. Shell Plane-Stress Enforcement (:math:`\sigma_{zz} = 0`):
   Thickness thinning increment:
   .. math::
       \Delta\varepsilon_{zz} = \frac{\nu}{1 - \nu} (\Delta\varepsilon_{xx}^p + \Delta\varepsilon_{yy}^p) + \Delta\varepsilon_{zz}^p

9. Instantaneous Sound Speeds:
   - Solid: :math:`c_{\mathrm{solid}} = \sqrt{\frac{E(1-\nu)}{(1+\nu)(1-2\nu)\rho_0}}`
   - Shell: :math:`c_{\mathrm{shell}} = \sqrt{\frac{E}{(1-\nu^2)\rho_0}}`

10. Algorithmic Consistent Tangents:
    Consistent elasto-plastic tangents for solid :math:`(n, 6, 6)` and shell :math:`(n, 3, 3)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Numerical guards matching OpenRadioss implicit_f.inc / param_c.inc
_EM20 = 1e-20
_EP20 = 1e20
_SQRT2PI = math.sqrt(2.0 * math.pi)


@dataclass
class Law52Params:
    """Parameters for /MAT/LAW52 (Gurson porous metal plasticity)."""
    E: float = 210000.0          # Young's modulus
    nu: float = 0.3              # Poisson's ratio
    rho0: float = 7.8e-9         # Initial density
    yield_a: float = 400.0       # Matrix initial yield stress (A)
    hard_b: float = 500.0        # Matrix hardening coefficient (B)
    hard_n: float = 0.2          # Matrix hardening exponent (n)
    csd: float = 0.0             # Strain rate coefficient C (0 = rate insensitive)
    visp: float = 1.0            # Strain rate exponent P (default 1.0)
    q1: float = 1.5              # GTN parameter q1
    q2: float = 1.0              # GTN parameter q2
    q3: float = 2.25             # GTN parameter q3 (usually q1^2)
    sn: float = 0.1              # Gaussian nucleation standard deviation s_N
    epsn: float = 0.3            # Gaussian nucleation mean strain eps_N
    fi: float = 0.0              # Initial void volume fraction f_I
    fn: float = 0.04             # Nucleating void volume fraction f_N
    fc: float = 0.15             # Critical void volume fraction f_C
    ff: float = 0.25             # Failure void volume fraction f_F
    fu: float = 0.0              # Ultimate void volume fraction f_u (default 1/q1)
    iflag: int = 0               # Formulation flag (0=standard GTN)
    table: Any = None            # Optional yield table

    def __post_init__(self):
        if self.fu <= 0.0:
            self.fu = 1.0 / max(self.q1, _EM20)
        if self.q3 <= 0.0:
            self.q3 = self.q1 * self.q1


def _extract_params(mat: Any) -> Law52Params:
    """Extract Law52Params from a Material entity, dict, or Law52Params object."""
    if isinstance(mat, Law52Params):
        return mat

    p: Dict[str, Any] = {}
    rho0 = 0.0
    if hasattr(mat, "params") and isinstance(mat.params, dict):
        p.update(mat.params)
        rho0 = getattr(mat, "rho0", 0.0)
    elif isinstance(mat, dict):
        p.update(mat.get("params", {}))
        p.update(mat)
        rho0 = mat.get("rho0", mat.get("rho", mat.get("MAT_RHO", 0.0)))
    else:
        p = getattr(mat, "__dict__", {})
        rho0 = getattr(mat, "rho0", 0.0)

    p_low = {str(k).lower(): v for k, v in p.items()}

    # Elastic constants
    E = float(p_low.get("e", p_low.get("young", p_low.get("mat_e", 210000.0))))
    nu = float(p_low.get("nu", p_low.get("mat_nu", 0.3)))
    if rho0 <= 0.0:
        rho0 = float(p_low.get("rho0", p_low.get("rho", p_low.get("mat_rho", 7.8e-9))))

    # Matrix plasticity constants
    yield_a = float(p_low.get("a", p_low.get("yield_a", p_low.get("yield", p_low.get("yeild", p_low.get("mat_a", 400.0))))))
    hard_b = float(p_low.get("b", p_low.get("hard_b", p_low.get("et", p_low.get("mat_b", 500.0)))))
    hard_n = float(p_low.get("n", p_low.get("hard_n", p_low.get("mat_n", 0.2))))

    # Rate sensitivity
    csd = float(p_low.get("c", p_low.get("csd", p_low.get("mat_c", 0.0))))
    visp = float(p_low.get("p", p_low.get("pc", p_low.get("visp", p_low.get("mat_pc", 1.0)))))

    # Gurson parameters
    q1 = float(p_low.get("q1", p_low.get("mat_q1", 1.5)))
    q2 = float(p_low.get("q2", p_low.get("mat_q2", 1.0)))
    q3 = float(p_low.get("q3", p_low.get("mat_q3", q1 * q1)))

    # Nucleation parameters
    sn = float(p_low.get("s_n", p_low.get("sn", p_low.get("mat_s_n", 0.1))))
    epsn = float(p_low.get("eps_n", p_low.get("epsn", p_low.get("mat_eps_n", 0.3))))

    # Void fractions
    fi = float(p_low.get("f_i", p_low.get("fi", p_low.get("mat_f_i", 0.0))))
    fn = float(p_low.get("f_n", p_low.get("fn", p_low.get("mat_f_n", 0.04))))
    fc = float(p_low.get("f_c", p_low.get("fc", p_low.get("mat_f_c", 0.15))))
    ff = float(p_low.get("f_f", p_low.get("ff", p_low.get("mat_f_f", 0.25))))
    fu = float(p_low.get("f_u", p_low.get("fu", 1.0 / max(q1, _EM20))))

    iflag = int(p_low.get("iflag", p_low.get("mat_iflag", 0)))
    table = p.get("table", p_low.get("table", None))

    return Law52Params(
        E=E, nu=nu, rho0=rho0,
        yield_a=yield_a, hard_b=hard_b, hard_n=hard_n,
        csd=csd, visp=visp,
        q1=q1, q2=q2, q3=q3,
        sn=sn, epsn=epsn,
        fi=fi, fn=fn, fc=fc, ff=ff, fu=fu,
        iflag=iflag, table=table
    )


def build_law52(rec: Any = None, **kwargs) -> Any:
    """Physics constructor for /MAT/LAW52 (/MAT/GURSON, /MAT/PLAS_GURS)."""
    from ..model.entities import Material
    params: Dict[str, Any] = {}
    _id = 0
    _title = "LAW52_GURSON"
    _rho0 = 0.0

    if rec is not None:
        if isinstance(rec, dict):
            params.update(rec.get("params", {}))
            params.update(rec)
            _id = rec.get("id", 0)
            _title = rec.get("title", "LAW52")
            _rho0 = rec.get("rho0", rec.get("rho", rec.get("MAT_RHO", 0.0)))
        elif hasattr(rec, "params") and isinstance(rec.params, dict):
            params.update(rec.params)
            _id = getattr(rec, "id", 0)
            _title = getattr(rec, "title", "LAW52")
            _rho0 = getattr(rec, "rho0", 0.0)

    params.update(kwargs)
    if "rho0" in kwargs:
        _rho0 = float(kwargs["rho0"])

    parsed = _extract_params({"params": params, "rho0": _rho0})
    params["E"] = parsed.E
    params["nu"] = parsed.nu
    params["rho0"] = parsed.rho0
    params["A"] = parsed.yield_a
    params["B"] = parsed.hard_b
    params["n"] = parsed.hard_n
    params["CSD"] = parsed.csd
    params["VISP"] = parsed.visp
    params["q1"] = parsed.q1
    params["q2"] = parsed.q2
    params["q3"] = parsed.q3
    params["s_N"] = parsed.sn
    params["eps_N"] = parsed.epsn
    params["f_I"] = parsed.fi
    params["f_N"] = parsed.fn
    params["f_C"] = parsed.fc
    params["f_F"] = parsed.ff
    params["f_u"] = parsed.fu
    params["iflag"] = parsed.iflag

    return Material(
        id=_id,
        law=52,
        rho0=parsed.rho0,
        title=_title,
        params=params,
        law_name="LAW52",
    )


# ===================================================================
# Acoustic Sound Speed Calculations
# ===================================================================

def sound_speed_solid_law52(mat: Any, rho: Optional[Union[float, np.ndarray]] = None, extra: Any = None) -> Union[float, np.ndarray]:
    """Instantaneous sound speed for 3D solid continuum elements:

    .. math::
        c_{\\mathrm{solid}} = \\sqrt{\\frac{E(1-\\nu)}{(1+\\nu)(1-2\\nu)\\rho_0}}
    """
    p = _extract_params(mat)
    r = rho if rho is not None else p.rho0
    c1 = p.E * (1.0 - p.nu) / ((1.0 + p.nu) * (1.0 - 2.0 * p.nu))
    if isinstance(r, np.ndarray):
        r_val = np.where(r > 0.0, r, p.rho0 if p.rho0 > 0.0 else 1.0)
        c_sq = c1 / np.maximum(r_val, _EM20)
        return np.sqrt(np.maximum(c_sq, _EM20))
    else:
        r_val = r if (r is not None and r > 0.0) else (p.rho0 if p.rho0 > 0.0 else 1.0)
        c_sq = c1 / max(r_val, _EM20)
        return math.sqrt(max(c_sq, _EM20))


def sound_speed_shell_law52(mat: Any, rho: Optional[Union[float, np.ndarray]] = None, extra: Any = None) -> Union[float, np.ndarray]:
    """Instantaneous sound speed for 2D shell plane-stress elements:

    .. math::
        c_{\\mathrm{shell}} = \\sqrt{\\frac{E}{(1-\\nu^2)\\rho_0}}
    """
    p = _extract_params(mat)
    r = rho if rho is not None else p.rho0
    a1 = p.E / (1.0 - p.nu * p.nu)
    if isinstance(r, np.ndarray):
        r_val = np.where(r > 0.0, r, p.rho0 if p.rho0 > 0.0 else 1.0)
        c_sq = a1 / np.maximum(r_val, _EM20)
        return np.sqrt(np.maximum(c_sq, _EM20))
    else:
        r_val = r if (r is not None and r > 0.0) else (p.rho0 if p.rho0 > 0.0 else 1.0)
        c_sq = a1 / max(r_val, _EM20)
        return math.sqrt(max(c_sq, _EM20))


# ===================================================================
# GTN Auxiliary Functions
# ===================================================================

def compute_f_star(f: Union[float, np.ndarray], fc: float, ff: float, fu: float) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute effective void volume fraction f* and derivative df*/df.

    .. math::
        f^* = \\begin{cases}
            f & \\text{if } f \\le f_c \\\\
            f_c + \\frac{f_u - f_c}{f_F - f_c}(f - f_c) & \\text{if } f > f_c
        \\end{cases}
    """
    is_array = isinstance(f, np.ndarray)
    f_arr = np.asarray(f, dtype=float)

    denom = ff - fc
    df_factor = (fu - fc) / denom if abs(denom) > 1e-15 else _EP20

    f_star = np.where(f_arr <= fc, f_arr, fc + df_factor * (f_arr - fc))
    df = np.where(f_arr <= fc, 1.0, df_factor)

    if not is_array:
        return float(f_star), float(df)
    return f_star, df


def gurson_yield_function(
    q: Union[float, np.ndarray],
    p: Union[float, np.ndarray],
    sig_m: Union[float, np.ndarray],
    f_star: Union[float, np.ndarray],
    q1: float = 1.5,
    q2: float = 1.0,
    q3: float = 2.25,
) -> Union[float, np.ndarray]:
    """Evaluate Gurson-Tvergaard-Needleman yield function:

    .. math::
        \\Phi(q, P, \\sigma_M, f^*) = \\left(\\frac{q}{\\sigma_M}\\right)^2
        + 2 q_1 f^* \\cosh\\left(\\frac{3 q_2 P}{2 \\sigma_M}\\right) - (1 + q_3 f^{*2})
    """
    sig_m_safe = np.maximum(sig_m, _EM20)
    var = 1.5 * q2 * p / sig_m_safe
    var_clipped = np.clip(var, -80.0, 80.0)
    cosh_term = np.cosh(var_clipped)
    phi = (q / sig_m_safe) ** 2 + 2.0 * q1 * f_star * cosh_term - (1.0 + q3 * (f_star ** 2))
    return phi


def _matrix_flow_stress(
    p: Law52Params,
    epsm: np.ndarray,
    epsp_rate: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate matrix flow stress sigma_M and hardening slope d(sigma_M)/d(eps_M)."""
    epsm_safe = np.maximum(epsm, _EM20)

    if p.csd > 0.0 and p.visp > 0.0:
        inv_c = 1.0 / p.csd
        inv_p = 1.0 / p.visp
        rate_term = 1.0 + np.maximum(epsp_rate * inv_c, 0.0) ** inv_p
    else:
        rate_term = np.ones_like(epsm)

    sig_m = (p.yield_a + p.hard_b * (epsm_safe ** p.hard_n)) * rate_term

    if p.hard_n == 1.0:
        dsepp = p.hard_b * rate_term
    else:
        dsepp = p.hard_b * p.hard_n * (epsm_safe ** (p.hard_n - 1.0)) * rate_term

    return sig_m, dsepp


# ===================================================================
# Solid Continuum Stress Update (sigeps52.F)
# ===================================================================

def solid_update_law52(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = False,
) -> Any:
    """Vectorized cutting-plane GTN solid update.

    Faithfully implements ``engine/source/materials/mat/mat052/sigeps52.F``.

    Parameters
    ----------
    mat : Material or Law52Params or dict
        Material definition.
    sig : np.ndarray
        Stress tensor (n, 6) or (6,) = [sxx, syy, szz, sxy, syz, szx]. Updated in place.
    deps : np.ndarray
        Strain increment tensor (n, 6) or (6,) with engineering shear.
    epsp : np.ndarray, optional
        Macroscopic equivalent plastic strain (n,) or scalar. Updated in place.
    dt : float
        Time step increment for strain rate calculation.
    extra : dict, optional
        Persistent internal state:
        - 'epsm': (n,) matrix plastic strain
        - 'sigm': (n,) matrix flow stress
        - 'dmg': (n, 5) damage tensor: [f*, fg, fn, f, f*]
        - 'off': (n,) active status mask (1.0=active, 0.0=deleted)
    return_sound_speed : bool, optional
        If True, return (sig, epsp, sound_speed).

    Returns
    -------
    sig, epsp (or sig, epsp, sound_speed)
    """
    params = _extract_params(mat)
    orig_shape = sig.shape
    if sig.ndim == 1:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)
        if epsp is not None and np.ndim(epsp) == 0:
            epsp = np.array([epsp], dtype=float)

    n = sig.shape[0]
    if n == 0:
        c_val = sound_speed_solid_law52(params)
        if return_sound_speed:
            return sig, (epsp if epsp is not None else np.zeros(0)), np.full(0, c_val)
        return sig, (epsp if epsp is not None else np.zeros(0))

    if extra is None:
        extra = {}

    if "epsm" not in extra:
        extra["epsm"] = np.full(n, _EM20 if params.hard_n < 1.0 else 0.0, dtype=float)
    epsm = extra["epsm"]

    if "sigm" not in extra:
        extra["sigm"] = np.full(n, params.yield_a, dtype=float)
    sigm = extra["sigm"]

    if "dmg" not in extra:
        dmg = np.zeros((n, 5), dtype=float)
        f_init = params.fi
        f_star_init, _ = compute_f_star(f_init, params.fc, params.ff, params.fu)
        dmg[:, 0] = f_star_init  # Global damage f*
        dmg[:, 1] = 0.0          # Void growth fg
        dmg[:, 2] = 0.0          # Void nucleation fn
        dmg[:, 3] = f_init       # Total void volume fraction f
        dmg[:, 4] = f_star_init  # Effective void volume fraction f*
        if "fg" in extra and np.any(extra["fg"]):
            dmg[:, 1] = extra["fg"]
        if "fn" in extra and np.any(extra["fn"]):
            dmg[:, 2] = extra["fn"]
        if "f" in extra and np.any(extra["f"]):
            dmg[:, 3] = extra["f"]
        if "fstar" in extra and np.any(extra["fstar"]):
            dmg[:, 0] = extra["fstar"]
            dmg[:, 4] = extra["fstar"]
        extra["dmg"] = dmg
    dmg = extra["dmg"]

    if "off" not in extra:
        if "off52" in extra:
            extra["off"] = extra["off52"].copy()
        else:
            extra["off"] = np.ones(n, dtype=float)
    off = extra["off"]

    if epsp is None:
        epsp = np.zeros(n, dtype=float)
    else:
        epsp = np.asarray(epsp, dtype=float)

    # Elastic moduli
    E = params.E
    nu = params.nu
    G = 0.5 * E / (1.0 + nu)
    c11 = E / (3.0 * (1.0 - 2.0 * nu))  # Bulk modulus K
    c1 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = c1 * nu / (1.0 - nu)
    c_sound = math.sqrt(max((c11 + (4.0 / 3.0) * G) / max(params.rho0, _EM20), _EM20))

    # Pre-rupture check: if element is already failed or deleted
    for i in range(n):
        if off[i] <= 0.0 or dmg[i, 3] >= params.ff or dmg[i, 4] >= params.fu:
            off[i] = 0.0
            sig[i, :] = 0.0

    # Elastic trial predictor (sigeps52.F lines 253-261)
    sig[:, 0] += c1 * deps[:, 0] + c2 * (deps[:, 1] + deps[:, 2])
    sig[:, 1] += c1 * deps[:, 1] + c2 * (deps[:, 0] + deps[:, 2])
    sig[:, 2] += c1 * deps[:, 2] + c2 * (deps[:, 0] + deps[:, 1])
    sig[:, 3] += G * deps[:, 3]
    sig[:, 4] += G * deps[:, 4]
    sig[:, 5] += G * deps[:, 5]

    # Enforce zero stress on deleted elements
    dead = off <= 0.0
    sig[dead] = 0.0

    # Chu-Needleman nucleation prefactor
    sn_safe = max(params.sn, _EM20)
    a1_nucleation = (params.fn / (sn_safe * _SQRT2PI)) * np.exp(
        -0.5 * (((epsm - params.epsn) / sn_safe) ** 2)
    )

    # Macroscopic strain rate estimate
    if dt > 0.0:
        deps_vm = np.sqrt(
            (2.0 / 9.0) * (
                (deps[:, 0] - deps[:, 1]) ** 2 +
                (deps[:, 1] - deps[:, 2]) ** 2 +
                (deps[:, 2] - deps[:, 0]) ** 2 +
                1.5 * (deps[:, 3] ** 2 + deps[:, 4] ** 2 + deps[:, 5] ** 2)
            )
        )
        epsp_rate = deps_vm / dt
    else:
        epsp_rate = np.zeros(n, dtype=float)

    # Cutting-plane plastic return mapping per element
    for i in range(n):
        if off[i] <= 0.0:
            continue

        f_curr = dmg[i, 3]
        f_star_curr, df_curr = compute_f_star(f_curr, params.fc, params.ff, params.fu)

        # Check failure limits
        if f_star_curr >= params.fu or f_curr >= params.ff:
            off[i] = 0.0
            sig[i, :] = 0.0
            continue

        # Hydrostatic pressure (tension > 0)
        pn = (sig[i, 0] + sig[i, 1] + sig[i, 2]) / 3.0
        sxx = sig[i, 0] - pn
        syy = sig[i, 1] - pn
        szz = sig[i, 2] - pn
        j2 = 0.5 * (sxx ** 2 + syy ** 2 + szz ** 2) + sig[i, 3] ** 2 + sig[i, 4] ** 2 + sig[i, 5] ** 2
        vm = math.sqrt(max(3.0 * j2, 0.0))

        sigm_i = max(sigm[i], _EM20)
        sigm1 = 1.0 / sigm_i
        var = 1.5 * params.q2 * pn * sigm1
        var_clipped = max(min(var, 80.0), -80.0)
        coh = math.cosh(var_clipped)
        sih = math.sinh(var_clipped)

        va = 1.0 + params.q3 * (f_star_curr ** 2) - 2.0 * params.q1 * f_star_curr * coh

        # Hydrostatic cavitation rupture under extreme tension: va <= 0 and pn > 0
        if va <= 0.0 and pn > 0.0:
            off[i] = 0.0
            sig[i, :] = 0.0
            dmg[i, 0] = params.fu
            dmg[i, 3] = max(dmg[i, 3], params.ff)
            dmg[i, 4] = params.fu
            continue

        va_sqrt = math.sqrt(max(0.0, va))
        yld = sigm_i * va_sqrt

        # Check yield
        if vm <= yld or yld <= 0.0:
            continue

        # Plastic flow requires return mapping
        dp11 = dp22 = dp33 = dp12 = dp13 = dp23 = 0.0
        sig_tr = sig[i].copy()

        a21 = _EP20 if f_curr >= 1.0 else (sigm1 / (1.0 - f_curr))
        a1_i = a1_nucleation[i]

        # 5 Cutting-plane iterations
        for _ in range(5):
            vm1 = 1.0 / max(vm, _EM20)
            va1 = 1.0 / max(va_sqrt, _EM20)
            va11 = 0.5 * params.q1 * params.q2 * f_star_curr * sih * va1

            # Flow gradient D = dPhi / dSigma
            d11 = 0.5 * (2.0 * sig[i, 0] - sig[i, 1] - sig[i, 2]) * vm1 + va11
            d22 = 0.5 * (2.0 * sig[i, 1] - sig[i, 0] - sig[i, 2]) * vm1 + va11
            d33 = 0.5 * (2.0 * sig[i, 2] - sig[i, 0] - sig[i, 1]) * vm1 + va11
            d12 = 3.0 * sig[i, 3] * vm1
            d13 = 3.0 * sig[i, 5] * vm1
            d23 = 3.0 * sig[i, 4] * vm1

            # Plastic work rate
            a2 = (d11 * sig[i, 0] + d22 * sig[i, 1] + d33 * sig[i, 2] +
                  2.0 * (d12 * sig[i, 3] + d13 * sig[i, 5] + d23 * sig[i, 4])) * a21

            dcrf = -sigm_i * (params.q3 * f_star_curr * df_curr - params.q1 * coh * df_curr) * va1
            dcrm = -va_sqrt - 3.0 * va11 * pn * sigm1

            # Matrix hardening slope
            if params.hard_n == 1.0:
                dsepp = params.hard_b
            else:
                dsepp = params.hard_b * params.hard_n * (max(epsm[i], _EM20) ** (params.hard_n - 1.0))
            if params.csd > 0.0 and params.visp > 0.0:
                dsepp *= (1.0 + (epsp_rate[i] / params.csd) ** (1.0 / params.visp))

            dcd = (c1 * (d11 ** 2 + d22 ** 2 + d33 ** 2) +
                   2.0 * c2 * (d11 * d22 + d11 * d33 + d22 * d33) +
                   2.0 * G * (d12 ** 2 + d13 ** 2 + d23 ** 2))

            tr_d = d11 + d22 + d33
            lam1 = dcd - dcrm * dsepp * a2 - dcrf * ((1.0 - f_curr) * tr_d + a1_i * a2)

            lamda = max(0.0, vm - yld) / max(lam1, 1e-15)

            dp11 += lamda * d11
            dp22 += lamda * d22
            dp33 += lamda * d33
            dp12 += lamda * d12
            dp13 += lamda * d13
            dp23 += lamda * d23

            # Update stress
            sig[i, 0] = sig_tr[0] - c1 * dp11 - c2 * (dp22 + dp33)
            sig[i, 1] = sig_tr[1] - c1 * dp22 - c2 * (dp11 + dp33)
            sig[i, 2] = sig_tr[2] - c1 * dp33 - c2 * (dp11 + dp22)
            sig[i, 3] = sig_tr[3] - 2.0 * G * dp12
            sig[i, 4] = sig_tr[4] - 2.0 * G * dp23
            sig[i, 5] = sig_tr[5] - 2.0 * G * dp13

            # Recompute invariants
            pn = (sig[i, 0] + sig[i, 1] + sig[i, 2]) / 3.0
            sxx = sig[i, 0] - pn
            syy = sig[i, 1] - pn
            szz = sig[i, 2] - pn
            j2 = 0.5 * (sxx ** 2 + syy ** 2 + szz ** 2) + sig[i, 3] ** 2 + sig[i, 4] ** 2 + sig[i, 5] ** 2
            vm = math.sqrt(max(3.0 * j2, 0.0))

            var = 1.5 * params.q2 * pn * sigm1
            var_clipped = max(min(var, 80.0), -80.0)
            coh = math.cosh(var_clipped)
            sih = math.sinh(var_clipped)
            va = 1.0 + params.q3 * (f_star_curr ** 2) - 2.0 * params.q1 * f_star_curr * coh
            va_sqrt = math.sqrt(max(0.0, va))
            yld = sigm_i * va_sqrt

        # Plastic work of matrix: (1 - f) * sigma_M * d_eps_M = sigma : d_eps_p
        deps_m = (sig[i, 0] * dp11 + sig[i, 1] * dp22 + sig[i, 2] * dp33 +
                  2.0 * (sig[i, 3] * dp12 + sig[i, 5] * dp13 + sig[i, 4] * dp23)) * a21
        deps_m = max(0.0, deps_m)

        # Void evolution
        delta_fg = (1.0 - f_curr) * (dp11 + dp22 + dp33)
        delta_fn = a1_i * deps_m
        if params.iflag in (2, 3) and pn < 0.0:
            delta_fn = 0.0  # Nucleation only under tension if IFLAG=2 or 3

        dmg[i, 1] += delta_fg           # fg
        dmg[i, 2] += delta_fn           # fn
        epsm[i] += deps_m
        epsp[i] += math.sqrt((2.0 / 3.0) * (
            dp11 ** 2 + dp22 ** 2 + dp33 ** 2 + 2.0 * (dp12 ** 2 + dp13 ** 2 + dp23 ** 2)
        ))

        f_new = params.fi + dmg[i, 1] + dmg[i, 2]
        if params.q1 == 0.0 and params.q2 == 0.0 and params.q3 == 0.0:
            f_new = 0.0
        f_new = max(0.0, f_new)
        dmg[i, 3] = f_new               # f

        # Update matrix flow stress
        sigm_new, _ = _matrix_flow_stress(
            params, np.array([epsm[i]]), np.array([epsp_rate[i]])
        )
        sigm[i] = sigm_new[0]

        # Update effective void volume fraction f*
        f_star_new, _ = compute_f_star(f_new, params.fc, params.ff, params.fu)
        dmg[i, 0] = f_star_new          # Global damage
        dmg[i, 4] = f_star_new          # f*

        # Rupture / element deletion criterion: f* >= fu or f >= ff
        if f_star_new >= params.fu or f_new >= params.ff:
            off[i] = 0.0
            sig[i, :] = 0.0

    if "fg" in extra:
        extra["fg"][:] = dmg[:, 1]
    if "fn" in extra:
        extra["fn"][:] = dmg[:, 2]
    if "f" in extra:
        extra["f"][:] = dmg[:, 3]
    if "fstar" in extra:
        extra["fstar"][:] = dmg[:, 0]
    if "off52" in extra:
        extra["off52"][:] = off[:]
    else:
        extra["off52"] = off.copy()

    if orig_shape == (6,):
        sig = sig.reshape(6)
        epsp = epsp[0]

    if return_sound_speed:
        return sig, epsp, np.full(n, c_sound)
    return sig, epsp


# ===================================================================
# Shell Plane-Stress Update (sigeps52c.F)
# ===================================================================

def shell_update_law52(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = False,
) -> Any:
    """Vectorized cutting-plane GTN shell update under plane stress (sigma_zz = 0).

    Faithfully implements ``engine/source/materials/mat/mat052/sigeps52c.F``.

    Parameters
    ----------
    mat : Material or Law52Params or dict
        Material definition.
    sig : np.ndarray
        In-plane stress tensor (n, 3) or (3,) = [sxx, syy, sxy] (or (n, 5) with transverse shear).
    deps : np.ndarray
        In-plane strain increment tensor (n, 3) or (3,) with engineering shear.
    epsp : np.ndarray, optional
        Macroscopic equivalent plastic strain (n,) or scalar. Updated in place.
    dt : float
        Time step increment for strain rate calculation.
    extra : dict, optional
        Persistent internal state:
        - 'epsm': (n,) matrix plastic strain
        - 'sigm': (n,) matrix flow stress
        - 'dmg': (n, 5) damage tensor: [f*, fg, fn, f, f*]
        - 'thk': (n,) current shell thickness
        - 'thk0': (n,) initial shell thickness
        - 'off': (n,) active status mask (1.0=active, 0.0=deleted)
    return_sound_speed : bool, optional
        If True, return (sig, epsp, sound_speed).

    Returns
    -------
    sig, epsp (or sig, epsp, sound_speed)
    """
    params = _extract_params(mat)
    orig_shape = sig.shape

    if sig.ndim == 1:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)
        if epsp is not None and np.ndim(epsp) == 0:
            epsp = np.array([epsp], dtype=float)

    n = sig.shape[0]
    ncomp = sig.shape[1]
    if n == 0:
        c_val = sound_speed_shell_law52(params)
        if return_sound_speed:
            return sig, (epsp if epsp is not None else np.zeros(0)), np.full(0, c_val)
        return sig, (epsp if epsp is not None else np.zeros(0))

    if extra is None:
        extra = {}

    if "epsm" not in extra:
        extra["epsm"] = np.full(n, _EM20 if params.hard_n < 1.0 else 0.0, dtype=float)
    epsm = extra["epsm"]

    if "sigm" not in extra:
        extra["sigm"] = np.full(n, params.yield_a, dtype=float)
    sigm = extra["sigm"]

    if "dmg" not in extra:
        dmg = np.zeros((n, 5), dtype=float)
        f_init = params.fi
        f_star_init, _ = compute_f_star(f_init, params.fc, params.ff, params.fu)
        dmg[:, 0] = f_star_init
        dmg[:, 1] = 0.0
        dmg[:, 2] = 0.0
        dmg[:, 3] = f_init
        dmg[:, 4] = f_star_init
        if "fg" in extra and np.any(extra["fg"]):
            dmg[:, 1] = extra["fg"]
        if "fn" in extra and np.any(extra["fn"]):
            dmg[:, 2] = extra["fn"]
        if "f" in extra and np.any(extra["f"]):
            dmg[:, 3] = extra["f"]
        if "fstar" in extra and np.any(extra["fstar"]):
            dmg[:, 0] = extra["fstar"]
            dmg[:, 4] = extra["fstar"]
        extra["dmg"] = dmg
    dmg = extra["dmg"]

    if "off" not in extra:
        if "off52" in extra:
            extra["off"] = extra["off52"].copy()
        else:
            extra["off"] = np.ones(n, dtype=float)
    off = extra["off"]

    if "thk" not in extra:
        extra["thk"] = np.ones(n, dtype=float)
    if "thk0" not in extra:
        extra["thk0"] = extra["thk"].copy()
    thk = extra["thk"]
    thk0 = extra["thk0"]

    if epsp is None:
        epsp = np.zeros(n, dtype=float)
    else:
        epsp = np.asarray(epsp, dtype=float)

    # Plane-stress elastic constants
    E = params.E
    nu = params.nu
    a1 = E / (1.0 - nu * nu)
    a2 = nu * a1
    G = 0.5 * E / (1.0 + nu)
    GS = G  # Transverse shear stiffness
    c1 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = c1 * nu / (1.0 - nu)
    nn1 = nu / (1.0 - nu)
    c_sound = math.sqrt(max(a1 / max(params.rho0, _EM20), _EM20))

    # Pre-rupture check
    for i in range(n):
        if off[i] <= 0.0 or dmg[i, 3] >= params.ff or dmg[i, 4] >= params.fu:
            off[i] = 0.0
            sig[i, :] = 0.0

    # Elastic predictor (sigeps52c.F lines 272-276)
    sig[:, 0] += a1 * deps[:, 0] + a2 * deps[:, 1]
    sig[:, 1] += a2 * deps[:, 0] + a1 * deps[:, 1]
    sig[:, 2] += G * deps[:, 2]
    if ncomp >= 5:
        sig[:, 3] += GS * deps[:, 3]
        sig[:, 4] += GS * deps[:, 4]

    # Initial elastic thickness thinning
    dezz_el = -nn1 * (deps[:, 0] + deps[:, 1])
    thk += dezz_el * thk0 * off

    dead = off <= 0.0
    sig[dead] = 0.0

    # Chu-Needleman nucleation prefactor
    sn_safe = max(params.sn, _EM20)
    a1_nucleation = (params.fn / (sn_safe * _SQRT2PI)) * np.exp(
        -0.5 * (((epsm - params.epsn) / sn_safe) ** 2)
    )

    # Macroscopic in-plane strain rate
    if dt > 0.0:
        rate = np.sqrt(
            (deps[:, 0] ** 2 + deps[:, 1] ** 2 - deps[:, 0] * deps[:, 1] + 0.75 * (deps[:, 2] ** 2))
        ) / dt
    else:
        rate = np.zeros(n, dtype=float)

    for i in range(n):
        if off[i] <= 0.0:
            continue

        f_curr = dmg[i, 3]
        f_star_curr, df_curr = compute_f_star(f_curr, params.fc, params.ff, params.fu)

        if f_star_curr >= params.fu or f_curr >= params.ff:
            off[i] = 0.0
            sig[i, :] = 0.0
            continue

        # Plane-stress von Mises stress (sigeps52c.F lines 315-317)
        s11 = sig[i, 0]
        s22 = sig[i, 1]
        s12 = sig[i, 2]
        s23 = sig[i, 3] if ncomp >= 5 else 0.0
        s31 = sig[i, 4] if ncomp >= 5 else 0.0

        vm2 = s11 ** 2 + s22 ** 2 - s11 * s22 + 3.0 * (s12 ** 2 + s23 ** 2 + s31 ** 2)
        vm = math.sqrt(max(vm2, 0.0))

        # Hydrostatic pressure (plane stress: sigma_zz = 0)
        pn = (s11 + s22) / 3.0

        sigm_i = max(sigm[i], _EM20)
        sigm1 = 1.0 / sigm_i
        var = 1.5 * params.q2 * pn * sigm1
        var_clipped = max(min(var, 80.0), -80.0)
        coh = math.cosh(var_clipped)
        sih = math.sinh(var_clipped)

        va = 1.0 + params.q3 * (f_star_curr ** 2) - 2.0 * params.q1 * f_star_curr * coh

        if va <= 0.0 and pn > 0.0:
            off[i] = 0.0
            sig[i, :] = 0.0
            dmg[i, 0] = params.fu
            dmg[i, 3] = max(dmg[i, 3], params.ff)
            dmg[i, 4] = params.fu
            continue

        va_sqrt = math.sqrt(max(0.0, va))
        yld = sigm_i * va_sqrt

        if vm <= yld or yld <= 0.0:
            continue

        # Cutting-plane plastic return
        dp11 = dp22 = dp33 = dp12 = dp13 = dp23 = 0.0
        sig_tr = sig[i].copy()

        a21 = _EP20 if f_curr >= 1.0 else (sigm1 / (1.0 - f_curr))
        a1_i = a1_nucleation[i]

        for _ in range(5):
            vm1 = 1.0 / max(vm, _EM20)
            va1 = 1.0 / max(va_sqrt, _EM20)
            va2 = 0.5 * params.q1 * params.q2 * f_star_curr * sih * va1

            # Plane-stress flow direction
            d11 = 0.5 * (2.0 * sig[i, 0] - sig[i, 1]) * vm1 + va2
            d22 = 0.5 * (2.0 * sig[i, 1] - sig[i, 0]) * vm1 + va2
            d33 = 0.5 * (-sig[i, 0] - sig[i, 1]) * vm1 + va2
            d12 = 3.0 * sig[i, 2] * vm1
            d23 = 3.0 * (sig[i, 3] if ncomp >= 5 else 0.0) * vm1
            d13 = 3.0 * (sig[i, 4] if ncomp >= 5 else 0.0) * vm1

            a22 = (d11 * sig[i, 0] + d22 * sig[i, 1] +
                   2.0 * (d12 * sig[i, 2] + d13 * (sig[i, 4] if ncomp >= 5 else 0.0) +
                          d23 * (sig[i, 3] if ncomp >= 5 else 0.0))) * a21

            dcrf = -sigm_i * (params.q3 * f_star_curr * df_curr - params.q1 * coh * df_curr) * va1
            dcrm = -va_sqrt - 3.0 * va2 * pn * sigm1

            if params.hard_n == 1.0:
                dsepp = params.hard_b
            else:
                dsepp = params.hard_b * params.hard_n * (max(epsm[i], _EM20) ** (params.hard_n - 1.0))
            if params.csd > 0.0 and params.visp > 0.0:
                dsepp *= (1.0 + (rate[i] / params.csd) ** (1.0 / params.visp))

            dcd = (c1 * (d11 ** 2 + d22 ** 2 + d33 ** 2) +
                   2.0 * c2 * (d11 * d22 + d11 * d33 + d22 * d33) +
                   2.0 * G * (d12 ** 2) + 2.0 * GS * (d13 ** 2 + d23 ** 2))

            tr_d = d11 + d22 + d33
            lam1 = dcd - dcrm * dsepp * a22 - dcrf * ((1.0 - f_curr) * tr_d + a1_i * a22)

            lamda = max(0.0, vm - yld) / max(lam1, 1e-15)

            dp11 += lamda * d11
            dp22 += lamda * d22
            dp33 += lamda * d33
            dp12 += lamda * d12
            dp23 += lamda * d23
            dp13 += lamda * d13

            # Update in-plane stress
            sig[i, 0] = sig_tr[0] - a1 * dp11 - a2 * dp22
            sig[i, 1] = sig_tr[1] - a2 * dp11 - a1 * dp22
            sig[i, 2] = sig_tr[2] - 2.0 * G * dp12
            if ncomp >= 5:
                sig[i, 3] = sig_tr[3] - 2.0 * GS * dp23
                sig[i, 4] = sig_tr[4] - 2.0 * GS * dp13

            s11 = sig[i, 0]
            s22 = sig[i, 1]
            s12 = sig[i, 2]
            s23 = sig[i, 3] if ncomp >= 5 else 0.0
            s31 = sig[i, 4] if ncomp >= 5 else 0.0
            vm2 = s11 ** 2 + s22 ** 2 - s11 * s22 + 3.0 * (s12 ** 2 + s23 ** 2 + s31 ** 2)
            vm = math.sqrt(max(vm2, 0.0))

            pn = (s11 + s22) / 3.0
            var = 1.5 * params.q2 * pn * sigm1
            var_clipped = max(min(var, 80.0), -80.0)
            coh = math.cosh(var_clipped)
            sih = math.sinh(var_clipped)
            va = 1.0 + params.q3 * (f_star_curr ** 2) - 2.0 * params.q1 * f_star_curr * coh
            va_sqrt = math.sqrt(max(0.0, va))
            yld = sigm_i * va_sqrt

        # Plastic thickness thinning: Delta_eps_zz_pl = nn1 * (dp11 + dp22) + dp33
        dezz_pl = nn1 * (dp11 + dp22) + dp33
        thk[i] += dezz_pl * thk0[i] * off[i]

        deps_m = (sig[i, 0] * dp11 + sig[i, 1] * dp22 +
                  2.0 * (sig[i, 2] * dp12 +
                         (sig[i, 3] * dp23 if ncomp >= 5 else 0.0) +
                         (sig[i, 4] * dp13 if ncomp >= 5 else 0.0))) * a21
        deps_m = max(0.0, deps_m)

        delta_fg = (1.0 - f_curr) * (dp11 + dp22 + dp33)
        delta_fn = a1_i * deps_m
        if params.iflag in (2, 3) and pn < 0.0:
            delta_fn = 0.0

        dmg[i, 1] += delta_fg
        dmg[i, 2] += delta_fn
        epsm[i] += deps_m
        epsp[i] += math.sqrt((2.0 / 3.0) * (dp11 ** 2 + dp22 ** 2 + dp33 ** 2 + 2.0 * dp12 ** 2))

        f_new = params.fi + dmg[i, 1] + dmg[i, 2]
        if params.q1 == 0.0 and params.q2 == 0.0 and params.q3 == 0.0:
            f_new = 0.0
        f_new = max(0.0, f_new)
        dmg[i, 3] = f_new

        sigm_new, _ = _matrix_flow_stress(
            params, np.array([epsm[i]]), np.array([rate[i]])
        )
        sigm[i] = sigm_new[0]

        f_star_new, _ = compute_f_star(f_new, params.fc, params.ff, params.fu)
        dmg[i, 0] = f_star_new
        dmg[i, 4] = f_star_new

        if f_star_new >= params.fu or f_new >= params.ff:
            off[i] = 0.0
            sig[i, :] = 0.0

    if "fg" in extra:
        extra["fg"][:] = dmg[:, 1]
    if "fn" in extra:
        extra["fn"][:] = dmg[:, 2]
    if "f" in extra:
        extra["f"][:] = dmg[:, 3]
    if "fstar" in extra:
        extra["fstar"][:] = dmg[:, 0]
    if "off52" in extra:
        extra["off52"][:] = off[:]
    else:
        extra["off52"] = off.copy()
    if "layfail" in extra:
        extra["layfail"][off == 0.0] = 0.0

    if orig_shape == (3,) or orig_shape == (5,):
        sig = sig.reshape(orig_shape[0])
        epsp = epsp[0]

    if return_sound_speed:
        return sig, epsp, np.full(n, c_sound)
    return sig, epsp


# ===================================================================
# Algorithmic Consistent Tangent Operators
# ===================================================================

def tangent_law52_solid(
    mat: Any,
    sig: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Algorithmic consistent elasto-plastic tangent for 3D solids, shape (n, 6, 6) or (6, 6).

    .. math::
        \\mathbf{C}^{\\mathrm{ep}} = \\mathbf{C}^{\\mathrm{el}}
        - \\frac{(\\mathbf{C}^{\\mathrm{el}} : \\mathbf{D}) \\otimes (\\mathbf{D} : \\mathbf{C}^{\\mathrm{el}})}
          {LAM1}
    """
    params = _extract_params(mat)
    is_1d = sig.ndim == 1
    if is_1d:
        sig = sig.reshape(1, -1)
        if epsp is not None and np.ndim(epsp) == 0:
            epsp = np.array([epsp])
        if epsp_incr is not None and np.ndim(epsp_incr) == 0:
            epsp_incr = np.array([epsp_incr])

    n = sig.shape[0]
    E = params.E
    nu = params.nu
    G = 0.5 * E / (1.0 + nu)
    c1 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = c1 * nu / (1.0 - nu)

    # Base elastic stiffness matrix (6x6 Voigt)
    Cel = np.zeros((6, 6), dtype=float)
    Cel[0, 0] = Cel[1, 1] = Cel[2, 2] = c1
    Cel[0, 1] = Cel[1, 0] = Cel[0, 2] = Cel[2, 0] = Cel[1, 2] = Cel[2, 1] = c2
    Cel[3, 3] = Cel[4, 4] = Cel[5, 5] = G

    D_tangent = np.broadcast_to(Cel, (n, 6, 6)).copy()

    off = extra.get("off", np.ones(n)) if extra else np.ones(n)
    sigm = extra.get("sigm", np.full(n, params.yield_a)) if extra else np.full(n, params.yield_a)
    epsm = extra.get("epsm", np.zeros(n)) if extra else np.zeros(n)
    dmg = extra.get("dmg", None) if extra else None

    # Identify plastic elements
    if epsp_incr is not None:
        plastic = (epsp_incr > 1e-12) & (off > 0.0)
    else:
        plastic = np.zeros(n, dtype=bool)

    dead = off <= 0.0
    D_tangent[dead] = 0.0

    if not np.any(plastic):
        if is_1d:
            return D_tangent[0]
        return D_tangent

    idx = np.where(plastic)[0]
    for i in idx:
        f_curr = dmg[i, 3] if dmg is not None else params.fi
        f_star_curr, df_curr = compute_f_star(f_curr, params.fc, params.ff, params.fu)

        pn = (sig[i, 0] + sig[i, 1] + sig[i, 2]) / 3.0
        sxx = sig[i, 0] - pn
        syy = sig[i, 1] - pn
        szz = sig[i, 2] - pn
        j2 = 0.5 * (sxx ** 2 + syy ** 2 + szz ** 2) + sig[i, 3] ** 2 + sig[i, 4] ** 2 + sig[i, 5] ** 2
        vm = math.sqrt(max(3.0 * j2, _EM20))

        sigm_i = max(sigm[i], _EM20)
        sigm1 = 1.0 / sigm_i
        var = 1.5 * params.q2 * pn * sigm1
        var_clipped = max(min(var, 80.0), -80.0)
        coh = math.cosh(var_clipped)
        sih = math.sinh(var_clipped)

        va = 1.0 + params.q3 * (f_star_curr ** 2) - 2.0 * params.q1 * f_star_curr * coh
        va_sqrt = math.sqrt(max(0.0, va))
        va1 = 1.0 / max(va_sqrt, _EM20)
        va11 = 0.5 * params.q1 * params.q2 * f_star_curr * sih * va1

        vm1 = 1.0 / vm
        D_flow = np.zeros(6, dtype=float)
        D_flow[0] = 0.5 * (2.0 * sig[i, 0] - sig[i, 1] - sig[i, 2]) * vm1 + va11
        D_flow[1] = 0.5 * (2.0 * sig[i, 1] - sig[i, 0] - sig[i, 2]) * vm1 + va11
        D_flow[2] = 0.5 * (2.0 * sig[i, 2] - sig[i, 0] - sig[i, 1]) * vm1 + va11
        D_flow[3] = 3.0 * sig[i, 3] * vm1
        D_flow[4] = 3.0 * sig[i, 4] * vm1
        D_flow[5] = 3.0 * sig[i, 5] * vm1

        a21 = _EP20 if f_curr >= 1.0 else (sigm1 / (1.0 - f_curr))
        a1_i = (params.fn / (max(params.sn, _EM20) * _SQRT2PI)) * math.exp(
            -0.5 * (((epsm[i] - params.epsn) / max(params.sn, _EM20)) ** 2)
        )
        a2 = (D_flow[0] * sig[i, 0] + D_flow[1] * sig[i, 1] + D_flow[2] * sig[i, 2] +
              2.0 * (D_flow[3] * sig[i, 3] + D_flow[4] * sig[i, 4] + D_flow[5] * sig[i, 5])) * a21

        dcrf = -sigm_i * (params.q3 * f_star_curr * df_curr - params.q1 * coh * df_curr) * va1
        dcrm = -va_sqrt - 3.0 * va11 * pn * sigm1

        if params.hard_n == 1.0:
            dsepp = params.hard_b
        else:
            dsepp = params.hard_b * params.hard_n * (max(epsm[i], _EM20) ** (params.hard_n - 1.0))

        CD = Cel @ D_flow
        DCD = float(D_flow @ CD)

        tr_d = D_flow[0] + D_flow[1] + D_flow[2]
        lam1 = DCD - dcrm * dsepp * a2 - dcrf * ((1.0 - f_curr) * tr_d + a1_i * a2)
        lam1 = max(lam1, 1e-12)

        D_tangent[i] = Cel - np.outer(CD, CD) / lam1

    if is_1d:
        return D_tangent[0]
    return D_tangent


def tangent_law52_shell(
    mat: Any,
    sig: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    epsp_incr: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Algorithmic consistent elasto-plastic tangent for plane-stress shells, shape (n, 3, 3) or (3, 3).

    .. math::
        \\mathbf{C}^{\\mathrm{ep}} = \\mathbf{C}^{\\mathrm{el}}
        - \\frac{(\\mathbf{C}^{\\mathrm{el}} : \\mathbf{D}) \\otimes (\\mathbf{D} : \\mathbf{C}^{\\mathrm{el}})}
          {LAM1}
    """
    params = _extract_params(mat)
    is_1d = sig.ndim == 1
    if is_1d:
        sig = sig.reshape(1, -1)
        if epsp is not None and np.ndim(epsp) == 0:
            epsp = np.array([epsp])
        if epsp_incr is not None and np.ndim(epsp_incr) == 0:
            epsp_incr = np.array([epsp_incr])

    n = sig.shape[0]
    E = params.E
    nu = params.nu
    a1 = E / (1.0 - nu * nu)
    a2 = nu * a1
    G = 0.5 * E / (1.0 + nu)
    c1 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = c1 * nu / (1.0 - nu)

    Cel = np.zeros((3, 3), dtype=float)
    Cel[0, 0] = a1
    Cel[1, 1] = a1
    Cel[0, 1] = Cel[1, 0] = a2
    Cel[2, 2] = G

    D_tangent = np.broadcast_to(Cel, (n, 3, 3)).copy()

    off = extra.get("off", np.ones(n)) if extra else np.ones(n)
    sigm = extra.get("sigm", np.full(n, params.yield_a)) if extra else np.full(n, params.yield_a)
    epsm = extra.get("epsm", np.zeros(n)) if extra else np.zeros(n)
    dmg = extra.get("dmg", None) if extra else None

    if epsp_incr is not None:
        plastic = (epsp_incr > 1e-12) & (off > 0.0)
    else:
        plastic = np.zeros(n, dtype=bool)

    dead = off <= 0.0
    D_tangent[dead] = 0.0

    if not np.any(plastic):
        if is_1d:
            return D_tangent[0]
        return D_tangent

    idx = np.where(plastic)[0]
    for i in idx:
        f_curr = dmg[i, 3] if dmg is not None else params.fi
        f_star_curr, df_curr = compute_f_star(f_curr, params.fc, params.ff, params.fu)

        s11 = sig[i, 0]
        s22 = sig[i, 1]
        s12 = sig[i, 2]
        vm2 = s11 ** 2 + s22 ** 2 - s11 * s22 + 3.0 * (s12 ** 2)
        vm = math.sqrt(max(vm2, _EM20))
        pn = (s11 + s22) / 3.0

        sigm_i = max(sigm[i], _EM20)
        sigm1 = 1.0 / sigm_i
        var = 1.5 * params.q2 * pn * sigm1
        var_clipped = max(min(var, 80.0), -80.0)
        coh = math.cosh(var_clipped)
        sih = math.sinh(var_clipped)

        va = 1.0 + params.q3 * (f_star_curr ** 2) - 2.0 * params.q1 * f_star_curr * coh
        va_sqrt = math.sqrt(max(0.0, va))
        va1 = 1.0 / max(va_sqrt, _EM20)
        va2 = 0.5 * params.q1 * params.q2 * f_star_curr * sih * va1

        vm1 = 1.0 / vm
        d11 = 0.5 * (2.0 * sig[i, 0] - sig[i, 1]) * vm1 + va2
        d22 = 0.5 * (2.0 * sig[i, 1] - sig[i, 0]) * vm1 + va2
        d33 = 0.5 * (-sig[i, 0] - sig[i, 1]) * vm1 + va2
        d12 = 3.0 * sig[i, 2] * vm1

        D_shell = np.array([d11, d22, d12], dtype=float)

        a21 = _EP20 if f_curr >= 1.0 else (sigm1 / (1.0 - f_curr))
        a1_i = (params.fn / (max(params.sn, _EM20) * _SQRT2PI)) * math.exp(
            -0.5 * (((epsm[i] - params.epsn) / max(params.sn, _EM20)) ** 2)
        )
        a22 = (d11 * sig[i, 0] + d22 * sig[i, 1] + 2.0 * d12 * sig[i, 2]) * a21

        dcrf = -sigm_i * (params.q3 * f_star_curr * df_curr - params.q1 * coh * df_curr) * va1
        dcrm = -va_sqrt - 3.0 * va2 * pn * sigm1

        if params.hard_n == 1.0:
            dsepp = params.hard_b
        else:
            dsepp = params.hard_b * params.hard_n * (max(epsm[i], _EM20) ** (params.hard_n - 1.0))

        dcd = (c1 * (d11 ** 2 + d22 ** 2 + d33 ** 2) +
               2.0 * c2 * (d11 * d22 + d11 * d33 + d22 * d33) +
               2.0 * G * (d12 ** 2))

        tr_d = d11 + d22 + d33
        lam1 = dcd - dcrm * dsepp * a22 - dcrf * ((1.0 - f_curr) * tr_d + a1_i * a22)
        lam1 = max(lam1, 1e-12)

        CD = Cel @ D_shell
        D_tangent[i] = Cel - np.outer(CD, CD) / lam1

    if is_1d:
        return D_tangent[0]
    return D_tangent


def shell_membrane_tangent(mat: Any) -> np.ndarray:
    """Return 3x3 plane-stress elastic membrane tangent matrix for LAW52."""
    p = _extract_params(mat)
    c = p.E / max(1.0 - p.nu * p.nu, 1e-15)
    g = p.E / max(2.0 * (1.0 + p.nu), 1e-15)
    return np.array([
        [c, p.nu * c, 0.0],
        [p.nu * c, c, 0.0],
        [0.0, 0.0, g],
    ], dtype=np.float64)


# Standard function aliases for solver compatibility
solid_update = solid_update_law52
shell_update = shell_update_law52
sound_speed_solid = sound_speed_solid_law52
sound_speed_shell = sound_speed_shell_law52
sound_speed = sound_speed_solid_law52
consistent_solid_tangent = tangent_law52_solid
consistent_shell_tangent = tangent_law52_shell
solid_tangent = tangent_law52_solid
shell_tangent = tangent_law52_shell


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Per-element persistent state shapes for LAW52."""
    if nip is None:
        return {
            "epsm": (),
            "sigm": (),
            "dmg": (5,),
            "fg": (),
            "fn": (),
            "f": (),
            "fstar": (),
            "off": (),
            "off52": (),
        }
    return {
        "epsm": (nip,),
        "sigm": (nip,),
        "dmg": (nip, 5),
        "fg": (nip,),
        "fn": (nip,),
        "f": (nip,),
        "fstar": (nip,),
        "thk": (nip,),
        "thk0": (nip,),
        "off": (nip,),
        "off52": (nip,),
    }


def _register():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in (52, "52", "LAW52", "GURSON", "PLAS_GURS", "MAT_LAW52", "MAT_GURSON", "MAT_PLAS_GURS"):
            MAT_PHYSICS_REGISTRY[key] = build_law52
    except Exception:
        pass


_register()
