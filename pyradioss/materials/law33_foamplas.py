"""LAW33 — Crushable foam plasticity (/MAT/LAW33, /MAT/FOAM_PLAS).

Fortran origin: ``engine/source/materials/mat/mat033/sigeps33.F`` (the
constitutive update — 3 ICASE branches) and
``starter/source/materials/mat/mat033/hm_read_mat33.F`` (parameter
extraction).

Theory (the sigeps33 algorithm)
-------------------------------
A crushable closed-cell polyurethane foam with an air-pressure
contribution from the closed cells and a principal-stress return.

All three formulations share:
  1. volumetric strain ``gamma = rho0/rho - 1 + gamma0``
  2. air pressure ``sig_air = max(0, -P0*gamma/(1 + gamma - phi + 1e-15))``
  3. yield stress from ``|A + B*(1+C*gamma)|`` or a function table
     (IFUNC1) scaled by ``FAC``, optionally multiplied by a strain-rate
     function (IFUNC2) scaled by ``FAC1``
  4. principal-stress return: eigendecomposition of the trial stress
     tensor, clamp each principal stress magnitude to yield, back-rotate
  5. final stress: subtract air pressure from normals

ICASE 1 (KEN=0): simple elastic trial + principal return
ICASE 2 (KEN=1): rate-dependent Kelvin viscoelastic model + principal
                  return (skip return when KEN < 0)
ICASE 3 (KEN=2): same as ICASE 1 but with tension cutoff (SIGT_CUTOFF)

Sound speed: ``c = sqrt(E/rho0)`` (ICASE 1/3), ``c = sqrt(E_eff/rho0)``
(ICASE 2 with rate-dependent E_eff).

Solids only (``SOLID_ISOTROPIC`` + SPH).
"""

from __future__ import annotations

import math

import numpy as np

from pyradioss.model.entities import Material

# Fortran constants
_ZERO = 0.0
_HALF = 0.5
_ONE = 1.0
_THIRD = 1.0 / 3.0
_THREE = 3.0
_THREE_HALF = 1.5
_EP20 = 1.0e20
_EM10 = 1.0e-10
_EM15 = 1.0e-15


# ------------------------------------------------------------------ #
# Parameter extraction (hm_read_mat33.F)
# ------------------------------------------------------------------ #

def build_law33(rec) -> Material:
    """``hm_read_mat33.F``: card parsing and validation → Material.

    Accepts a ``Material`` instance, a dict (from parsed deck), or an
    object with attributes matching the CFG field names.
    """
    # --- unpack input --------------------------------------------------
    if isinstance(rec, Material):
        p = dict(rec.params) if rec.params else {}
        mat_id = rec.id
        title = rec.title
        density = rec.rho0
    elif isinstance(rec, dict):
        p = dict(rec.get("params", rec))
        mat_id = rec.get("id", 1)
        title = rec.get("title", "")
        val = None
        for k in ("density", "rho", "rho0", "MAT_RHO"):
            if k in rec and rec[k] is not None:
                val = rec[k]; break
        if val is None:
            for k in ("density", "rho", "rho0", "MAT_RHO"):
                if k in p and p[k] is not None:
                    val = p[k]; break
        density = float(val) if val is not None else 1.0
    else:
        p = dict(getattr(rec, "params", {}))
        mat_id = getattr(rec, "id", 1)
        title = getattr(rec, "title", "")
        val = None
        for k in ("density", "rho", "rho0", "MAT_RHO"):
            if hasattr(rec, k) and getattr(rec, k) is not None:
                val = getattr(rec, k); break
        density = float(val) if val is not None else 1.0

    if density <= 0.0:
        raise ValueError("LAW33: Density rho0 must be positive (hm_read_mat33 check)")

    # --- helper to fetch from p with fallback chains ---
    def _g(keys, default=0.0):
        for k in keys:
            v = p.get(k)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return default

    def _gi(keys, default=0):
        for k in keys:
            v = p.get(k)
            if v is not None:
                try:
                    return int(v)
                except (ValueError, TypeError):
                    pass
        return default

    e = _g(["MAT_E", "e", "E"])
    if e <= 0.0:
        raise ValueError(f"LAW33/{mat_id}: Young modulus E must be > 0")

    ken = _gi(["Itype", "itype", "KEN", "ken"])
    ifn1 = _gi(["FUN_A1", "fun_a1", "ifn1", "IFN1"])
    fac = _g(["IFscale", "ifscale", "fac", "FAC"])
    if fac == 0.0:
        fac = 1.0   # hm_read_mat33: IF (FAC == ZERO) FAC = ONE * FAC_UNIT

    # Hidden parameters (not exposed in current HM reader)
    ifn2 = 0
    fac1 = 0.0

    p0 = _g(["MAT_P0", "p0", "P0"])
    phi = _g(["MAT_PHI", "phi", "PHI"])
    gama0 = _g(["MAT_GAMA0", "gama0", "gamma0", "GAMA0"])
    a = _g(["MAT_A0", "a0", "A0", "a", "A"])
    b = _g(["MAT_A1", "a1", "A1", "b", "B"])
    c = _g(["MAT_A2", "a2", "A2", "c_coeff", "C"])

    icase = abs(ken) + 1

    params = {
        "E": e, "nu": 0.0,   # foam: nu ≈ 0 for the generic elastic properties
        "KEN": ken,
        "A": a, "B": b, "C": c,
        "P0": p0, "phi": phi, "gama0": gama0,
        "FAC": fac, "FAC1": fac1,
        "IFN1": ifn1, "IFN2": ifn2,
    }

    if icase == 3:
        # KEN = 2 (or -2): tension cutoff branch
        sigt_cutoff = _g(["MAT_SIGT_CUTOFF", "sigt_cutoff", "SIGT_CUTOFF"])
        if sigt_cutoff == 0.0:
            sigt_cutoff = _EP20   # hm_read_mat33: IF(SIGT_COFF==ZERO) SIGT_COFF=EP20
        params["SIGT_CUTOFF"] = sigt_cutoff
    elif icase == 2:
        # KEN = 1: Kelvin model parameters
        c1_val = _g(["MAT_E1", "e1", "E1", "c1_kelvin"])
        c2_val = _g(["MAT_E2", "e2", "E2", "c2_kelvin"])
        et = _g(["MAT_ETAN", "etan", "ETAN", "et", "ET"])
        vmu = _g(["MAT_ETA1", "eta1", "ETA1", "vmu", "VMU"])
        vmu0 = _g(["MAT_ETA2", "eta2", "ETA2", "vmu0", "VMU0"])
        if vmu <= 0.0 or vmu0 <= 0.0:
            raise ValueError(
                f"LAW33/{mat_id}: viscous coefficients VMU (eta1={vmu:g}) "
                f"and VMU0 (eta2={vmu0:g}) must be > 0 (hm_read_mat33 error 310)")
        params.update({"C1_kelvin": c1_val, "C2_kelvin": c2_val,
                       "Et": et, "VMU": vmu, "VMU0": vmu0})

    mat = Material(id=mat_id, law=33, rho0=density, title=title, params=params)
    return mat


def resolve(mat: Material, model, log) -> None:
    """Pull optional /FUNCT yield curve (IFN1) into plain arrays.

    Called from ``starter/initialization.py:resolve_materials()`` after
    all /FUNCT cards have been parsed — deck order between /MAT and
    /FUNCT is free (hm_read_mat33 stores IFN1, the starter resolves the
    actual function data later).
    """
    p = mat.params
    ifn1 = p.get("IFN1", 0)
    if ifn1 and ifn1 != 0:
        fct = model.functions.get(ifn1)
        if fct is None:
            if hasattr(log, "error"):
                log.error(f"/MAT/LAW33/{mat.id}: function {ifn1} "
                          f"(FUN_A1) not defined", "MAT CHECK")
            return
        p["yield_curve"] = (fct.x.copy(), fct.y.copy())


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _air_pressure(rho0, rho, p0, phi, gama0):
    """Closed-cell air pressure (sigeps33 lines 142-146).

    gamma = rho0/rho - 1 + gama0
    sig_air = max(0, -P0 * gamma / (1 + gamma - phi + 1e-15))
    """
    gamma = rho0 / rho - _ONE + gama0
    var = -(p0 * gamma) / (_ONE + gamma - phi + _EM15)
    sig_air = np.maximum(_ZERO, var)
    return gamma, sig_air


def _compute_yield(gamma, syield_formula, fac, yield_curve=None, deps_rate=None, rate_curve=None, fac1=0.0):
    """Compute yield stress from formula or function table.

    Matches sigeps33 lines 155-173 (ICASE=1), 307-325 (ICASE=2),
    460-478 (ICASE=3).
    """
    a_coeff, b_coeff, c_coeff = syield_formula

    if yield_curve is not None:
        xs, ys = yield_curve
        syield = fac * np.interp(gamma, xs, ys)
    else:
        syield = np.abs(a_coeff + b_coeff * (_ONE + c_coeff * gamma))

    if rate_curve is not None and deps_rate is not None:
        xs_r, ys_r = rate_curve
        rate_factor = np.interp(deps_rate, xs_r, ys_r)
        syield = fac1 * syield * rate_factor

    return syield


def _principal_return_clamp(sig_trial, syield):
    """Eigenvalue decomposition, clamp, back-rotate (vectorized).

    Matches the VALPVEC_V path of sigeps33 (lines 229-267, 405-438).
    sig_trial: (n, 6) Voigt [xx, yy, zz, xy, yz, zx]
    syield: (n,) per-element yield
    Returns: (n, 6) clamped stress in global frame.
    """
    n = sig_trial.shape[0]
    if n == 0:
        return sig_trial.copy()

    # Build symmetric 3x3 from Voigt
    S = np.zeros((n, 3, 3), dtype=sig_trial.dtype)
    S[:, 0, 0] = sig_trial[:, 0]
    S[:, 1, 1] = sig_trial[:, 1]
    S[:, 2, 2] = sig_trial[:, 2]
    S[:, 0, 1] = sig_trial[:, 3]
    S[:, 1, 0] = sig_trial[:, 3]
    S[:, 1, 2] = sig_trial[:, 4]
    S[:, 2, 1] = sig_trial[:, 4]
    S[:, 2, 0] = sig_trial[:, 5]
    S[:, 0, 2] = sig_trial[:, 5]

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(S)  # (n,3), (n,3,3)

    # Clamp: sign(eigval) * min(|eigval|, syield)
    sy = syield[:, None]   # (n, 1)
    clamped = np.sign(eigvals) * np.minimum(np.abs(eigvals), sy)

    # Back-rotate: sig_global = V @ diag(clamped) @ V.T
    # Efficient: (n,3,3) @ (n,3,3) @ (n,3,3)
    S_new = np.einsum("nik,nk,njk->nij", eigvecs, clamped, eigvecs)

    # Extract Voigt
    sig_out = np.empty_like(sig_trial)
    sig_out[:, 0] = S_new[:, 0, 0]
    sig_out[:, 1] = S_new[:, 1, 1]
    sig_out[:, 2] = S_new[:, 2, 2]
    sig_out[:, 3] = S_new[:, 0, 1]
    sig_out[:, 4] = S_new[:, 1, 2]
    sig_out[:, 5] = S_new[:, 2, 0]

    return sig_out


def _principal_return_tension_cutoff(sig_trial, syield, gamma, sigt_cutoff):
    """Eigenvalue decomposition with tension cutoff (ICASE=3).

    Matches sigeps33 lines 498-566 (KEN=2 return regimes).
    """
    n = sig_trial.shape[0]
    if n == 0:
        return sig_trial.copy()

    # Build symmetric 3x3 from Voigt
    S = np.zeros((n, 3, 3), dtype=sig_trial.dtype)
    S[:, 0, 0] = sig_trial[:, 0]
    S[:, 1, 1] = sig_trial[:, 1]
    S[:, 2, 2] = sig_trial[:, 2]
    S[:, 0, 1] = sig_trial[:, 3]
    S[:, 1, 0] = sig_trial[:, 3]
    S[:, 1, 2] = sig_trial[:, 4]
    S[:, 2, 1] = sig_trial[:, 4]
    S[:, 2, 0] = sig_trial[:, 5]
    S[:, 0, 2] = sig_trial[:, 5]

    eigvals, eigvecs = np.linalg.eigh(S)  # (n,3), (n,3,3)

    # Apply cutoff regimes element-by-element (branchy logic)
    clamped = eigvals.copy()
    for i in range(n):
        g = gamma[i]
        sy = syield[i]
        sc = sigt_cutoff

        if abs(g) < _EM10:
            # Near zero volumetric strain: clamp to ±SIGT_CUTOFF
            for k in range(3):
                clamped[i, k] = math.copysign(min(abs(clamped[i, k]), sc), clamped[i, k])
        elif g < _ZERO:
            # Compression (gamma < 0): clamp to ±SYIELD
            for k in range(3):
                clamped[i, k] = math.copysign(min(abs(clamped[i, k]), sy), clamped[i, k])
            # Then cap tension at SIGT_CUTOFF
            for k in range(3):
                clamped[i, k] = min(clamped[i, k], sc)
        else:
            # Expansion (gamma > 0): clamp to ±SIGT_CUTOFF, then zero negative
            for k in range(3):
                clamped[i, k] = math.copysign(min(abs(clamped[i, k]), sc), clamped[i, k])
            for k in range(3):
                clamped[i, k] = max(clamped[i, k], _ZERO)

    # Back-rotate
    S_new = np.einsum("nik,nk,njk->nij", eigvecs, clamped, eigvecs)

    sig_out = np.empty_like(sig_trial)
    sig_out[:, 0] = S_new[:, 0, 0]
    sig_out[:, 1] = S_new[:, 1, 1]
    sig_out[:, 2] = S_new[:, 2, 2]
    sig_out[:, 3] = S_new[:, 0, 1]
    sig_out[:, 4] = S_new[:, 1, 2]
    sig_out[:, 5] = S_new[:, 2, 0]

    return sig_out


# ------------------------------------------------------------------ #
# Constitutive update (solid only)
# ------------------------------------------------------------------ #

def solid_update(mat, sig, deps, epsp, dt, extra=None):
    """sigeps33.F: LAW33 solid stress update.

    Parameters
    ----------
    mat : Material  — law 33 with params from build_law33
    sig : (n, 6)    — old stress (Voigt: xx, yy, zz, xy, yz, zx)
    deps : (n, 6)   — strain increment (engineering shear)
    epsp : any       — unused (no scalar plastic strain)
    dt : float       — time step
    extra : dict     — must contain 'rho': (n,) current density

    Returns
    -------
    (sig_new, epsp, c) where c is sound speed array or None.
    """
    n = sig.shape[0]
    if n == 0:
        return sig.copy(), epsp, None

    p = mat.params
    ken = p["KEN"]
    icase = abs(ken) + 1
    e = p["E"]
    a_coeff = p["A"]
    b_coeff = p["B"]
    c_coeff = p["C"]
    p0 = p["P0"]
    phi = p["phi"]
    gama0 = p["gama0"]
    fac = p["FAC"]

    # Get density
    rho0 = mat.rho0
    if extra is not None and "rho" in extra:
        rho = np.atleast_1d(np.asarray(extra["rho"], dtype=sig.dtype))
        if rho.ndim == 0:
            rho = np.full(n, float(rho))
    else:
        rho = np.full(n, rho0, dtype=sig.dtype)

    rho0_arr = np.full(n, rho0, dtype=sig.dtype)

    # Volumetric strain and air pressure
    gamma, sig_air = _air_pressure(rho0_arr, rho, p0, phi, gama0)

    # Yield stress
    yield_curve = None
    if p.get("IFN1") and p.get("IFN1") != 0:
        yield_curve = p.get("yield_curve")

    syield = _compute_yield(gamma, (a_coeff, b_coeff, c_coeff), fac,
                            yield_curve=yield_curve)

    if dt <= 0.0:
        return sig.copy(), epsp, None

    # Strain rate (deps / dt)
    deps_rate = deps / dt

    if icase == 1:
        # ---- KEN=0: Simple elastic trial + principal return ----------
        sig_s = sig.copy()
        # Add air pressure to normals (sigeps33 lines 175-182)
        sig_s[:, 0] += sig_air
        sig_s[:, 1] += sig_air
        sig_s[:, 2] += sig_air

        # Trial stress (sigeps33 lines 185-192)
        # Fortran: EY * EPSPXX * TIMESTEP = E * deps (rate × dt = increment)
        sig_s[:, 0] += e * deps[:, 0]
        sig_s[:, 1] += e * deps[:, 1]
        sig_s[:, 2] += e * deps[:, 2]
        sig_s[:, 3] += e * deps[:, 3] * _HALF
        sig_s[:, 4] += e * deps[:, 4] * _HALF
        sig_s[:, 5] += e * deps[:, 5] * _HALF

        # Principal stress return
        sig_s = _principal_return_clamp(sig_s, syield)

        # Final: subtract air pressure (sigeps33 lines 271-280)
        sig_new = sig_s.copy()
        sig_new[:, 0] -= sig_air
        sig_new[:, 1] -= sig_air
        sig_new[:, 2] -= sig_air

        c = np.sqrt(e / rho0_arr)
        return sig_new, epsp, c

    elif icase == 2:
        # ---- KEN=1: Kelvin viscoelastic model + principal return -----
        c1_k = p["C1_kelvin"]
        c2_k = p["C2_kelvin"]
        et = p["Et"]
        vmu = p["VMU"]
        vmu0 = p["VMU0"]

        # Rate-dependent Young modulus (sigeps33 lines 294-304)
        edot = np.maximum(
            np.abs(deps_rate[:, 0]),
            np.maximum(np.abs(deps_rate[:, 1]),
            np.maximum(np.abs(deps_rate[:, 2]),
            np.maximum(np.abs(deps_rate[:, 3]),
            np.maximum(np.abs(deps_rate[:, 4]),
                       np.abs(deps_rate[:, 5]))))))
        e_eff = np.maximum(c1_k * edot + c2_k, e)

        epet = (e_eff + et) / vmu
        emet = (e_eff * et) / vmu
        epets = (e_eff + et) / vmu0
        emets = 2.0 * (e_eff * et) / vmu0

        sig_s = sig.copy()
        # Add air pressure
        sig_s[:, 0] += sig_air
        sig_s[:, 1] += sig_air
        sig_s[:, 2] += sig_air

        # Get total strain from extra and update in-place FIRST
        # (Fortran mulaw.F90 lines 881-895: strain = strain + de BEFORE
        # calling sigeps33, so EPSXX inside sigeps33 is eps^{n+1}).
        # The extra["eps33"] is a view into persistent st["mat_extra"],
        # so in-place += modifies the backing array directly.
        if extra is not None and "eps33" in extra:
            extra["eps33"][:] += deps      # in-place update on the view
            eps_total = extra["eps33"]     # reference, NOT copy
        else:
            eps_total = deps.copy()

        # Stress rates (sigeps33 lines 350-357)
        # Normal components: dsig = E*deps_rate - EPET*sig + EMET*eps
        # Shear components:  dsig = E*deps_rate - EPETS*sig + EMETS*eps (with half factors)
        dsig = np.zeros_like(sig)
        for comp in range(3):
            dsig[:, comp] = (e_eff * deps_rate[:, comp]
                             - epet * sig_s[:, comp]
                             + emet * eps_total[:, comp])
        for comp in range(3, 6):
            dsig[:, comp] = (e_eff * (deps_rate[:, comp] * _HALF)
                             - epets * sig_s[:, comp]
                             + emets * (eps_total[:, comp] * _HALF))

        # Trial stress
        sig_s += dsig * dt

        # Principal stress return (skip if KEN < 0)
        if ken >= 0:
            sig_s = _principal_return_clamp(sig_s, syield)

        # Final: subtract air pressure
        sig_new = sig_s.copy()
        sig_new[:, 0] -= sig_air
        sig_new[:, 1] -= sig_air
        sig_new[:, 2] -= sig_air

        c = np.sqrt(e_eff / rho0_arr)
        return sig_new, epsp, c

    elif icase == 3:
        # ---- KEN=2: Elastic trial + tension cutoff return ------------
        sigt_cutoff = p["SIGT_CUTOFF"]

        sig_s = sig.copy()
        sig_s[:, 0] += sig_air
        sig_s[:, 1] += sig_air
        sig_s[:, 2] += sig_air

        # Trial stress (same as ICASE 1)
        sig_s[:, 0] += e * deps[:, 0]
        sig_s[:, 1] += e * deps[:, 1]
        sig_s[:, 2] += e * deps[:, 2]
        sig_s[:, 3] += e * deps[:, 3] * _HALF
        sig_s[:, 4] += e * deps[:, 4] * _HALF
        sig_s[:, 5] += e * deps[:, 5] * _HALF

        # Tension cutoff principal return
        sig_s = _principal_return_tension_cutoff(sig_s, syield, gamma, sigt_cutoff)

        sig_new = sig_s.copy()
        sig_new[:, 0] -= sig_air
        sig_new[:, 1] -= sig_air
        sig_new[:, 2] -= sig_air

        c = np.sqrt(e / rho0_arr)
        return sig_new, epsp, c

    else:
        raise NotImplementedError(f"LAW33 KEN={ken} (ICASE={icase}) not supported")


def shell_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """LAW33 is a solid/SPH-only law — shells are not supported."""
    raise NotImplementedError(
        "LAW33 (FOAM_PLAS) is for solid/SPH elements only — "
        "see hm_read_mat33.F INIT_MAT_KEYWORD(SOLID_ISOTROPIC)")


# ------------------------------------------------------------------ #
# Consistent tangent (implicit solver)
# ------------------------------------------------------------------ #

def consistent_solid_tangent(mat, sig, epsp=None, epsp_incr=None, extra=None):
    """Consistent solid tangent for the implicit solver.

    For ICASE=1/3 (elastic trial), the tangent is the isotropic elastic
    stiffness tensor.  For ICASE=2 (Kelvin model), the tangent accounts
    for the rate-dependent modulus.

    Returns (n, 6, 6).
    """
    n = sig.shape[0]
    if n == 0:
        return np.zeros((0, 6, 6), dtype=sig.dtype)

    p = mat.params
    ken = p["KEN"]
    icase = abs(ken) + 1
    e = p["E"]

    # For the elastic trial (ICASE 1/3), use E as the Young modulus
    # with nu ≈ 0 for foam (no lateral constraint — the Fortran has no
    # Poisson coupling at all in the trial stress integration).
    # D_ij = E * delta_ij for normals, E/2 for shears — matching the
    # trial:  dsig_xx = E * deps_xx,  dsig_xy = E/2 * deps_xy.
    if icase in (1, 3):
        D = np.zeros((n, 6, 6), dtype=sig.dtype)
        for i in range(3):
            D[:, i, i] = e
        for i in range(3, 6):
            D[:, i, i] = e * _HALF
        return D

    elif icase == 2:
        # Kelvin model: the instantaneous tangent is dominated by the
        # E_eff * I term from the stress rate dsig ≈ E_eff * deps
        # (the relaxation terms are proportional to existing stress/strain).
        # Use E_eff from the rate state if available.
        if extra is not None and "rho" in extra:
            rho = np.atleast_1d(np.asarray(extra.get("rho", mat.rho0), dtype=sig.dtype))
        else:
            rho = np.full(n, mat.rho0, dtype=sig.dtype)

        c1_k = p["C1_kelvin"]
        c2_k = p["C2_kelvin"]
        dt_val = float(extra.get("dt", 1e-6)) if extra is not None else 1e-6

        # Estimate E_eff from current strain rate (via epsp_incr/dt)
        if epsp_incr is not None and dt_val > 0:
            edot = np.max(np.abs(epsp_incr), axis=-1) / dt_val if epsp_incr.ndim > 1 else np.zeros(n)
        else:
            edot = np.zeros(n, dtype=sig.dtype)
        e_eff = np.maximum(c1_k * edot + c2_k, e)

        D = np.zeros((n, 6, 6), dtype=sig.dtype)
        for i in range(n):
            for j in range(3):
                D[i, j, j] = e_eff[i]
            for j in range(3, 6):
                D[i, j, j] = e_eff[i] * _HALF
        return D

    raise NotImplementedError(f"LAW33 KEN={ken} tangent not implemented")


# ------------------------------------------------------------------ #
# Registration
# ------------------------------------------------------------------ #

def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW33"] = build_law33
    MAT_PHYSICS_REGISTRY["FOAM_PLAS"] = build_law33


_register()
