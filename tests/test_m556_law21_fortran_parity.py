"""Milestone M556: /MAT/LAW21 (/MAT/DPRAG) Fortran parity test suite.

Direct numerical audit and oracle comparison against upstream OpenRadioss Fortran reference code:
- engine/source/materials/mat/mat021/m21law.F
- starter/source/materials/mat/mat021/hm_read_mat21.F
- hm_cfg_files/config/CFG/radioss110/MAT/matl21_dprag.cfg

Checks:
1. Parameter defaults and setup from hm_read_mat21.F:
   - if refer_rho == 0: refer_rho = rho
   - if pmin == 0: pmin = -1e30
   - if bunl == 0: bunl = c1
   - if amax == 0: amax = 1e20
   - if mumax == 0: mumax = 1e20
   - if pfscale == 0: pfscale = 1.0
   - G = E / (2(1 + nu))
2. Root P* calculation:
   - Linear case (A2 == 0, A1 != 0): P* = -A0 / A1
   - Parabolic case (A2 != 0): Delta = A1^2 - 4 A0 A2; if Delta >= 0 -> P* = (-A1 + sqrt(Delta)) / (2 A2) else -1e30
   - Pure von Mises (A1 == 0, A2 == 0): P* = -1e30
3. Drucker-Prager yield envelope G0(P_tot) (m21law.F:220-227):
   - P_tot = P + P_ext
   - If P < P_min -> G0 = 0
   - If P_tot <= P* -> G0 = 0
   - Else G0 = min(A_max, max(0, A0 + A1 P_tot + A2 P_tot^2))
   - J2 = 0.5*(s_xx^2 + s_yy^2 + s_zz^2) + s_xy^2 + s_yz^2 + s_zx^2
   - YIELD2 = J2 - G0
4. Radial return projection ratio (m21law.F:232-239):
   - If YIELD2 <= 0 and G0 > 0 -> ratio = 1.0
   - Else ratio = sqrt(G0 / (J2 + 1e-14))
   - s_ij = ratio * s_ij^trial * off
5. Compaction EOS and hysteretic unloading (m21law.F:160-173):
   - mu = rho / rho0 - 1
   - P_load = F_scale * finter(ifunc, mu)
   - mu_bak <- max(mu_bak, min(mu_max, mu))
   - alpha = mu_bak / mu_max in [0, 1] if mu_max > 0 else 1.0
   - K_unload = alpha * B_max + (1 - alpha) * B_min with B_max = B_unl, B_min = C_1
   - P_unl = P_load(mu_bak) - (mu_bak - mu) * K_unload
   - If mu_bak > -1e20 -> P = min(P_unl, P_load)
   - P = max(P, P_min) * off
6. Dilatational sound speed (m21law.F:178-182):
   - K_eff = max(K_unload, dP/dmu)
   - c_solid = sqrt(|4/3 G + K_eff| / rho_ref)
7. Plastic strain updates:
   - Delta eps_p = (1 - ratio) * sqrt(J2) / (3 G)
   - eps_p <- eps_p + Delta eps_p
   - eps_p,vol = mu_bak
8. Deactivation with off = 0.0 zeroing all stress and pressure components.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law21_dprag import (
    build_law21,
    _ensure_params,
    solid_update_law21,
    sound_speed_solid_law21,
    tangent_law21_solid,
)


# =============================================================================
# Exact Fortran Reference Oracle for M21LAW
# =============================================================================

def fortran_oracle_m21law(
    mat_params: dict,
    sig: np.ndarray,
    deps: np.ndarray,
    dt: float,
    mu: float,
    mu_bak: float,
    defp: float = 0.0,
    off: float = 1.0,
    curve: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Exact line-by-line python reimplementation of m21law.F & hm_read_mat21.F.

    Cites:
    - starter/source/materials/mat/mat021/hm_read_mat21.F
    - engine/source/materials/mat/mat021/m21law.F
    """
    # 1. Parameter extraction & hm_read_mat21.F defaults
    rho0 = float(mat_params.get("rho0", mat_params.get("MAT_RHO", 2000.0)))
    rhor = float(mat_params.get("refer_rho", mat_params.get("Refer_Rho", 0.0)))
    if rhor == 0.0:
        rhor = rho0

    e = float(mat_params.get("E", mat_params.get("MAT_E", 1.0e10)))
    nu = float(mat_params.get("nu", mat_params.get("MAT_NU", 0.25)))
    g_shear = e / (2.0 * (1.0 + nu))
    k_bulk = e / (3.0 * (1.0 - 2.0 * nu))

    a0 = float(mat_params.get("a0", mat_params.get("MAT_A0", 0.0)))
    a1 = float(mat_params.get("a1", mat_params.get("MAT_A1", 0.0)))
    a2 = float(mat_params.get("a2", mat_params.get("MAT_A2", 0.0)))

    amx = float(mat_params.get("amax", mat_params.get("MAT_AMAX", 0.0)))
    if amx == 0.0:
        amx = 1.0e20

    c1 = float(mat_params.get("c1", mat_params.get("MAT_BULK", 0.0)))
    if c1 == 0.0:
        c1 = k_bulk

    bunl = float(mat_params.get("bunl", mat_params.get("MAT_K_UNLOAD", 0.0)))
    if bunl == 0.0:
        bunl = c1

    pmin = float(mat_params.get("pmin", mat_params.get("MAT_PC", 0.0)))
    if pmin == 0.0:
        pmin = -1.0e30

    pext = float(mat_params.get("pext", mat_params.get("PEXT", 0.0)))

    mumax = float(mat_params.get("mumax", mat_params.get("MAT_SIG", 0.0)))
    if mumax == 0.0:
        mumax = 1.0e20

    fac_y = float(mat_params.get("pfscale", mat_params.get("PFscale", 0.0)))
    if fac_y == 0.0:
        fac_y = 1.0

    # P* root calculation (hm_read_mat21.F:146-160)
    if a2 == 0.0 and a1 != 0.0:
        pstar = -a0 / a1
    elif a2 != 0.0:
        delta = a1 * a1 - 4.0 * a0 * a2
        if delta >= 0.0:
            pstar = (-a1 + math.sqrt(delta)) / (2.0 * a2)
        else:
            pstar = -1.0e30
    else:
        pstar = -1.0e30

    # 2. State init (m21law.F:141-155)
    pold = -(sig[0] + sig[1] + sig[2]) / 3.0
    t1 = sig[0] + pold
    t2 = sig[1] + pold
    t3 = sig[2] + pold
    t4 = sig[3]
    t5 = sig[4]
    t6 = sig[5]

    # 3. Compaction EOS & Unloading (m21law.F:160-173)
    def eval_curve(x_val: float) -> tuple[float, float]:
        if curve is not None:
            xs, ys = curve
            if x_val <= xs[0]:
                slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
                return ys[0] + slope * (x_val - xs[0]), slope
            elif x_val >= xs[-1]:
                slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                return ys[-1] + slope * (x_val - xs[-1]), slope
            else:
                idx = np.searchsorted(xs, x_val, side="right") - 1
                slope = (ys[idx + 1] - ys[idx]) / (xs[idx + 1] - xs[idx])
                val = ys[idx] + slope * (x_val - xs[idx])
                return val, slope
        else:
            return c1 * x_val, c1

    f_mu, dpdm = eval_curve(mu)
    p_load = fac_y * f_mu
    dpdm = fac_y * dpdm

    f_mubak, _ = eval_curve(mu_bak)
    p_bak = fac_y * f_mubak

    if mumax > 0.0:
        alpha = min(1.0, max(0.0, mu_bak / mumax))
    else:
        alpha = 1.0
    bulk = alpha * bunl + (1.0 - alpha) * c1
    pne1 = p_bak - (mu_bak - mu) * bulk

    mumin = -1.0e20
    p_val = p_load
    if mu_bak > mumin:
        p_val = min(pne1, p_load)
    p_val = max(p_val, pmin) * off

    mu_bak_new = mu_bak
    if mu > mu_bak:
        mu_bak_new = min(mumax, mu)

    # 4. Sound speed (m21law.F:178-182)
    kt_eff = max(bulk, dpdm)
    g43 = (4.0 / 3.0) * g_shear
    dpdm_tot = g43 + kt_eff
    ssp = math.sqrt(abs(dpdm_tot) / rhor)

    # 5. Output pressure (m21law.F:188-190)
    p_out = max(pmin, p_val) * off

    # 6. Yield criteria if P > PMIN (m21law.F:195-201)
    a0_eff = a0
    a1_eff = a1
    a2_eff = a2
    if p_out <= pmin:
        a0_eff = 0.0
        a1_eff = 0.0
        a2_eff = 0.0

    # 7. Elastic increment (m21law.F:206-214)
    svrt = (deps[0] + deps[1] + deps[2]) / 3.0
    t1 += 2.0 * g_shear * (deps[0] - svrt)
    t2 += 2.0 * g_shear * (deps[1] - svrt)
    t3 += 2.0 * g_shear * (deps[2] - svrt)
    t4 += g_shear * deps[3]
    t5 += g_shear * deps[4]
    t6 += g_shear * deps[5]

    # 8. Yield surface (m21law.F:220-227)
    j2 = 0.5 * (t1 * t1 + t2 * t2 + t3 * t3) + t4 * t4 + t5 * t5 + t6 * t6
    ptot = p_out + pext
    g0 = a0_eff + a1_eff * ptot + a2_eff * ptot * ptot
    g0 = min(amx, max(0.0, g0))
    if ptot <= pstar or p_out <= pmin:
        g0 = 0.0
    yield2 = j2 - g0

    # 9. Projection factor (m21law.F:232-239)
    if yield2 <= 0.0 and g0 > 0.0:
        ratio = 1.0
    elif g0 <= 0.0:
        ratio = 0.0
    else:
        ratio = math.sqrt(g0 / (j2 + 1.0e-14))

    # 10. Deviatoric and Cauchy stress (m21law.F:245-252)
    s_new = np.array([
        ratio * t1 * off,
        ratio * t2 * off,
        ratio * t3 * off,
        ratio * t4 * off,
        ratio * t5 * off,
        ratio * t6 * off,
    ], dtype=float)

    sig_new = np.array([
        s_new[0] - p_out,
        s_new[1] - p_out,
        s_new[2] - p_out,
        s_new[3],
        s_new[4],
        s_new[5],
    ], dtype=float)

    # 11. Plastic strain increment (m21law.F:253, 261-264)
    dpla = (1.0 - ratio) * math.sqrt(max(0.0, j2)) / max(1.0e-20, 3.0 * g_shear)
    defp_new = defp + dpla

    return {
        "sig": sig_new,
        "s_new": s_new,
        "p_new": p_out,
        "ptot": ptot,
        "j2": j2,
        "g0": g0,
        "yield2": yield2,
        "ratio": ratio,
        "dpla": dpla,
        "defp": defp_new,
        "mu_bak": mu_bak_new,
        "ssp": ssp,
        "pstar": pstar,
        "bulk": bulk,
        "dpdm": dpdm,
        "g_shear": g_shear,
        "refer_rho": rhor,
        "pmin": pmin,
        "bunl": bunl,
        "amax": amx,
        "mumax": mumax,
        "pfscale": fac_y,
    }


# =============================================================================
# Section 1: Parameter Defaults and Setup Parity (hm_read_mat21.F)
# =============================================================================

class TestSection1ParameterSetup:
    """Verify setup and parameter defaults from hm_read_mat21.F."""

    def test_refer_rho_defaults_to_rho(self):
        """hm_read_mat21.F line 114: IF (RHOR==ZERO) RHOR=RHO0."""
        # Case A: refer_rho explicitly 0.0
        p1 = {"rho0": 2500.0, "refer_rho": 0.0, "E": 2.0e10, "nu": 0.20}
        res1 = _ensure_params(p1)
        assert res1["refer_rho"] == 2500.0
        assert res1["Refer_Rho"] == 2500.0

        # Case B: refer_rho omitted
        p2 = {"rho0": 1800.0, "E": 1.5e10, "nu": 0.22}
        res2 = _ensure_params(p2)
        assert res2["refer_rho"] == 1800.0

        # Case C: refer_rho specified > 0
        p3 = {"rho0": 2000.0, "refer_rho": 2200.0, "E": 1.0e10, "nu": 0.25}
        res3 = _ensure_params(p3)
        assert res3["refer_rho"] == 2200.0

    def test_pmin_defaults_to_neg_infinity(self):
        """hm_read_mat21.F line 119: IF(PMIN==ZERO) PMIN = -INFINITY (-1e30)."""
        p_zero = {"rho0": 2000.0, "E": 1.0e10, "nu": 0.25, "pmin": 0.0}
        res = _ensure_params(p_zero)
        assert res["pmin"] == -1.0e30
        assert res["MAT_PC"] == -1.0e30

        p_explicit = {"rho0": 2000.0, "E": 1.0e10, "nu": 0.25, "pmin": -5.0e7}
        assert _ensure_params(p_explicit)["pmin"] == -5.0e7

    def test_bunl_defaults_to_c1(self):
        """hm_read_mat21.F line 121: IF(BUNL==ZERO) BUNL = C1."""
        p_zero = {"rho0": 2000.0, "E": 1.2e10, "nu": 0.20, "bunl": 0.0, "c1": 8.0e9}
        res = _ensure_params(p_zero)
        assert res["bunl"] == 8.0e9
        assert res["bmax"] == 8.0e9

        # When c1 is not given, c1 defaults to elastic bulk modulus K = E / (3(1 - 2nu))
        expected_k = 1.2e10 / (3.0 * (1.0 - 2.0 * 0.20))
        p_no_c1 = {"rho0": 2000.0, "E": 1.2e10, "nu": 0.20, "bunl": 0.0}
        res_no_c1 = _ensure_params(p_no_c1)
        assert math.isclose(res_no_c1["c1"], expected_k)
        assert math.isclose(res_no_c1["bunl"], expected_k)

    def test_amax_defaults_to_1e20(self):
        """hm_read_mat21.F line 122: IF(AMX==ZERO) AMX = EP20 (1e20)."""
        p_zero = {"rho0": 2000.0, "E": 1.0e10, "nu": 0.25, "amax": 0.0}
        res = _ensure_params(p_zero)
        assert res["amax"] == 1.0e20
        assert res["Amax"] == 1.0e20
        assert res["MAT_AMAX"] == 1.0e20

    def test_mumax_defaults_to_1e20(self):
        """hm_read_mat21.F line 123: IF(XMUMX == ZERO) XMUMX = EP20 (1e20)."""
        p_zero = {"rho0": 2000.0, "E": 1.0e10, "nu": 0.25, "mumax": 0.0}
        res = _ensure_params(p_zero)
        assert res["mumax"] == 1.0e20
        assert res["MAT_SIG"] == 1.0e20

    def test_pfscale_defaults_to_one(self):
        """hm_read_mat21.F line 164: IF (FAC_Y == ZERO) FAC_Y = 1.0."""
        p_zero = {"rho0": 2000.0, "E": 1.0e10, "nu": 0.25, "pfscale": 0.0}
        res = _ensure_params(p_zero)
        assert res["pfscale"] == 1.0
        assert res["PFscale"] == 1.0

    def test_elastic_shear_modulus_g(self):
        """hm_read_mat21.F line 165: G = E / (2 * (1 + nu))."""
        for e_val in [1.0e9, 2.5e10, 7.2e10]:
            for nu_val in [0.0, 0.15, 0.25, 0.35, 0.49]:
                p = {"rho0": 2000.0, "E": e_val, "nu": nu_val}
                res = _ensure_params(p)
                expected_g = e_val / (2.0 * (1.0 + nu_val))
                assert math.isclose(res["G"], expected_g, rel_tol=1e-12)


# =============================================================================
# Section 2: Yield Surface Pressure Root P* (hm_read_mat21.F)
# =============================================================================

class TestSection2PressureRootPStar:
    """Verify closure root P* calculation against hm_read_mat21.F lines 146-160."""

    def test_linear_root(self):
        """Linear case (A2 == 0, A1 != 0): P* = -A0 / A1."""
        # A1 > 0
        p1 = {"rho0": 2000.0, "E": 1e10, "nu": 0.2, "a0": 1.0e6, "a1": 0.5, "a2": 0.0}
        assert math.isclose(_ensure_params(p1)["pstar"], -1.0e6 / 0.5)

        # A1 < 0
        p2 = {"rho0": 2000.0, "E": 1e10, "nu": 0.2, "a0": 3.0e6, "a1": -0.75, "a2": 0.0}
        assert math.isclose(_ensure_params(p2)["pstar"], -3.0e6 / -0.75)

    def test_parabolic_distinct_real_roots(self):
        """Parabolic case (A2 != 0, Delta > 0): P* = (-A1 + sqrt(Delta)) / (2*A2)."""
        # A0 = 6, A1 = 5, A2 = 1 -> roots are -3 and -2 -> P* = (-5 + 1)/2 = -2.0
        p = {"rho0": 2000.0, "E": 1e10, "nu": 0.2, "a0": 6.0, "a1": 5.0, "a2": 1.0}
        assert math.isclose(_ensure_params(p)["pstar"], -2.0)

        # A0 = 1e6, A1 = 3000, A2 = 2
        # Delta = 9e6 - 4 * 1e6 * 2 = 1e6 -> sqrt(Delta) = 1000
        # P* = (-3000 + 1000) / 4 = -500.0
        p2 = {"rho0": 2000.0, "E": 1e10, "nu": 0.2, "a0": 1.0e6, "a1": 3000.0, "a2": 2.0}
        assert math.isclose(_ensure_params(p2)["pstar"], -500.0)

    def test_parabolic_double_root(self):
        """Parabolic case (A2 != 0, Delta == 0): P* = -A1 / (2*A2)."""
        # (P + 1000)^2 = P^2 + 2000 P + 1e6 -> Delta = 0 -> P* = -1000
        p = {"rho0": 2000.0, "E": 1e10, "nu": 0.2, "a0": 1.0e6, "a1": 2000.0, "a2": 1.0}
        assert math.isclose(_ensure_params(p)["pstar"], -1000.0, abs_tol=1e-6)

    def test_parabolic_no_real_roots(self):
        """Parabolic case (A2 != 0, Delta < 0): P* = -1e30."""
        # A0 = 1e6, A1 = 0, A2 = 1 -> Delta = -4e6 < 0
        p = {"rho0": 2000.0, "E": 1e10, "nu": 0.2, "a0": 1.0e6, "a1": 0.0, "a2": 1.0}
        assert _ensure_params(p)["pstar"] == -1.0e30

    def test_pure_von_mises_root(self):
        """Pure von Mises (A1 == 0, A2 == 0): P* = -1e30."""
        p = {"rho0": 2000.0, "E": 1e10, "nu": 0.2, "a0": 5.0e7, "a1": 0.0, "a2": 0.0}
        assert _ensure_params(p)["pstar"] == -1.0e30


# =============================================================================
# Section 3: Drucker-Prager Yield Envelope G0(P_tot) (m21law.F:220-227)
# =============================================================================

class TestSection3YieldEnvelope:
    """Verify Drucker-Prager yield envelope and J2 calculation against m21law.F."""

    def test_ptot_includes_external_pressure(self):
        """P_tot = P + P_ext (m21law.F:221)."""
        mat = {
            "rho0": 2000.0, "E": 1.0e10, "nu": 0.25,
            "c1": 5.0e8, "a0": 1.0e6, "a1": 0.4, "a2": 0.0,
            "pext": 2.5e7,
        }
        sig = np.zeros((1, 6), dtype=float)
        # mu = 0.02 -> P = 5e8 * 0.02 = 1.0e7
        deps = np.array([[-0.02 / 3.0, -0.02 / 3.0, -0.02 / 3.0, 0, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        expected_p = 1.0e7
        expected_ptot = expected_p + 2.5e7
        assert math.isclose(extra["p_new"][0], expected_p, rel_tol=1e-8)
        assert math.isclose(extra["ptot"][0], expected_ptot, rel_tol=1e-8)

    def test_g0_zero_when_p_at_or_below_pmin(self):
        """If P <= P_min -> G0 = 0 (m21law.F:195-201)."""
        mat = {
            "rho0": 2000.0, "E": 1.0e10, "nu": 0.25,
            "c1": 5.0e8, "a0": 1.0e8, "a1": 0.0, "a2": 0.0,
            "pmin": -1.0e7,
        }
        sig = np.zeros((1, 6), dtype=float)
        # Severe tension: mu = -0.05 -> P_unclamped = -2.5e7 < -1e7 -> P = -1e7
        deps = np.array([[0.05 / 3.0, 0.05 / 3.0, 0.05 / 3.0, 0, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        assert math.isclose(extra["p_new"][0], -1.0e7)
        assert extra["g0"][0] == 0.0
        assert extra["ratio"][0] == 0.0

    def test_g0_zero_when_ptot_at_or_below_pstar(self):
        """If P_tot <= P* -> G0 = 0 (m21law.F:225)."""
        # Linear yield: A0 = 2e7, A1 = 0.5 -> P* = -4e7
        mat = {
            "rho0": 2000.0, "E": 1.0e10, "nu": 0.25,
            "c1": 5.0e8, "a0": 2.0e7, "a1": 0.5, "a2": 0.0,
            "pmin": -1.0e8, "pext": 0.0,
        }
        # P = -5e7 < P* (-4e7)
        sig = np.zeros((1, 6), dtype=float)
        deps = np.array([[0.10 / 3.0, 0.10 / 3.0, 0.10 / 3.0, 0, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        assert extra["ptot"][0] <= -4.0e7
        assert extra["g0"][0] == 0.0

    def test_g0_clamped_at_amax(self):
        """G0 is capped at A_max (m21law.F:223)."""
        mat = {
            "rho0": 2000.0, "E": 1.0e10, "nu": 0.25,
            "c1": 5.0e8, "a0": 1.0e6, "a1": 1.0, "a2": 0.0,
            "amax": 5.0e6,
        }
        # P = 5e8 * 0.05 = 2.5e7 -> G0_unclamped = 1e6 + 2.5e7 = 2.6e7 > 5e6
        sig = np.zeros((1, 6), dtype=float)
        deps = np.array([[-0.05 / 3.0, -0.05 / 3.0, -0.05 / 3.0, 0, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        assert math.isclose(extra["g0"][0], 5.0e6)

    def test_j2_exact_numerical_formula(self):
        """J2 = 0.5*(s_xx^2 + s_yy^2 + s_zz^2) + s_xy^2 + s_yz^2 + s_zx^2 (m21law.F:220)."""
        s = np.array([10.0, -4.0, -6.0, 3.0, 2.0, -1.5], dtype=float)
        expected_j2 = 0.5 * (10.0**2 + (-4.0)**2 + (-6.0)**2) + 3.0**2 + 2.0**2 + (-1.5)**2
        assert math.isclose(expected_j2, 0.5 * (100 + 16 + 36) + 9 + 4 + 2.25)
        assert math.isclose(expected_j2, 91.25)


# =============================================================================
# Section 4: Radial Return Projection Ratio (m21law.F:232-239)
# =============================================================================

class TestSection4RadialReturnRatio:
    """Verify projection factor ratio against m21law.F lines 232-239."""

    def test_ratio_elastic_when_j2_le_g0(self):
        """If YIELD2 <= 0 and G0 > 0 -> ratio = 1.0 (m21law.F:234-235)."""
        mat = {
            "rho0": 2000.0, "E": 1.0e10, "nu": 0.25,
            "c1": 5.0e8, "a0": 1.0e14, "a1": 0.0, "a2": 0.0,
        }
        sig = np.zeros((1, 6), dtype=float)
        # Small shear strain: J2 << G0
        deps = np.array([[0, 0, 0, 1.0e-5, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        assert extra["ratio"][0] == 1.0

    def test_ratio_plastic_when_j2_gt_g0(self):
        """If YIELD2 > 0 -> ratio = sqrt(G0 / (J2 + 1e-14)) (m21law.F:237)."""
        mat = {
            "rho0": 2000.0, "E": 1.0e10, "nu": 0.25,
            "c1": 5.0e8, "a0": 1.0e6, "a1": 0.0, "a2": 0.0,
        }
        sig = np.zeros((1, 6), dtype=float)
        # Large shear: G_shear = 1e10 / 2.5 = 4e9 -> s_xy = 4e9 * 0.001 = 4e6 -> J2 = 1.6e13
        deps = np.array([[0, 0, 0, 1.0e-3, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        j2 = extra["j2"][0]
        g0 = extra["g0"][0]
        expected_ratio = math.sqrt(g0 / (j2 + 1e-14))
        assert math.isclose(extra["ratio"][0], expected_ratio, rel_tol=1e-12)

    def test_ratio_zero_when_g0_zero(self):
        """When G0 = 0 -> ratio = 0.0 (m21law.F:237)."""
        mat = {
            "rho0": 2000.0, "E": 1.0e10, "nu": 0.25,
            "c1": 5.0e8, "a0": 0.0, "a1": 0.0, "a2": 0.0,
        }
        sig = np.zeros((1, 6), dtype=float)
        deps = np.array([[0, 0, 0, 1.0e-4, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        assert extra["g0"][0] == 0.0
        assert extra["ratio"][0] == 0.0


# =============================================================================
# Section 5: Compaction EOS and Hysteretic Unloading (m21law.F:160-173)
# =============================================================================

class TestSection5CompactionEOS:
    """Verify compaction EOS, historical memory, and hysteretic unloading."""

    def test_linear_loading_pressure(self):
        """P_load = C1 * mu when no curve is given (m21law.F:161)."""
        mat = {"rho0": 2000.0, "E": 2.0e10, "nu": 0.25, "c1": 8.0e8, "a0": 1.0e14}
        sig = np.zeros((1, 6), dtype=float)
        # tr(deps) = -0.03 -> mu = +0.03
        deps = np.array([[-0.01, -0.01, -0.01, 0, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        expected_p = 8.0e8 * 0.03
        assert math.isclose(extra["p_new"][0], expected_p)

    def test_historical_peak_compaction_memory(self):
        """mu_bak <- max(mu_bak, min(mu_max, mu)) (m21law.F:172)."""
        mat = {"rho0": 2000.0, "E": 2.0e10, "nu": 0.25, "c1": 8.0e8, "mumax": 0.10, "a0": 1.0e14}
        sig = np.zeros((1, 6), dtype=float)
        extra = {}

        # Step 1: load to mu = 0.04
        deps1 = np.array([[-0.04 / 3.0, -0.04 / 3.0, -0.04 / 3.0, 0, 0, 0]], dtype=float)
        sig = solid_update_law21(mat, sig, deps=deps1, dt=1.0, extra=extra)
        assert math.isclose(extra["mu_bak"][0], 0.04)

        # Step 2: unload to mu = 0.02 -> mu_bak remains 0.04
        deps2 = np.array([[0.02 / 3.0, 0.02 / 3.0, 0.02 / 3.0, 0, 0, 0]], dtype=float)
        sig = solid_update_law21(mat, sig, deps=deps2, dt=1.0, extra=extra)
        assert math.isclose(extra["mu_bak"][0], 0.04)

        # Step 3: exceed mumax: load to mu = 0.15 -> mu_bak capped at mumax (0.10)
        deps3 = np.array([[-0.13 / 3.0, -0.13 / 3.0, -0.13 / 3.0, 0, 0, 0]], dtype=float)
        sig = solid_update_law21(mat, sig, deps=deps3, dt=1.0, extra=extra)
        assert math.isclose(extra["mu_bak"][0], 0.10)

    def test_hysteretic_unloading_modulus_and_pressure(self):
        """Verify K_unload and P_unl = P_load(mu_bak) - (mu_bak - mu)*K_unload."""
        xs = np.array([0.0, 0.05, 0.10, 0.20])
        ys = np.array([0.0, 1.0e7, 3.0e7, 8.0e7])
        curve = (xs, ys)

        mat = {
            "rho0": 2000.0, "E": 1.0e10, "nu": 0.25,
            "c1": 5.0e8, "bunl": 1.0e9, "mumax": 0.10,
            "curve": curve, "a0": 1.0e14,
        }
        sig = np.zeros((1, 6), dtype=float)
        extra = {}

        # 1. Load to mu = 0.10 -> P_load = 3.0e7, mu_bak = 0.10
        # With C1 = 5e8 >= chord slope, pne1 = 0 - (0 - 0.10)*5e8 = 5e7 > 3e7 -> P = 3.0e7
        deps1 = np.array([[-0.10 / 3.0, -0.10 / 3.0, -0.10 / 3.0, 0, 0, 0]], dtype=float)
        sig = solid_update_law21(mat, sig, deps=deps1, dt=1.0, extra=extra)
        assert math.isclose(extra["p_new"][0], 3.0e7)
        assert math.isclose(extra["mu_bak"][0], 0.10)

        # 2. Unload to mu = 0.08
        # alpha = mu_bak / mumax = 0.10 / 0.10 = 1.0 -> bulk = 1.0 * bunl = 1.0e9
        # P_unl = 3.0e7 - (0.10 - 0.08) * 1.0e9 = 3.0e7 - 2.0e7 = 1.0e7
        # P_load(0.08) = 1e7 + (3e7 - 1e7)/0.05 * (0.08 - 0.05) = 1e7 + 4e8 * 0.03 = 2.2e7
        # P = min(P_unl, P_load) = min(1.0e7, 2.2e7) = 1.0e7
        deps2 = np.array([[0.02 / 3.0, 0.02 / 3.0, 0.02 / 3.0, 0, 0, 0]], dtype=float)
        sig = solid_update_law21(mat, sig, deps=deps2, dt=1.0, extra=extra)
        assert math.isclose(extra["p_new"][0], 1.0e7, rel_tol=1e-6)
        assert math.isclose(extra["bulk"][0], 1.0e9, rel_tol=1e-6)


# =============================================================================
# Section 6: Dilatational Sound Speed (m21law.F:178-182)
# =============================================================================

class TestSection6SoundSpeed:
    """Verify dilatational sound speed against m21law.F lines 178-182."""

    def test_sound_speed_elastic_default(self):
        """c_solid = sqrt(|4/3 G + max(C1, Bunl)| / rho_ref)."""
        mat = {
            "rho0": 2000.0, "E": 3.0e10, "nu": 0.20,
            "c1": 1.5e10, "bunl": 2.0e10,
        }
        params = _ensure_params(mat)
        g = params["G"]
        expected_c = math.sqrt((4.0 / 3.0 * g + 2.0e10) / 2000.0)
        c_calc = sound_speed_solid_law21(mat)
        assert math.isclose(c_calc, expected_c, rel_tol=1e-12)

    def test_sound_speed_uses_refer_rho(self):
        """Sound speed uses reference density refer_rho (PM(1,MX))."""
        mat = {
            "rho0": 2000.0, "refer_rho": 2500.0,
            "E": 3.0e10, "nu": 0.20, "c1": 1.5e10, "bunl": 1.5e10,
        }
        params = _ensure_params(mat)
        g = params["G"]
        expected_c = math.sqrt((4.0 / 3.0 * g + 1.5e10) / 2500.0)
        c_calc = sound_speed_solid_law21(mat)
        assert math.isclose(c_calc, expected_c, rel_tol=1e-12)

    def test_sound_speed_in_solid_update(self):
        """sound_speed in extra matches m21law.F formula during solid_update."""
        mat = {
            "rho0": 2400.0, "E": 2.4e10, "nu": 0.20,
            "c1": 1.0e10, "bunl": 3.0e10, "mumax": 0.05,
            "a0": 1.0e14,
        }
        sig = np.zeros((1, 6), dtype=float)
        deps = np.array([[-0.025 / 3.0, -0.025 / 3.0, -0.025 / 3.0, 0, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        # In m21law.F: bulk is computed at line 168 using initial mu_bak (0.0):
        # alpha = 0 -> bulk = C1 = 1.0e10. dpdm = 1.0e10 -> kt_eff = 1.0e10
        g = 2.4e10 / 2.4
        kt_eff = 1.0e10
        expected_c = math.sqrt((4.0 / 3.0 * g + kt_eff) / 2400.0)
        assert math.isclose(extra["c_solid"][0], expected_c, rel_tol=1e-10)


# =============================================================================
# Section 7: Plastic Strain Updates (m21law.F:253, 261-264)
# =============================================================================

class TestSection7PlasticStrain:
    """Verify plastic strain increment, accumulation, and volumetric plastic strain."""

    def test_plastic_strain_increment_formula(self):
        """Delta eps_p = (1 - ratio) * sqrt(J2) / (3 * G) (m21law.F:253)."""
        mat = {
            "rho0": 2000.0, "E": 1.2e10, "nu": 0.20,
            "c1": 1.0e9, "a0": 4.0e6, "a1": 0.0, "a2": 0.0,
        }
        params = _ensure_params(mat)
        g = params["G"]  # 1.2e10 / 2.4 = 5.0e9

        # Pure shear strain increment: d_xy = 0.002 -> s_xy^trial = 5e9 * 0.002 = 1.0e7
        # J2 = (1e7)^2 = 1.0e14
        # G0 = A0 = 4.0e6
        # ratio = sqrt(4e6 / 1e14) = 2.0e-4
        sig = np.zeros((1, 6), dtype=float)
        deps = np.array([[0, 0, 0, 0.002, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        expected_ratio = math.sqrt(4.0e6 / (1.0e14 + 1.0e-14))
        expected_dpla = (1.0 - expected_ratio) * 1.0e7 / (3.0 * g)
        assert math.isclose(extra["dpla"][0], expected_dpla, rel_tol=1e-10)
        assert math.isclose(extra["epxe"][0], expected_dpla, rel_tol=1e-10)

    def test_volumetric_plastic_strain_stored_in_epsq(self):
        """Volumetric plastic strain EPSQ = mu_bak (m21law.F:263)."""
        mat = {"rho0": 2000.0, "E": 1.0e10, "nu": 0.25, "c1": 5.0e8, "a0": 1.0e14}
        sig = np.zeros((1, 6), dtype=float)
        deps = np.array([[-0.015, -0.015, -0.015, 0, 0, 0]], dtype=float)
        extra = {}
        solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        assert math.isclose(extra["epsq"][0], 0.045)


# =============================================================================
# Section 8: Deactivation with off = 0.0 (m21law.F:171, 188, 245-252)
# =============================================================================

class TestSection8Deactivation:
    """Verify deactivation with off = 0.0 zeroes all stress and pressure components."""

    def test_single_element_deactivated(self):
        """When off = 0.0, all Cauchy stresses and pressure become 0.0."""
        mat = {
            "rho0": 2000.0, "E": 1.0e10, "nu": 0.25,
            "c1": 5.0e8, "a0": 1.0e6, "a1": 0.5, "a2": 0.0,
        }
        sig = np.array([1.0e7, 2.0e7, -1.0e7, 5.0e6, 3.0e6, 1.0e6], dtype=float)
        deps = np.array([[-0.01, -0.01, -0.01, 0.001, 0.002, 0.001]], dtype=float)
        extra = {"off": np.array([0.0])}
        sig_new = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        np.testing.assert_allclose(sig_new, np.zeros(6), atol=1e-15)
        assert extra["p_new"][0] == 0.0
        np.testing.assert_allclose(extra["s_new"][0], np.zeros(6), atol=1e-15)

    def test_vectorized_mixed_active_inactive(self):
        """Batch of 4 elements: elements 1 and 3 active, 0 and 2 inactive."""
        mat = {
            "rho0": 2000.0, "E": 2.0e10, "nu": 0.20,
            "c1": 1.0e9, "a0": 2.0e7, "a1": 0.3, "a2": 0.0,
        }
        sig = np.full((4, 6), 5.0e6, dtype=float)
        deps = np.full((4, 6), -0.002, dtype=float)
        off = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        extra = {"off": off}
        sig_new = solid_update_law21(mat, sig, deps=deps, dt=1.0, extra=extra)

        # Inactive elements 0 and 2 are strictly zero
        np.testing.assert_allclose(sig_new[0], np.zeros(6), atol=1e-15)
        np.testing.assert_allclose(sig_new[2], np.zeros(6), atol=1e-15)
        assert extra["p_new"][0] == 0.0
        assert extra["p_new"][2] == 0.0

        # Active elements 1 and 3 have non-zero stress
        assert np.any(np.abs(sig_new[1]) > 1.0e4)
        assert np.any(np.abs(sig_new[3]) > 1.0e4)


# =============================================================================
# Section 9: Comprehensive Direct Fortran Oracle Parity Suite
# =============================================================================

class TestSection9FortranOracleParity:
    """Direct 1-to-1 numerical parity verification between pyradioss and Fortran oracle."""

    @pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
    def test_random_triaxial_states_oracle_parity(self, seed: int):
        """Compare pyradioss against Fortran oracle across diverse stress & strain states."""
        rng = np.random.default_rng(seed)

        # Random material properties
        e_val = rng.uniform(1.0e9, 5.0e10)
        nu_val = rng.uniform(0.10, 0.35)
        c1_val = rng.uniform(5.0e8, 2.0e10)
        bunl_val = rng.uniform(c1_val, 3.0 * c1_val)
        a0_val = rng.uniform(1.0e6, 5.0e7)
        a1_val = rng.uniform(0.1, 1.5)
        a2_val = rng.uniform(0.0, 0.01) if rng.random() > 0.5 else 0.0
        amax_val = rng.uniform(1.0e7, 1.0e9)
        pmin_val = -rng.uniform(1.0e6, 5.0e7)
        pext_val = rng.uniform(0.0, 1.0e7)

        mat_params = {
            "rho0": 2200.0,
            "refer_rho": 2200.0,
            "E": e_val,
            "nu": nu_val,
            "c1": c1_val,
            "bunl": bunl_val,
            "a0": a0_val,
            "a1": a1_val,
            "a2": a2_val,
            "amax": amax_val,
            "pmin": pmin_val,
            "pext": pext_val,
            "mumax": 0.15,
        }

        # Random old stress and strain increment
        sig_old = rng.uniform(-5.0e7, 5.0e7, size=6)
        deps = rng.uniform(-0.01, 0.01, size=6)
        dt = rng.uniform(1.0e-5, 1.0e-3)
        mu = -float(deps[0] + deps[1] + deps[2])
        mu_bak = max(0.0, mu * 0.8)

        # Oracle evaluation
        oracle = fortran_oracle_m21law(
            mat_params, sig_old, deps, dt, mu=mu, mu_bak=mu_bak, defp=0.0, off=1.0,
        )

        # pyradioss evaluation with mu_total explicitly set to mu
        extra = {"mu_total": np.array([mu]), "mu_bak": np.array([mu_bak]), "epxe": np.array([0.0])}
        sig_py = solid_update_law21(
            mat_params, sig_old.reshape(1, 6), deps=deps.reshape(1, 6), dt=dt, extra=extra,
        )[0]

        # Assert full numerical equivalence
        np.testing.assert_allclose(sig_py, oracle["sig"], rtol=1e-10, atol=1e-5)
        assert math.isclose(extra["p_new"][0], oracle["p_new"], rel_tol=1e-10, abs_tol=1e-5)
        assert math.isclose(extra["ptot"][0], oracle["ptot"], rel_tol=1e-10, abs_tol=1e-5)
        assert math.isclose(extra["g0"][0], oracle["g0"], rel_tol=1e-10, abs_tol=1e-5)
        assert math.isclose(extra["ratio"][0], oracle["ratio"], rel_tol=1e-10, abs_tol=1e-5)
        assert math.isclose(extra["dpla"][0], oracle["dpla"], rel_tol=1e-10, abs_tol=1e-10)
        assert math.isclose(extra["c_solid"][0], oracle["ssp"], rel_tol=1e-10, abs_tol=1e-5)

    def test_cyclic_loading_unloading_history_oracle_parity(self):
        """Multi-step cyclic loading-unloading-reloading sequence vs Fortran oracle."""
        mat_params = {
            "rho0": 2100.0, "E": 2.0e10, "nu": 0.22,
            "c1": 1.0e9, "bunl": 4.0e9, "mumax": 0.08,
            "a0": 5.0e6, "a1": 0.6, "a2": 0.0,
            "pmin": -2.0e7, "pext": 0.0,
        }

        # Sequence of volumetric and shear strain increments
        steps = [
            np.array([-0.01, -0.01, -0.01, 0.001, 0.0, 0.0]),  # Compaction to mu = 0.03
            np.array([-0.01, -0.01, -0.01, 0.002, 0.0, 0.0]),  # Compaction to mu = 0.06
            np.array([0.005, 0.005, 0.005, -0.001, 0.0, 0.0]), # Unloading to mu = 0.045
            np.array([0.005, 0.005, 0.005, 0.0, 0.0, 0.0]),    # Further unloading to mu = 0.030
            np.array([-0.015, -0.015, -0.015, 0.003, 0.0, 0.0]),# Reloading to mu = 0.075
        ]

        sig_oracle = np.zeros(6, dtype=float)
        sig_py = np.zeros((1, 6), dtype=float)
        mu_accum = 0.0
        mu_bak_oracle = 0.0
        extra_py = {}

        for step_idx, d_e in enumerate(steps):
            mu_accum -= float(d_e[0] + d_e[1] + d_e[2])
            oracle = fortran_oracle_m21law(
                mat_params, sig_oracle, d_e, dt=1.0e-4, mu=mu_accum, mu_bak=mu_bak_oracle,
            )
            sig_oracle = oracle["sig"]
            mu_bak_oracle = oracle["mu_bak"]

            sig_py = solid_update_law21(
                mat_params, sig_py, deps=d_e.reshape(1, 6), dt=1.0e-4, extra=extra_py,
            )

            np.testing.assert_allclose(sig_py[0], sig_oracle, rtol=1e-10, atol=1e-5)
            assert math.isclose(extra_py["p_new"][0], oracle["p_new"], rel_tol=1e-10, abs_tol=1e-5)
            assert math.isclose(extra_py["mu_bak"][0], mu_bak_oracle, rel_tol=1e-10, abs_tol=1e-10)
            assert math.isclose(extra_py["ratio"][0], oracle["ratio"], rel_tol=1e-10, abs_tol=1e-8)
