# pyradioss — OpenRadioss Python port
# Milestone M554: Consistent Tangent Verification Suite for /MAT/LAW52 (/MAT/GURSON)
# Rigorous verification of algorithmic consistent tangent stiffness tensors against
# numerical central finite differences across all physical regimes, loading states,
# vectorization widths, and calling conventions.

import copy
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pytest

from pyradioss.materials.law52_gurson import (
    Law52Params,
    solid_update_law52,
    shell_update_law52,
    tangent_law52_solid,
    tangent_law52_shell,
    consistent_solid_tangent,
    consistent_shell_tangent,
    solid_tangent,
    shell_tangent,
    shell_membrane_tangent,
    _copy_extra,
)


def _compute_num_solid_tangent(
    p: Law52Params,
    sig: Optional[np.ndarray],
    deps: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
    dt: float = 0.0,
    epsp: Optional[np.ndarray] = None,
    h: float = 1e-7,
) -> np.ndarray:
    deps_arr = np.asarray(deps, dtype=float)
    single = (deps_arr.ndim == 1)
    if single:
        deps_arr = deps_arr.reshape(1, 6)
    nel = deps_arr.shape[0]

    if sig is not None:
        s0 = np.asarray(sig, dtype=float).copy()
        if s0.ndim == 1:
            s0 = s0.reshape(1, 6)
        if s0.shape[0] == 1 and nel > 1:
            s0 = np.repeat(s0, nel, axis=0)
    else:
        s0 = np.zeros((nel, 6), dtype=float)

    if epsp is not None:
        ep_arr = np.asarray(epsp, dtype=float).flatten()
        if len(ep_arr) == 1 and nel > 1:
            ep_arr = np.full(nel, ep_arr[0])
    else:
        ep_arr = np.zeros(nel, dtype=float)

    D_num = np.zeros((nel, 6, 6), dtype=float)
    for j in range(6):
        ej = np.zeros((nel, 6), dtype=float)
        ej[:, j] = h
        ex_p = _copy_extra(extra)
        ex_m = _copy_extra(extra)
        sp, _ = solid_update_law52(p, s0.copy(), deps_arr + ej, extra=ex_p, dt=dt, epsp=ep_arr.copy())
        sm, _ = solid_update_law52(p, s0.copy(), deps_arr - ej, extra=ex_m, dt=dt, epsp=ep_arr.copy())
        if sp.ndim == 1:
            sp = sp.reshape(1, 6)
            sm = sm.reshape(1, 6)
        D_num[:, :, j] = (sp - sm) / (2.0 * h)

    return D_num[0] if single else D_num


def _compute_num_shell_tangent(
    p: Law52Params,
    sig: Optional[np.ndarray],
    deps: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
    dt: float = 0.0,
    epsp: Optional[np.ndarray] = None,
    h: float = 1e-7,
) -> np.ndarray:
    deps_arr = np.asarray(deps, dtype=float)
    single = (deps_arr.ndim == 1)
    if single:
        deps_arr = deps_arr.reshape(1, 3)
    nel = deps_arr.shape[0]

    if sig is not None:
        s0 = np.asarray(sig, dtype=float).copy()
        if s0.ndim == 1:
            s0 = s0.reshape(1, 3)
        if s0.shape[0] == 1 and nel > 1:
            s0 = np.repeat(s0, nel, axis=0)
    else:
        s0 = np.zeros((nel, 3), dtype=float)

    if epsp is not None:
        ep_arr = np.asarray(epsp, dtype=float).flatten()
        if len(ep_arr) == 1 and nel > 1:
            ep_arr = np.full(nel, ep_arr[0])
    else:
        ep_arr = np.zeros(nel, dtype=float)

    C_num = np.zeros((nel, 3, 3), dtype=float)
    for j in range(3):
        ej = np.zeros((nel, 3), dtype=float)
        ej[:, j] = h
        ex_p = _copy_extra(extra)
        ex_m = _copy_extra(extra)
        sp, _ = shell_update_law52(p, s0.copy(), deps_arr + ej, extra=ex_p, dt=dt, epsp=ep_arr.copy())
        sm, _ = shell_update_law52(p, s0.copy(), deps_arr - ej, extra=ex_m, dt=dt, epsp=ep_arr.copy())
        if sp.ndim == 1:
            sp = sp.reshape(1, 3)
            sm = sm.reshape(1, 3)
        C_num[:, :, j] = (sp - sm) / (2.0 * h)

    return C_num[0] if single else C_num


def _rel_error(A: np.ndarray, B: np.ndarray) -> float:
    norm_B = float(np.linalg.norm(B))
    if norm_B < 1e-12:
        return float(np.max(np.abs(A - B)))
    return float(np.linalg.norm(A - B) / norm_B)


# =========================================================================
# SOLID TANGENT TESTS (n, 6, 6)
# =========================================================================

class TestSolidLaw52Tangents:
    @pytest.fixture
    def base_params(self) -> Law52Params:
        return Law52Params(
            E=210000.0,
            nu=0.3,
            yield_a=400.0,
            hard_b=500.0,
            hard_n=0.5,
            fi=0.01,
            fc=0.15,
            ff=0.25,
            fu=0.3333333333333333,
            epsn=0.30,
            sn=0.10,
            fn=0.04,
            q1=1.5,
            q2=1.0,
            q3=2.25,
        )

    def test_solid_elastic_regime(self, base_params: Law52Params):
        """Verify solid tangent in elastic regime matches Hooke tensor and central FD < 1e-8."""
        deps_el = np.array([0.0002, -0.00005, -0.00005, 0.0001, 0.0, 0.0])
        E, nu = base_params.E, base_params.nu
        G = 0.5 * E / (1.0 + nu)
        c1 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c2 = c1 * nu / (1.0 - nu)
        D_hooke = np.zeros((6, 6))
        D_hooke[0, 0] = D_hooke[1, 1] = D_hooke[2, 2] = c1
        D_hooke[0, 1] = D_hooke[1, 0] = D_hooke[0, 2] = D_hooke[2, 0] = D_hooke[1, 2] = D_hooke[2, 1] = c2
        D_hooke[3, 3] = D_hooke[4, 4] = D_hooke[5, 5] = G

        for h in (1e-7, 1e-6):
            D_alg = tangent_law52_solid(base_params, deps=deps_el, h=h)
            D_num = _compute_num_solid_tangent(base_params, None, deps_el, h=h)

            err_num = _rel_error(D_alg, D_num)
            err_hooke = _rel_error(D_alg, D_hooke)

            assert err_num < 1e-8, f"Solid elastic numerical error {err_num} >= 1e-8 at h={h}"
            assert err_hooke < 1e-8, f"Solid elastic Hooke error {err_hooke} >= 1e-8"

    def test_solid_initial_yielding_no_nucleation(self, base_params: Law52Params):
        """Verify initial yielding without void nucleation (f = f0, epsM << epsN) < 1e-4."""
        deps_y = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])
        extra_y = {"epsm": np.array([0.001]), "sigm": np.array([400.0])}

        for h in (1e-7, 1e-6):
            D_alg = tangent_law52_solid(base_params, deps=deps_y, extra=extra_y, h=h)
            D_num = _compute_num_solid_tangent(base_params, None, deps_y, extra=extra_y, h=h)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"Solid initial yield error {err} >= 1e-4 at h={h}"

    def test_solid_active_void_nucleation(self, base_params: Law52Params):
        """Verify active void nucleation regime (epsM ~ epsN = 0.30) < 1e-4."""
        deps_nuc = np.array([0.006, -0.0015, -0.0015, 0.0025, 0.0, 0.0])
        extra_nuc = {"epsm": np.array([0.30]), "sigm": np.array([550.0])}

        for h in (1e-7, 1e-6):
            D_alg = tangent_law52_solid(base_params, deps=deps_nuc, extra=extra_nuc, h=h)
            D_num = _compute_num_solid_tangent(base_params, None, deps_nuc, extra=extra_nuc, h=h)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"Solid nucleation error {err} >= 1e-4 at h={h}"

    def test_solid_void_growth_high_triaxiality(self, base_params: Law52Params):
        """Verify void growth under high triaxiality (hydrostatic tension) < 1e-4."""
        deps_triax = np.array([0.004, 0.004, 0.004, 0.0, 0.0, 0.0])
        extra_triax = {"epsm": np.array([0.05]), "sigm": np.array([450.0])}

        for h in (1e-7, 1e-6):
            D_alg = tangent_law52_solid(base_params, deps=deps_triax, extra=extra_triax, h=h)
            D_num = _compute_num_solid_tangent(base_params, None, deps_triax, extra=extra_triax, h=h)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"Solid high triaxiality error {err} >= 1e-4 at h={h}"

    def test_solid_void_growth_low_triaxiality(self, base_params: Law52Params):
        """Verify void growth under low triaxiality (pure shear) < 1e-4."""
        deps_shear = np.array([0.0, 0.0, 0.0, 0.015, 0.0, 0.0])
        extra_shear = {"epsm": np.array([0.05]), "sigm": np.array([450.0])}

        for h in (1e-7, 1e-6):
            D_alg = tangent_law52_solid(base_params, deps=deps_shear, extra=extra_shear, h=h)
            D_num = _compute_num_solid_tangent(base_params, None, deps_shear, extra=extra_shear, h=h)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"Solid pure shear error {err} >= 1e-4 at h={h}"

    def test_solid_void_coalescence(self, base_params: Law52Params):
        """Verify void coalescence regime (f > fc = 0.15) < 1e-4."""
        dmg_coal = np.zeros((1, 5))
        dmg_coal[0, 3] = 0.18  # f > fc = 0.15
        extra_coal = {"epsm": np.array([0.20]), "sigm": np.array([500.0]), "dmg": dmg_coal}
        deps_coal = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])

        for h in (1e-7, 1e-6):
            D_alg = tangent_law52_solid(base_params, deps=deps_coal, extra=extra_coal, h=h)
            D_num = _compute_num_solid_tangent(base_params, None, deps_coal, extra=extra_coal, h=h)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"Solid coalescence error {err} >= 1e-4 at h={h}"

    def test_solid_ruptured_eroded_identically_zero(self, base_params: Law52Params):
        """Verify tangent vanishes identically (|D| <= 1e-12) when eroded or ruptured."""
        deps = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])

        # Subcase A: element eroded by off=0
        extra_off = {"off": np.array([0.0])}
        D_off = tangent_law52_solid(base_params, deps=deps, extra=extra_off)
        np.testing.assert_allclose(D_off, 0.0, atol=1e-12)

        # Subcase B: element ruptured by void fraction f >= ff
        dmg_ff = np.zeros((1, 5))
        dmg_ff[0, 3] = 0.26  # f >= ff = 0.25
        extra_ff = {"dmg": dmg_ff}
        D_ff = tangent_law52_solid(base_params, deps=deps, extra=extra_ff)
        np.testing.assert_allclose(D_ff, 0.0, atol=1e-12)

        # Subcase C: element ruptured by f* >= fu
        dmg_fu = np.zeros((1, 5))
        dmg_fu[0, 4] = 0.35  # f* >= fu
        extra_fu = {"dmg": dmg_fu}
        D_fu = tangent_law52_solid(base_params, deps=deps, extra=extra_fu)
        np.testing.assert_allclose(D_fu, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "state_name, deps_vec",
        [
            ("uniaxial_tension", np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])),
            ("equibiaxial_tension", np.array([0.004, 0.004, -0.002, 0.0, 0.0, 0.0])),
            ("pure_shear", np.array([0.0, 0.0, 0.0, 0.010, 0.0, 0.0])),
            ("triaxial_compression", np.array([-0.005, -0.005, -0.005, 0.0, 0.0, 0.0])),
            ("mixed_shear_normal", np.array([0.004, -0.002, 0.001, 0.003, 0.002, 0.001])),
        ],
    )
    def test_solid_multiaxial_loading_states(self, base_params: Law52Params, state_name: str, deps_vec: np.ndarray):
        """Verify solid tangent across diverse multiaxial stress/strain paths < 1e-4."""
        extra = {"epsm": np.array([0.05]), "sigm": np.array([450.0])}
        D_alg = tangent_law52_solid(base_params, deps=deps_vec, extra=extra, h=1e-7)
        D_num = _compute_num_solid_tangent(base_params, None, deps_vec, extra=extra, h=1e-7)
        err = _rel_error(D_alg, D_num)
        assert err < 1e-4, f"Solid {state_name} error {err} >= 1e-4"

    @pytest.mark.parametrize("n", [1, 8, 32])
    def test_solid_vectorization_batches(self, base_params: Law52Params, n: int):
        """Verify vectorized solid tangent (n, 6, 6) across heterogeneous element states."""
        deps = np.zeros((n, 6), dtype=float)
        epsm = np.zeros(n, dtype=float)
        sigm = np.full(n, 400.0, dtype=float)
        dmg = np.zeros((n, 5), dtype=float)
        off = np.ones(n, dtype=float)

        for i in range(n):
            mode = i % 4
            if mode == 0:
                # Elastic
                deps[i] = np.array([0.0002, -0.00005, -0.00005, 0.0001, 0.0, 0.0])
            elif mode == 1:
                # Active yield
                deps[i] = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])
                epsm[i] = 0.05
            elif mode == 2:
                # Coalescence
                deps[i] = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])
                epsm[i] = 0.20
                dmg[i, 3] = 0.18
            else:
                # Eroded
                deps[i] = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])
                off[i] = 0.0

        extra = {"epsm": epsm, "sigm": sigm, "dmg": dmg, "off": off}
        D_batch = tangent_law52_solid(base_params, deps=deps, extra=extra, h=1e-7)

        assert D_batch.shape == (n, 6, 6), f"Expected shape ({n}, 6, 6), got {D_batch.shape}"

        # Verify each element matches single-element call
        for i in range(n):
            ex_i = {
                "epsm": np.array([epsm[i]]),
                "sigm": np.array([sigm[i]]),
                "dmg": dmg[i:i+1].copy(),
                "off": np.array([off[i]]),
            }
            D_single = tangent_law52_solid(base_params, deps=deps[i], extra=ex_i, h=1e-7)
            np.testing.assert_allclose(D_batch[i], D_single, atol=1e-10)

    def test_solid_calling_conventions_and_symmetry(self, base_params: Law52Params):
        """Verify positional, keyword, aliases, and symmetric=True enforcement for solid."""
        deps = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])
        extra = {"epsm": np.array([0.05]), "sigm": np.array([450.0])}

        # 1. Calling forms
        D1 = tangent_law52_solid(base_params, None, deps, extra=extra)
        D2 = consistent_solid_tangent(base_params, deps=deps, extra=extra)
        D3 = solid_tangent(base_params, deps=deps, extra=extra)
        np.testing.assert_allclose(D1, D2, atol=1e-12)
        np.testing.assert_allclose(D1, D3, atol=1e-12)

        # 2. Symmetry check
        D_unsym = tangent_law52_solid(base_params, deps=deps, extra=extra, symmetric=False)
        D_sym = tangent_law52_solid(base_params, deps=deps, extra=extra, symmetric=True)

        # Plastic consistent tangent is naturally non-symmetric
        asym = float(np.max(np.abs(D_unsym - D_unsym.T)))
        assert asym > 1.0, f"Expected non-symmetric plastic tangent, asym={asym}"

        # Symmetrized tangent satisfies D = D.T exactly
        sym_diff = float(np.max(np.abs(D_sym - D_sym.T)))
        assert sym_diff < 1e-12, f"Symmetrized tangent violates symmetry: {sym_diff}"
        np.testing.assert_allclose(D_sym, 0.5 * (D_unsym + D_unsym.T), atol=1e-12)

        # 3. Fallback when deps is None
        D_fallback = tangent_law52_solid(base_params)
        assert D_fallback.shape == (6, 6)
        assert np.any(np.abs(D_fallback) > 0.0)


# =========================================================================
# SHELL MEMBRANE TANGENT TESTS (n, 3, 3)
# =========================================================================

class TestShellLaw52Tangents:
    @pytest.fixture
    def base_params(self) -> Law52Params:
        return Law52Params(
            E=210000.0,
            nu=0.3,
            yield_a=400.0,
            hard_b=500.0,
            hard_n=0.5,
            fi=0.01,
            fc=0.15,
            ff=0.25,
            fu=0.3333333333333333,
            epsn=0.30,
            sn=0.10,
            fn=0.04,
            q1=1.5,
            q2=1.0,
            q3=2.25,
        )

    def test_shell_elastic_regime(self, base_params: Law52Params):
        """Verify shell membrane tangent in elastic regime matches plane stress matrix and central FD < 1e-8."""
        deps_el = np.array([0.0002, -0.00006, 0.0001])
        C_el = shell_membrane_tangent(base_params)

        for h in (1e-7, 1e-6):
            C_alg = tangent_law52_shell(base_params, deps=deps_el, h=h)
            C_num = _compute_num_shell_tangent(base_params, None, deps_el, h=h)

            err_num = _rel_error(C_alg, C_num)
            err_el = _rel_error(C_alg, C_el)

            assert err_num < 1e-8, f"Shell elastic numerical error {err_num} >= 1e-8 at h={h}"
            assert err_el < 1e-8, f"Shell elastic plane stress error {err_el} >= 1e-8"

    def test_shell_initial_yielding_no_nucleation(self, base_params: Law52Params):
        """Verify shell initial yielding without void nucleation (f = f0, epsM << epsN) < 1e-4."""
        deps_y = np.array([0.005, -0.001, 0.002])
        extra_y = {"epsm": np.array([0.001]), "sigm": np.array([400.0])}

        for h in (1e-7, 1e-6):
            C_alg = tangent_law52_shell(base_params, deps=deps_y, extra=extra_y, h=h)
            C_num = _compute_num_shell_tangent(base_params, None, deps_y, extra=extra_y, h=h)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"Shell initial yield error {err} >= 1e-4 at h={h}"

    def test_shell_active_void_nucleation(self, base_params: Law52Params):
        """Verify shell active void nucleation regime (epsM ~ epsN = 0.30) < 1e-4."""
        deps_nuc = np.array([0.006, -0.0015, 0.0025])
        extra_nuc = {"epsm": np.array([0.30]), "sigm": np.array([550.0])}

        for h in (1e-7, 1e-6):
            C_alg = tangent_law52_shell(base_params, deps=deps_nuc, extra=extra_nuc, h=h)
            C_num = _compute_num_shell_tangent(base_params, None, deps_nuc, extra=extra_nuc, h=h)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"Shell nucleation error {err} >= 1e-4 at h={h}"

    def test_shell_void_growth_high_triaxiality(self, base_params: Law52Params):
        """Verify shell void growth under biaxial tension < 1e-4."""
        deps_triax = np.array([0.004, 0.004, 0.0])
        extra_triax = {"epsm": np.array([0.05]), "sigm": np.array([450.0])}

        for h in (1e-7, 1e-6):
            C_alg = tangent_law52_shell(base_params, deps=deps_triax, extra=extra_triax, h=h)
            C_num = _compute_num_shell_tangent(base_params, None, deps_triax, extra=extra_triax, h=h)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"Shell biaxial tension error {err} >= 1e-4 at h={h}"

    def test_shell_void_growth_low_triaxiality(self, base_params: Law52Params):
        """Verify shell void growth under pure shear < 1e-4."""
        deps_shear = np.array([0.0, 0.0, 0.012])
        extra_shear = {"epsm": np.array([0.05]), "sigm": np.array([450.0])}

        for h in (1e-7, 1e-6):
            C_alg = tangent_law52_shell(base_params, deps=deps_shear, extra=extra_shear, h=h)
            C_num = _compute_num_shell_tangent(base_params, None, deps_shear, extra=extra_shear, h=h)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"Shell pure shear error {err} >= 1e-4 at h={h}"

    def test_shell_void_coalescence(self, base_params: Law52Params):
        """Verify shell void coalescence regime (f > fc = 0.15) < 1e-4."""
        dmg_coal = np.zeros((1, 5))
        dmg_coal[0, 3] = 0.18  # f > fc = 0.15
        extra_coal = {"epsm": np.array([0.20]), "sigm": np.array([500.0]), "dmg": dmg_coal}
        deps_coal = np.array([0.005, -0.001, 0.002])

        for h in (1e-7, 1e-6):
            C_alg = tangent_law52_shell(base_params, deps=deps_coal, extra=extra_coal, h=h)
            C_num = _compute_num_shell_tangent(base_params, None, deps_coal, extra=extra_coal, h=h)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"Shell coalescence error {err} >= 1e-4 at h={h}"

    def test_shell_ruptured_eroded_identically_zero(self, base_params: Law52Params):
        """Verify shell tangent vanishes identically (|C| <= 1e-12) when eroded or ruptured."""
        deps = np.array([0.005, -0.001, 0.002])

        # Subcase A: off=0
        extra_off = {"off": np.array([0.0])}
        C_off = tangent_law52_shell(base_params, deps=deps, extra=extra_off)
        np.testing.assert_allclose(C_off, 0.0, atol=1e-12)

        # Subcase B: f >= ff
        dmg_ff = np.zeros((1, 5))
        dmg_ff[0, 3] = 0.26
        extra_ff = {"dmg": dmg_ff}
        C_ff = tangent_law52_shell(base_params, deps=deps, extra=extra_ff)
        np.testing.assert_allclose(C_ff, 0.0, atol=1e-12)

        # Subcase C: f* >= fu
        dmg_fu = np.zeros((1, 5))
        dmg_fu[0, 4] = 0.35
        extra_fu = {"dmg": dmg_fu}
        C_fu = tangent_law52_shell(base_params, deps=deps, extra=extra_fu)
        np.testing.assert_allclose(C_fu, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "state_name, deps_vec",
        [
            ("uniaxial_tension", np.array([0.005, -0.0015, 0.0])),
            ("equibiaxial_tension", np.array([0.003, 0.003, 0.0])),
            ("pure_shear", np.array([0.0, 0.0, 0.008])),
            ("compression", np.array([-0.004, -0.004, 0.0])),
            ("mixed_shear_normal", np.array([0.003, -0.001, 0.004])),
        ],
    )
    def test_shell_multiaxial_loading_states(self, base_params: Law52Params, state_name: str, deps_vec: np.ndarray):
        """Verify shell tangent across diverse plane stress multiaxial paths < 1e-4."""
        extra = {"epsm": np.array([0.05]), "sigm": np.array([450.0])}
        C_alg = tangent_law52_shell(base_params, deps=deps_vec, extra=extra, h=1e-7)
        C_num = _compute_num_shell_tangent(base_params, None, deps_vec, extra=extra, h=1e-7)
        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"Shell {state_name} error {err} >= 1e-4"

    @pytest.mark.parametrize("n", [1, 8, 32])
    def test_shell_vectorization_batches(self, base_params: Law52Params, n: int):
        """Verify vectorized shell tangent (n, 3, 3) across heterogeneous element states."""
        deps = np.zeros((n, 3), dtype=float)
        epsm = np.zeros(n, dtype=float)
        sigm = np.full(n, 400.0, dtype=float)
        dmg = np.zeros((n, 5), dtype=float)
        off = np.ones(n, dtype=float)

        for i in range(n):
            mode = i % 4
            if mode == 0:
                # Elastic
                deps[i] = np.array([0.0002, -0.00006, 0.0001])
            elif mode == 1:
                # Active yield
                deps[i] = np.array([0.005, -0.001, 0.002])
                epsm[i] = 0.05
            elif mode == 2:
                # Coalescence
                deps[i] = np.array([0.005, -0.001, 0.002])
                epsm[i] = 0.20
                dmg[i, 3] = 0.18
            else:
                # Eroded
                deps[i] = np.array([0.005, -0.001, 0.002])
                off[i] = 0.0

        extra = {"epsm": epsm, "sigm": sigm, "dmg": dmg, "off": off}
        C_batch = tangent_law52_shell(base_params, deps=deps, extra=extra, h=1e-7)

        assert C_batch.shape == (n, 3, 3), f"Expected shape ({n}, 3, 3), got {C_batch.shape}"

        for i in range(n):
            ex_i = {
                "epsm": np.array([epsm[i]]),
                "sigm": np.array([sigm[i]]),
                "dmg": dmg[i:i+1].copy(),
                "off": np.array([off[i]]),
            }
            C_single = tangent_law52_shell(base_params, deps=deps[i], extra=ex_i, h=1e-7)
            np.testing.assert_allclose(C_batch[i], C_single, atol=1e-10)

    def test_shell_calling_conventions_and_symmetry(self, base_params: Law52Params):
        """Verify positional, keyword, aliases, and symmetric=True enforcement for shell."""
        deps = np.array([0.005, -0.001, 0.002])
        extra = {"epsm": np.array([0.05]), "sigm": np.array([450.0])}

        # 1. Calling forms
        C1 = tangent_law52_shell(base_params, None, deps, extra=extra)
        C2 = consistent_shell_tangent(base_params, deps=deps, extra=extra)
        C3 = shell_tangent(base_params, deps=deps, extra=extra)
        np.testing.assert_allclose(C1, C2, atol=1e-12)
        np.testing.assert_allclose(C1, C3, atol=1e-12)

        # 2. Symmetry check
        C_unsym = tangent_law52_shell(base_params, deps=deps, extra=extra, symmetric=False)
        C_sym = tangent_law52_shell(base_params, deps=deps, extra=extra, symmetric=True)

        asym = float(np.max(np.abs(C_unsym - C_unsym.T)))
        assert asym > 1.0, f"Expected non-symmetric plastic shell tangent, asym={asym}"

        sym_diff = float(np.max(np.abs(C_sym - C_sym.T)))
        assert sym_diff < 1e-12, f"Symmetrized shell tangent violates symmetry: {sym_diff}"
        np.testing.assert_allclose(C_sym, 0.5 * (C_unsym + C_unsym.T), atol=1e-12)

        # 3. Fallback when deps is None
        C_fallback = tangent_law52_shell(base_params)
        assert C_fallback.shape == (3, 3)
        assert np.any(np.abs(C_fallback) > 0.0)
