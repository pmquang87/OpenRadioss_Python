"""Tests for Milestone M542: /MAT/LAW32 and /MAT/HILL integration and starter validation.

Covers:
1. Starter validation checks: check_mat_law32 parameter verification:
   - Positive initial density (rho0 > 0)
   - Positive Young's modulus (e > 0)
   - Poisson's ratio bounds (0 <= nu < 0.5)
   - Yield stress (a > 0 / sigy > 0)
   - Hardening exponent bound (n <= 1.0, error if hard > 1.0 per hm_read_mat32.F line 219)
   - Reference strain rate (eps0 > 0, error if srp <= 0 per hm_read_mat32.F line 224)
   - Lankford parameters bounds (r00 > 0, r45 > 0, r90 > 0)
   - Registered in starter validation check_model loop
2. Element family compatibility:
   - _ALLOWED_LAWS accepts 32, "32", "LAW32", "HILL" on shells, shells_qbat, shells_qeph, sh3n
   - check_model explicitly rejects LAW32 on bricks, tetras, penta6, pyra5, trusses, beams
     with diagnostic message:
     "/MAT/LAW32/{mat.id} (/MAT/HILL) is not supported for {name} elements (shells only: shells, shells_qbat, shells_qeph, sh3n)"
3. Materials module integration:
   - Registration of 32, "32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL" in MAT_PHYSICS_REGISTRY
   - Registration of uv32: (2,) in _STATE_VAR_COUNT
   - extra_shapes returns uv32: (2,) without nip, and (nip, 2) with nip
   - solid_update raises NotImplementedError for LAW32 / HILL
   - shell_update, sound_speed, shell_membrane_tangent, shell_layer_tangent dispatch
   - extra["uv32"] of shape (n, 2) initialization and update
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
from pyradioss.materials import (
    _STATE_VAR_COUNT,
    extra_shapes,
    shell_layer_tangent,
    shell_membrane_tangent,
    shell_update,
    solid_update,
    sound_speed,
)
from pyradioss.materials.law32_hill import (
    build_law32,
    consistent_shell_tangent as law32_shell_tangent,
    shell_membrane_tangent as law32_membrane_tangent,
    shell_update as law32_shell_update,
    sound_speed as law32_sound_speed,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law32, check_model


class TestLaw32StarterChecks:
    """Starter checks parameter validation for /MAT/LAW32 and /MAT/HILL (M542)."""

    def test_valid_law32_parameters(self):
        mat = Material(
            id=1,
            law=32,
            rho0=7800.0,
            params={
                "e": 2.1e11,
                "nu": 0.3,
                "sigy": 200.0e6,
                "beta": 0.01,
                "hard": 0.25,
                "srp": 1.0,
                "r00": 1.5,
                "r45": 1.2,
                "r90": 1.8,
            },
        )
        log = MessageLog()
        check_mat_law32(mat, log)
        assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"

    def test_negative_and_zero_density_rejected(self):
        # Negative density
        mat = Material(
            id=2,
            law=32,
            rho0=-10.0,
            params={"e": 2e11, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": 1.0},
        )
        log = MessageLog()
        check_mat_law32(mat, log)
        assert any("initial density RHO must be > 0" in msg for msg in log.errors)

        # Zero density
        mat0 = Material(
            id=3,
            law=32,
            rho0=0.0,
            params={"e": 2e11, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": 1.0},
        )
        log0 = MessageLog()
        check_mat_law32(mat0, log0)
        assert any("initial density RHO must be > 0" in msg for msg in log0.errors)

    def test_zero_and_negative_youngs_modulus_rejected(self):
        # Zero modulus
        mat = Material(
            id=4,
            law=32,
            rho0=7800.0,
            params={"e": 0.0, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": 1.0},
        )
        log = MessageLog()
        check_mat_law32(mat, log)
        assert any("Young's modulus E must be > 0" in msg for msg in log.errors)

        # Negative modulus
        mat_neg = Material(
            id=5,
            law=32,
            rho0=7800.0,
            params={"e": -2.1e11, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": 1.0},
        )
        log_neg = MessageLog()
        check_mat_law32(mat_neg, log_neg)
        assert any("Young's modulus E must be > 0" in msg for msg in log_neg.errors)

    def test_bad_poisson_ratio_rejected(self):
        # Negative Poisson's ratio
        mat_neg = Material(
            id=6,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": -0.1, "sigy": 200e6, "hard": 0.5, "srp": 1.0},
        )
        log_neg = MessageLog()
        check_mat_law32(mat_neg, log_neg)
        assert any("Poisson's ratio NU must be in [0, 0.5)" in msg for msg in log_neg.errors)

        # Poisson's ratio == 0.5
        mat_half = Material(
            id=7,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.5, "sigy": 200e6, "hard": 0.5, "srp": 1.0},
        )
        log_half = MessageLog()
        check_mat_law32(mat_half, log_half)
        assert any("Poisson's ratio NU must be in [0, 0.5)" in msg for msg in log_half.errors)

        # Poisson's ratio > 0.5
        mat_big = Material(
            id=8,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.6, "sigy": 200e6, "hard": 0.5, "srp": 1.0},
        )
        log_big = MessageLog()
        check_mat_law32(mat_big, log_big)
        assert any("Poisson's ratio NU must be in [0, 0.5)" in msg for msg in log_big.errors)

    def test_bad_yield_stress_rejected(self):
        # Zero yield stress
        mat_zero = Material(
            id=9,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.3, "sigy": 0.0, "hard": 0.5, "srp": 1.0},
        )
        log_zero = MessageLog()
        check_mat_law32(mat_zero, log_zero)
        assert any("yield stress A (SIGY) must be > 0" in msg for msg in log_zero.errors)

        # Negative yield stress
        mat_neg = Material(
            id=10,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.3, "sigy": -100e6, "hard": 0.5, "srp": 1.0},
        )
        log_neg = MessageLog()
        check_mat_law32(mat_neg, log_neg)
        assert any("yield stress A (SIGY) must be > 0" in msg for msg in log_neg.errors)

    def test_hard_greater_than_one_rejected(self):
        # hard > 1.0 per hm_read_mat32.F line 219
        mat = Material(
            id=11,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.3, "sigy": 200e6, "hard": 1.2, "srp": 1.0},
        )
        log = MessageLog()
        check_mat_law32(mat, log)
        assert any("hardening exponent n must be <= 1.0" in msg for msg in log.errors)

        # hard == 1.0 is valid
        mat_ok = Material(
            id=12,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.3, "sigy": 200e6, "hard": 1.0, "srp": 1.0},
        )
        log_ok = MessageLog()
        check_mat_law32(mat_ok, log_ok)
        assert len(log_ok.errors) == 0

    def test_bad_srp_rejected(self):
        # srp <= 0 per hm_read_mat32.F line 224
        # srp == 0
        mat_zero = Material(
            id=13,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": 0.0},
        )
        log_zero = MessageLog()
        check_mat_law32(mat_zero, log_zero)
        assert any("reference strain rate EPS_DOT_0 (srp) must be > 0" in msg for msg in log_zero.errors)

        # srp < 0
        mat_neg = Material(
            id=14,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": -1.0},
        )
        log_neg = MessageLog()
        check_mat_law32(mat_neg, log_neg)
        assert any("reference strain rate EPS_DOT_0 (srp) must be > 0" in msg for msg in log_neg.errors)

    def test_bad_lankford_parameters_rejected(self):
        # r00 <= 0
        mat_r00 = Material(
            id=15,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": 1.0, "r00": 0.0},
        )
        log_r00 = MessageLog()
        check_mat_law32(mat_r00, log_r00)
        assert any("Lankford parameter r00 must be > 0" in msg for msg in log_r00.errors)

        # r45 <= 0
        mat_r45 = Material(
            id=16,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": 1.0, "r45": -0.5},
        )
        log_r45 = MessageLog()
        check_mat_law32(mat_r45, log_r45)
        assert any("Lankford parameter r45 must be > 0" in msg for msg in log_r45.errors)

        # r90 <= 0
        mat_r90 = Material(
            id=17,
            law=32,
            rho0=7800.0,
            params={"e": 2e11, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": 1.0, "r90": -1.0},
        )
        log_r90 = MessageLog()
        check_mat_law32(mat_r90, log_r90)
        assert any("Lankford parameter r90 must be > 0" in msg for msg in log_r90.errors)


class TestLaw32ElementCompatibility:
    """Element family compatibility for /MAT/LAW32 and /MAT/HILL (M542)."""

    def test_allowed_laws_includes_shells(self):
        shell_families = ["shells", "shells_qbat", "shells_qeph", "sh3n"]
        for fam in shell_families:
            assert fam in _ALLOWED_LAWS, f"Family {fam} not in _ALLOWED_LAWS"
            allowed = _ALLOWED_LAWS[fam]
            for key in (32, "32", "LAW32", "HILL"):
                assert key in allowed, f"Key {key!r} not in _ALLOWED_LAWS[{fam!r}]"

    @pytest.mark.parametrize("fam", ["shells", "shells_qbat", "shells_qeph", "sh3n"])
    def test_shells_accepted_in_check_model(self, fam: str):
        model = Model()
        model.add_nodes(np.array([1, 2, 3, 4]), np.zeros((4, 3)))
        mat = Material(
            id=1,
            law=32,
            rho0=7800.0,
            params={"e": 2.1e11, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": 1.0},
        )
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(fam, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert len(log.errors) == 0, f"Expected no errors on {fam} elements, got: {log.errors}"

    @pytest.mark.parametrize("family", ["bricks", "tetras", "penta6", "pyra5", "trusses", "beams"])
    def test_rejected_on_non_shells_with_diagnostic(self, family: str):
        model = Model()
        model.add_nodes(np.array([1, 2, 3, 4, 5, 6, 7, 8]), np.zeros((8, 3)))
        mat = Material(
            id=32,
            law=32,
            rho0=7800.0,
            params={"e": 2.1e11, "nu": 0.3, "sigy": 200e6, "hard": 0.5, "srp": 1.0},
        )
        model.materials[32] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(family, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        expected_msg = (
            f"/MAT/LAW32/{mat.id} (/MAT/HILL) is not supported for {family} elements "
            f"(shells only: shells, shells_qbat, shells_qeph, sh3n)"
        )
        assert any(
            expected_msg in msg for msg in log.errors
        ), f"Missing expected diagnostic error for family {family}: {log.errors}"


class TestLaw32MaterialDispatchAndInit:
    """Material registry, state variables, dispatch, and tangents (M542)."""

    def test_registry_contains_all_aliases(self):
        for key in (32, "32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL"):
            assert key in MAT_PHYSICS_REGISTRY, f"Key {key!r} not in MAT_PHYSICS_REGISTRY"

    def test_state_var_count_and_extra_shapes(self):
        assert "uv32" in _STATE_VAR_COUNT
        assert _STATE_VAR_COUNT["uv32"] == (2,)

        # Without nip (per-element / default)
        mat = Material(id=1, law=32, rho0=7800.0, params={"e": 2.1e11, "nu": 0.3})
        shapes = extra_shapes(mat)
        assert "uv32" in shapes
        assert shapes["uv32"] == (2,)

        # With nip (per-layer)
        shapes_layer = extra_shapes(mat, nip=5)
        assert "uv32" in shapes_layer
        assert shapes_layer["uv32"] == (5, 2)

        # Also for string law aliases
        for law_str in ("32", "LAW32", "HILL"):
            mat_alias = Material(id=2, law=law_str, rho0=7800.0, params={"e": 2.1e11, "nu": 0.3})
            assert extra_shapes(mat_alias)["uv32"] == (2,)
            assert extra_shapes(mat_alias, nip=3)["uv32"] == (3, 2)

    def test_solid_update_raises_not_implemented(self):
        for law_val in (32, "32", "LAW32", "HILL"):
            mat = Material(id=10, law=law_val, rho0=7800.0, params={"e": 2.1e11, "nu": 0.3})
            sig = np.zeros((2, 6))
            deps = np.zeros((2, 6))
            with pytest.raises(NotImplementedError, match="shell elements only"):
                solid_update(mat, sig, deps, None, 1e-5)

    def test_sound_speed(self):
        mat = build_law32(
            id=1,
            MAT_RHO=7800.0,
            MAT_E=2.1e11,
            MAT_NU=0.3,
            MAT_SIGY=200e6,
        )
        c = sound_speed(mat)
        # For thin shells: c = sqrt(E / (rho0 * (1 - nu^2)))
        expected_c = math.sqrt(2.1e11 / (7800.0 * (1.0 - 0.3 ** 2)))
        assert math.isclose(float(c), expected_c, rel_tol=1e-4)

    def test_shell_membrane_tangent(self):
        mat = build_law32(
            id=1,
            MAT_RHO=7800.0,
            MAT_E=2.1e11,
            MAT_NU=0.3,
            MAT_SIGY=200e6,
        )
        C_mem = shell_membrane_tangent(mat)
        assert C_mem.shape == (3, 3)
        # Elastic plane-stress membrane stiffness
        c11 = 2.1e11 / (1.0 - 0.3 ** 2)
        c12 = 0.3 * c11
        g = 2.1e11 / (2.0 * (1.0 + 0.3))
        assert math.isclose(C_mem[0, 0], c11, rel_tol=1e-4)
        assert math.isclose(C_mem[1, 1], c11, rel_tol=1e-4)
        assert math.isclose(C_mem[0, 1], c12, rel_tol=1e-4)
        assert math.isclose(C_mem[2, 2], g, rel_tol=1e-4)

    def test_shell_layer_tangent(self):
        n = 3
        mat = build_law32(
            id=1,
            MAT_RHO=7800.0,
            MAT_E=2.1e11,
            MAT_NU=0.3,
            MAT_SIGY=200e6,
        )
        sig = np.zeros((n, 3))
        C_lay = shell_layer_tangent(mat, sig, epsp=np.zeros(n), epsp_incr=np.zeros(n))
        assert C_lay.shape == (n, 3, 3)
        for i in range(n):
            assert math.isclose(C_lay[i, 0, 0], 2.1e11 / (1.0 - 0.09), rel_tol=1e-4)

    def test_shell_update_with_uv32_shape_n_2(self):
        n = 4
        mat = build_law32(
            id=1,
            MAT_RHO=7800.0,
            MAT_E=2.1e11,
            MAT_NU=0.3,
            MAT_SIGY=200e6,
            MAT_BETA=0.01,
            MAT_HARD=0.2,
        )
        sig = np.zeros((n, 3))
        deps = np.full((n, 3), 0.005)  # Plastic strain increment
        epsp = np.zeros(n)
        uv32 = np.zeros((n, 2))  # shape (n, 2)
        extra = {"uv32": uv32}

        sig_out, epsp_out = shell_update(mat, sig, deps, epsp, dt=1.0e-5, extra=extra)

        # Verify stress is updated
        assert not np.allclose(sig_out, 0.0)
        # Verify uv32 maintains shape (n, 2)
        assert uv32.shape == (n, 2)
        # Verify plastic strain was accumulated in uv32[:, 0]
        assert np.all(uv32[:, 0] > 0.0)
        assert np.allclose(epsp_out, uv32[:, 0])
