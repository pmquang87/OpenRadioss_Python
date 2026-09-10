"""Tests for Milestone M541: /MAT/LAW38 and /MAT/VISC_TAB integration and starter validation.

Covers:
1. Starter validation checks: check_mat_law38 parameter verification:
   - Positive initial density (rho0 > 0)
   - Positive Young's modulus (e > 0)
   - Poisson's ratios bounds (0 <= nu_t < 0.5, 0 <= nu_c < 0.5)
   - Function count bounds (1 <= nfunc <= 5)
   - Air content active (kcompair == 1) verification: 0 <= phi < 1 and P0 >= 0
   - Registered in starter validation check_model loop
2. Element family compatibility:
   - _ALLOWED_LAWS accepts 38, "38", "LAW38", "VISC_TAB" on bricks, tetras, penta6, pyra5
   - check_model explicitly rejects LAW38 on shells, shells_qbat, shells_qeph, sh3n, trusses, beams
3. Materials module integration:
   - Registration of LAW38 and VISC_TAB in MAT_PHYSICS_REGISTRY
   - Registration of uv38: (33,) in _STATE_VAR_COUNT
   - extra_shapes returns uv38: (33,)
   - needs_env includes 38, "38", "LAW38", "VISC_TAB"
   - solid_update, sound_speed, consistent_solid_tangent dispatch
   - extra["uv38"] of shape (n, 33) initialization and update
   - Curve resolution hook for /FUNCT references so model.curves can be accessed
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
from pyradioss.materials import (
    _STATE_VAR_COUNT,
    consistent_solid_tangent as materials_solid_tangent,
    extra_shapes,
    needs_env,
    resolve_curves,
    solid_tangent,
    solid_update,
    sound_speed,
)
from pyradioss.materials.law38_visc_tab import (
    build_law38,
    consistent_solid_tangent as law38_solid_tangent,
    resolve as law38_resolve,
    solid_update as law38_solid_update,
    sound_speed as law38_sound_speed,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law38, check_model
from pyradioss.starter.initialization import resolve_materials


class TestLaw38StarterChecks:
    """Starter checks parameter validation for /MAT/LAW38 and /MAT/VISC_TAB (M541)."""

    def test_valid_law38_parameters(self):
        mat = Material(
            id=1,
            law=38,
            rho0=1000.0,
            params={
                "e": 2.1e9,
                "nu_t": 0.3,
                "nu_c": 0.35,
                "nfunc": 2,
                "kcompair": 0,
            },
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"

    def test_negative_density_rejected(self):
        mat = Material(
            id=2,
            law=38,
            rho0=-10.0,
            params={"e": 1e9, "nu_t": 0.3, "nu_c": 0.3, "nfunc": 1},
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("initial density RHO must be > 0" in msg for msg in log.errors)

    def test_zero_modulus_rejected(self):
        mat = Material(
            id=3,
            law=38,
            rho0=1000.0,
            params={"e": 0.0, "nu_t": 0.3, "nu_c": 0.3, "nfunc": 1},
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("Young's modulus E must be > 0" in msg for msg in log.errors)

    def test_negative_modulus_rejected(self):
        mat = Material(
            id=4,
            law=38,
            rho0=1000.0,
            params={"e": -5.0e8, "nu_t": 0.3, "nu_c": 0.3, "nfunc": 1},
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("Young's modulus E must be > 0" in msg for msg in log.errors)

    def test_bad_poisson_ratio_tension(self):
        # Negative tensile Poisson's ratio
        mat = Material(
            id=5,
            law=38,
            rho0=1000.0,
            params={"e": 1e9, "nu_t": -0.05, "nu_c": 0.3, "nfunc": 1},
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("tensile Poisson's ratio nu_t must be in [0, 0.5)" in msg for msg in log.errors)

        # Tensile Poisson's ratio >= 0.5
        mat.params["nu_t"] = 0.5
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("tensile Poisson's ratio nu_t must be in [0, 0.5)" in msg for msg in log.errors)

        mat.params["nu_t"] = 0.55
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("tensile Poisson's ratio nu_t must be in [0, 0.5)" in msg for msg in log.errors)

    def test_bad_poisson_ratio_compression(self):
        # Negative compressive Poisson's ratio
        mat = Material(
            id=6,
            law=38,
            rho0=1000.0,
            params={"e": 1e9, "nu_t": 0.3, "nu_c": -0.1, "nfunc": 1},
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("compressive Poisson's ratio nu_c must be in [0, 0.5)" in msg for msg in log.errors)

        # Compressive Poisson's ratio >= 0.5
        mat.params["nu_c"] = 0.5
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("compressive Poisson's ratio nu_c must be in [0, 0.5)" in msg for msg in log.errors)

    def test_bad_function_count(self):
        # nfunc < 1
        mat = Material(
            id=7,
            law=38,
            rho0=1000.0,
            params={"e": 1e9, "nu_t": 0.3, "nu_c": 0.3, "nfunc": 0},
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("number of functions nfunc must be between 1 and 5" in msg for msg in log.errors)

        # nfunc > 5
        mat.params["nfunc"] = 6
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("number of functions nfunc must be between 1 and 5" in msg for msg in log.errors)

    def test_air_content_inactive_allows_arbitrary_phi(self):
        # When kcompair == 0, porosity phi is not checked
        mat = Material(
            id=8,
            law=38,
            rho0=1000.0,
            params={"e": 1e9, "nu_t": 0.3, "nu_c": 0.3, "nfunc": 1, "kcompair": 0, "phi": 1.5, "p0": -1.0},
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert len(log.errors) == 0

    def test_air_content_active_valid(self):
        mat = Material(
            id=9,
            law=38,
            rho0=1000.0,
            params={
                "e": 1e9,
                "nu_t": 0.3,
                "nu_c": 0.3,
                "nfunc": 1,
                "kcompair": 1,
                "phi": 0.45,
                "p0": 101325.0,
            },
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert len(log.errors) == 0

    def test_air_content_active_bad_porosity(self):
        # Negative porosity
        mat = Material(
            id=10,
            law=38,
            rho0=1000.0,
            params={"e": 1e9, "nu_t": 0.3, "nu_c": 0.3, "nfunc": 1, "kcompair": 1, "phi": -0.1, "p0": 1e5},
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("porosity phi must be in [0, 1) when air content is active" in msg for msg in log.errors)

        # Porosity >= 1.0
        mat.params["phi"] = 1.0
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("porosity phi must be in [0, 1) when air content is active" in msg for msg in log.errors)

        mat.params["phi"] = 1.2
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("porosity phi must be in [0, 1) when air content is active" in msg for msg in log.errors)

    def test_air_content_active_bad_pressure(self):
        mat = Material(
            id=11,
            law=38,
            rho0=1000.0,
            params={"e": 1e9, "nu_t": 0.3, "nu_c": 0.3, "nfunc": 1, "kcompair": 1, "phi": 0.5, "p0": -100.0},
        )
        log = MessageLog()
        check_mat_law38(mat, log)
        assert any("initial air pressure P0 must be >= 0" in msg for msg in log.errors)


class TestLaw38ElementCompatibility:
    """Element family compatibility for /MAT/LAW38 and /MAT/VISC_TAB (M541)."""

    def test_allowed_laws_includes_solids(self):
        solid_families = ["bricks", "tetras", "penta6", "pyra5"]
        for fam in solid_families:
            assert fam in _ALLOWED_LAWS, f"Family {fam} not in _ALLOWED_LAWS"
            allowed = _ALLOWED_LAWS[fam]
            for key in (38, "38", "LAW38", "VISC_TAB"):
                assert key in allowed, f"Key {key!r} not in _ALLOWED_LAWS[{fam!r}]"

    def test_solids_accepted_in_check_model(self):
        model = Model()
        model.add_nodes(np.array([1, 2, 3, 4, 5, 6, 7, 8]), np.zeros((8, 3)))
        mat = Material(
            id=1,
            law=38,
            rho0=1000.0,
            params={"e": 1e9, "nu_t": 0.3, "nu_c": 0.3, "nfunc": 1},
        )
        model.materials[1] = mat

        class FakeSolidGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [("bricks", FakeSolidGroup())]

        log = MessageLog()
        check_model(model, log)
        assert len(log.errors) == 0, f"Expected no errors on solid elements, got: {log.errors}"

    @pytest.mark.parametrize("family", ["shells", "shells_qbat", "shells_qeph", "sh3n", "trusses", "beams"])
    def test_rejected_on_non_solids_with_diagnostic(self, family: str):
        model = Model()
        model.add_nodes(np.array([1, 2, 3, 4]), np.zeros((4, 3)))
        mat = Material(
            id=42,
            law=38,
            rho0=1000.0,
            params={"e": 1e9, "nu_t": 0.3, "nu_c": 0.3, "nfunc": 1},
        )
        model.materials[42] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(family, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert any(
            f"/MAT/LAW38/42 (/MAT/VISC_TAB) is not supported for {family} elements" in msg
            for msg in log.errors
        ), f"Missing diagnostic error for family {family}: {log.errors}"


class TestLaw38MaterialDispatchAndInit:
    """Material registry, state variables, dispatch, and curve resolution (M541)."""

    def test_registry_contains_law38_and_visc_tab(self):
        for key in (38, "38", "LAW38", "VISC_TAB"):
            assert key in MAT_PHYSICS_REGISTRY, f"Key {key!r} not in MAT_PHYSICS_REGISTRY"

    def test_build_law38(self):
        rec = {
            "id": 101,
            "title": "FOAM_38",
            "params": {
                "MAT_RHO": 80.0,
                "MAT_E": 1.5e7,
                "MAT_NU": 0.2,
                "MAT_NUt": 0.25,
                "NFUNC": 3,
                "MAT_Kair": 1,
                "MAT_POROS": 0.75,
                "MAT_P0": 101300.0,
            },
        }
        mat = build_law38(rec)
        assert isinstance(mat, Material)
        assert mat.id == 101
        assert mat.law == 38
        assert mat.law_name == "LAW38"
        assert math.isclose(mat.rho0, 80.0)
        assert math.isclose(mat.E, 1.5e7)
        assert math.isclose(mat.nu, 0.25)
        assert mat.params["nfunc"] == 3
        assert mat.params["kcompair"] == 1
        assert math.isclose(mat.params["phi"], 0.75)

    def test_state_var_count_and_extra_shapes(self):
        assert "uv38" in _STATE_VAR_COUNT
        assert _STATE_VAR_COUNT["uv38"] == (33,)

        mat = Material(id=1, law=38, rho0=100.0, params={"e": 1e7, "nu": 0.2})
        shapes = extra_shapes(mat)
        assert "uv38" in shapes
        assert shapes["uv38"] == (33,)

        # Also for string law
        mat_str = Material(id=2, law="LAW38", rho0=100.0, params={"e": 1e7, "nu": 0.2})
        assert extra_shapes(mat_str)["uv38"] == (33,)

        mat_visc = Material(id=3, law="VISC_TAB", rho0=100.0, params={"e": 1e7, "nu": 0.2})
        assert extra_shapes(mat_visc)["uv38"] == (33,)

    def test_needs_env(self):
        for val in (38, "38", "LAW38", "VISC_TAB"):
            mat = Material(id=1, law=val, rho0=100.0)
            assert needs_env(mat) is True, f"needs_env returned False for law={val!r}"

    def test_sound_speed(self):
        mat = Material(
            id=1,
            law=38,
            rho0=1000.0,
            params={"e": 2.1e9, "nu_t": 0.3, "nu_c": 0.3, "npcurve": 0},
        )
        c = sound_speed(mat)
        # For nu=0.3, E=2.1e9, rho=1000:
        # fac = (1 - 0.3)/((1 + 0.3)*(1 - 0.6)) = 0.7 / (1.3 * 0.4) = 0.7 / 0.52 = 1.3461538
        # c = sqrt(2.1e9 * 1.3461538 / 1000) = sqrt(2.826923e6) = 1681.34
        expected_c = math.sqrt(2.1e9 * (0.7 / 0.52) / 1000.0)
        assert math.isclose(float(c), expected_c, rel_tol=1e-4)

        # With array of densities
        rho_arr = np.array([500.0, 1000.0, 2000.0])
        c_arr = sound_speed(mat, rho=rho_arr)
        assert c_arr.shape == (3,)
        assert math.isclose(c_arr[1], expected_c, rel_tol=1e-4)

    def test_solid_update_with_uv38(self):
        n = 4
        mat = Material(
            id=1,
            law=38,
            rho0=1000.0,
            params={"e": 1.0e8, "nu_t": 0.25, "nu_c": 0.25, "npcurve": 0},
        )
        sig = np.zeros((n, 6))
        deps = np.ones((n, 6)) * 1.0e-4
        epsp = np.zeros(n)
        uv38 = np.zeros((n, 33))
        extra = {
            "uv38": uv38,
            "rho": np.full(n, 1000.0),
            "time": 0.005,
        }

        sig_out, epsp_out, c_out = solid_update(mat, sig, deps, epsp, dt=1.0e-5, extra=extra)

        sig_out, epsp_out, c_out = solid_update(mat, sig, deps, epsp, dt=1.0e-5, extra=extra)

        # Check returned stress is non-zero
        assert np.all(sig_out != 0.0)
        # Check uv38 has shape (n, 33) and was updated by constitutive update
        assert uv38.shape == (n, 33)
        assert not np.all(uv38 == 0.0)
        # Check sound speed output
        assert c_out is not None
        assert np.all(c_out > 0.0)

    def test_consistent_solid_tangent(self):
        mat = Material(
            id=1,
            law=38,
            rho0=1000.0,
            params={"e": 1.0e8, "nu_t": 0.25, "nu_c": 0.25},
        )
        sig = np.zeros((2, 6))
        tangent = solid_tangent(mat, sig=sig, epsp=np.zeros(2), epsp_incr=np.zeros(2))
        assert tangent.shape == (2, 6, 6)
        # Symmetry check: C_ijkl == C_klij
        for i in range(2):
            assert np.allclose(tangent[i], tangent[i].T)
        # Positive diagonal stiffness
        assert np.all(tangent[:, 0, 0] > 0.0)
        assert np.all(tangent[:, 3, 3] > 0.0)

    def test_curve_resolution_hook(self):
        model = Model()
        fct1 = FunctTable(101, np.array([0.0, 0.1, 0.5]), np.array([0.0, 1.0e6, 5.0e6]))
        fct2 = FunctTable(102, np.array([0.0, 0.2, 0.6]), np.array([0.0, 2.0e6, 8.0e6]))
        fct_p = FunctTable(201, np.array([0.8, 1.0, 1.2]), np.array([1e5, 0.0, -1e5]))

        model.functions[101] = fct1
        model.functions[102] = fct2
        model.functions[201] = fct_p

        mat = Material(
            id=50,
            law=38,
            rho0=100.0,
            params={
                "e": 1e7,
                "nu": 0.3,
                "ifload": [101, 102],
                "npcurve": 201,
            },
        )
        model.materials[50] = mat

        log = MessageLog()
        resolve_materials(model, log)
        assert len(log.errors) == 0

        # Verify model.curves is accessible
        assert hasattr(model, "curves")
        assert 101 in model.curves
        assert 201 in model.curves

        # Verify curves were resolved into material params
        assert "curve_load" in mat.params
        assert mat.params["curve_load"][0] is not None
        assert mat.params["curve_load"][1] is not None
        assert mat.params["curve_pressure"] is not None

        # Direct resolve_curves call verification
        mat2 = Material(
            id=51,
            law=38,
            rho0=100.0,
            params={"e": 1e7, "nu": 0.3, "ifload": [101]},
        )
        resolve_curves(mat2, model, log)
        assert mat2.params["curve_load"][0] is not None
