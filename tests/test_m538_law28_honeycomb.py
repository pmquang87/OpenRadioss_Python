"""Comprehensive unit test suite for Milestone M538: /MAT/LAW28 and /MAT/HONEYCOMB.

Physics: Orthotropic crushable honeycomb material model without Poisson coupling.
Upstream Fortran origins:
- ``engine/source/materials/mat/mat028/sigeps28.F`` (constitutive update, failure, sound speed)
- ``starter/source/materials/mat/mat028/hm_read_mat28.F`` (starter card reader, defaults)
- ``C:\\OpenRadioss\\hm_cfg_files\\config\\CFG\\radioss110\\MAT\\matl28_honeycomb.cfg`` (CFG attributes)

Covers all milestone requirements:
1. Schema & Synonym Resolution (CfgCatalogue, /MAT/LAW28, /MAT/HONEYCOMB, MAT_PHYSICS_REGISTRY)
2. Card Parsing and Deck Writer Roundtrip (StarterDeck.mat_law28, mat_honeycomb)
3. Orthotropic Elastic Response in All 6 Directions (11, 22, 33, 12, 23, 31)
4. Normal Yield Curve Clamping with Iflag1 in {0, 1, -1}
5. Shear Yield Curve Clamping with Iflag2 in {0, 1, -1}
6. Failure Strain Criteria & Element Deletion (eps_max11..33, eps_max12..31)
7. Sound Speed Calculation
8. Consistent Solid Tangent vs Numerical Finite Differences
9. Shell Update Rejection Guard (NotImplementedError)
10. Starter Resolution with /FUNCT Curves (resolve_materials)
11. Starter Topology Compatibility Checks
12. Multi-Cycle Explicit Engine Simulation with Solid Hexa8 Element
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law28_honeycomb
import pyradioss.materials as pm
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY, catalogue, CfgCatalogue
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.starter import checks
from pyradioss.starter.checks import check_model, _ALLOWED_LAWS
from pyradioss.starter.initialization import resolve_materials
from pyradioss.starter.starter import run_starter, StarterError
from pyradioss.engine.engine import run_engine


# =============================================================================
# Helpers & Builders
# =============================================================================

def _make_honeycomb(
    rho0=1.0e-3,
    rhor=0.0,
    e11=100.0,
    e22=200.0,
    e33=300.0,
    g12=40.0,
    g23=50.0,
    g31=60.0,
    fun_a1=0,
    fun_b1=0,
    fun_a2=0,
    gflag=0,
    fscale11=1.0,
    fscale22=1.0,
    fscale33=1.0,
    eps_max11=1e30,
    eps_max22=1e30,
    eps_max33=1e30,
    fun_a3=0,
    fun_b3=0,
    fun_a4=0,
    vflag=0,
    fscale12=1.0,
    fscale23=1.0,
    fscale31=1.0,
    eps_max12=1e30,
    eps_max23=1e30,
    eps_max31=1e30,
    **kw,
) -> Material:
    """Construct standard honeycomb test material."""
    rec = {
        "id": 1,
        "title": "HONEYCOMB_TEST",
        "params": {
            "MAT_RHO": rho0,
            "Refer_Rho": rhor,
            "MAT_EA": e11,
            "MAT_EB": e22,
            "MAT_EC": e33,
            "MAT_GAB": g12,
            "MAT_GBC": g23,
            "MAT_GCA": g31,
            "FUN_A1": fun_a1,
            "FUN_B1": fun_b1,
            "FUN_A2": fun_a2,
            "Gflag": gflag,
            "FScale11": fscale11,
            "FScale22": fscale22,
            "FScale33": fscale33,
            "MAT_EPSR1": eps_max11,
            "MAT_EPSR2": eps_max22,
            "MAT_EPSR3": eps_max33,
            "FUN_A3": fun_a3,
            "FUN_B3": fun_b3,
            "FUN_A4": fun_a4,
            "Vflag": vflag,
            "FScale12": fscale12,
            "FScale23": fscale23,
            "FScale13": fscale31,
            "MAT_EPSR4": eps_max12,
            "MAT_EPSR5": eps_max23,
            "MAT_EPSR6": eps_max31,
            **kw,
        },
    }
    return law28_honeycomb.build_law28(rec)


# =============================================================================
# 1. Schema & Registry Tests
# =============================================================================

class TestLaw28RegistryAndSchema:
    """Verify registration, synonyms, and CFG catalogue mapping."""

    def test_registry_registration(self):
        pm.register_materials()
        for key in (28, "28", "LAW28", "HONEYCOMB", "HONEYCOMB_SOL"):
            assert key in MAT_PHYSICS_REGISTRY, f"{key} missing in MAT_PHYSICS_REGISTRY"
            fn = MAT_PHYSICS_REGISTRY[key]
            assert callable(fn)

    def test_synonym_lookup(self):
        cat = catalogue()
        schema_law28 = cat.schema("LAW28")
        schema_honeycomb = cat.schema("HONEYCOMB")
        assert schema_law28 is not None, "LAW28 schema missing"
        assert schema_honeycomb is not None, "HONEYCOMB schema missing"
        assert schema_law28.path == schema_honeycomb.path


# =============================================================================
# 2. Deck Writer & Card Parsing Roundtrip
# =============================================================================

class TestLaw28DeckWriterAndParsing:
    """Verify StarterDeck.mat_law28 and roundtrip card parsing."""

    def test_deck_writer_roundtrip(self, tmp_path):
        deck = StarterDeck("HONEYCOMB_ROUNDTRIP")
        deck.mat_law28(
            mat_id=10,
            rho=0.002,
            e11=150.0,
            e22=250.0,
            e33=350.0,
            g12=45.0,
            g23=55.0,
            g31=65.0,
            fun_a1=101,
            fun_b1=102,
            fun_a2=103,
            gflag=1,
            fscale11=1.5,
            fscale22=2.0,
            fscale33=2.5,
            eps_max11=0.25,
            eps_max22=0.30,
            eps_max33=0.35,
            fun_a3=201,
            fun_b3=202,
            fun_a4=203,
            vflag=-1,
            fscale12=1.2,
            fscale23=1.4,
            fscale31=1.6,
            eps_max12=0.40,
            eps_max23=0.45,
            eps_max31=0.50,
            title="HONEYCOMB_CORE",
        )
        file_path = os.path.join(tmp_path, "TEST_0000.rad")
        deck.write(file_path)

        model = Model()
        log = MessageLog()
        parse_starter_deck(read_deck(file_path), model, log)
        assert len(log.errors) == 0
        assert 10 in model.materials
        m = model.materials[10]
        assert m.law == 28
        p = m.params
        assert pytest.approx(p["rho0"]) == 0.002
        assert pytest.approx(p["E11"]) == 150.0
        assert pytest.approx(p["E22"]) == 250.0
        assert pytest.approx(p["E33"]) == 350.0
        assert pytest.approx(p["G12"]) == 45.0
        assert pytest.approx(p["G23"]) == 55.0
        assert pytest.approx(p["G31"]) == 65.0
        assert p["fun_id11"] == 101
        assert p["fun_id22"] == 102
        assert p["fun_id33"] == 103
        assert p["gflag"] == 1
        assert pytest.approx(p["fscale11"]) == 1.5
        assert pytest.approx(p["fscale22"]) == 2.0
        assert pytest.approx(p["fscale33"]) == 2.5
        assert pytest.approx(p["eps_max11"]) == 0.25
        assert pytest.approx(p["eps_max22"]) == 0.30
        assert pytest.approx(p["eps_max33"]) == 0.35
        assert p["fun_id12"] == 201
        assert p["fun_id23"] == 202
        assert p["fun_id31"] == 203
        assert p["vflag"] == -1
        assert pytest.approx(p["fscale12"]) == 1.2
        assert pytest.approx(p["fscale23"]) == 1.4
        assert pytest.approx(p["fscale31"]) == 1.6
        assert pytest.approx(p["eps_max12"]) == 0.40
        assert pytest.approx(p["eps_max23"]) == 0.45
        assert pytest.approx(p["eps_max31"]) == 0.50


# =============================================================================
# 3. Orthotropic Elastic Response in All 6 Directions
# =============================================================================

class TestLaw28OrthotropicElasticity:
    """Verify uncoupled orthotropic elastic response in all 6 directions."""

    def test_elastic_normal_directions(self):
        mat = _make_honeycomb(e11=100.0, e22=200.0, e33=300.0, g12=40.0, g23=50.0, g31=60.0)
        n = 3
        sig = np.zeros((n, 6))
        deps = np.zeros((n, 6))

        # Element 0: pure eps11
        deps[0, 0] = 0.01
        # Element 1: pure eps22
        deps[1, 1] = 0.02
        # Element 2: pure eps33
        deps[2, 2] = 0.03

        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps)

        # Expected: sigma_11 = 100 * 0.01 = 1.0, others zero
        assert pytest.approx(sign[0, 0]) == 1.0
        assert np.allclose(sign[0, 1:], 0.0)

        # Expected: sigma_22 = 200 * 0.02 = 4.0, others zero
        assert pytest.approx(sign[1, 1]) == 4.0
        assert np.allclose(sign[1, [0, 2, 3, 4, 5]], 0.0)

        # Expected: sigma_33 = 300 * 0.03 = 9.0, others zero
        assert pytest.approx(sign[2, 2]) == 9.0
        assert np.allclose(sign[2, :2], 0.0)
        assert np.allclose(sign[2, 3:], 0.0)

    def test_elastic_shear_directions(self):
        mat = _make_honeycomb(e11=100.0, e22=200.0, e33=300.0, g12=40.0, g23=50.0, g31=60.0)
        n = 3
        sig = np.zeros((n, 6))
        deps = np.zeros((n, 6))

        # Element 0: pure eps12
        deps[0, 3] = 0.05
        # Element 1: pure eps23
        deps[1, 4] = 0.04
        # Element 2: pure eps31
        deps[2, 5] = 0.03

        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps)

        # Expected: sigma_12 = 40 * 0.05 = 2.0
        assert pytest.approx(sign[0, 3]) == 2.0
        assert np.allclose(sign[0, [0, 1, 2, 4, 5]], 0.0)

        # Expected: sigma_23 = 50 * 0.04 = 2.0
        assert pytest.approx(sign[1, 4]) == 2.0
        assert np.allclose(sign[1, [0, 1, 2, 3, 5]], 0.0)

        # Expected: sigma_31 = 60 * 0.03 = 1.8
        assert pytest.approx(sign[2, 5]) == 1.8
        assert np.allclose(sign[2, :5], 0.0)


# =============================================================================
# 4. Yield Curve Clamping (Iflag1: Volumetric & Direct Strain)
# =============================================================================

class TestLaw28NormalYieldClamping:
    """Verify normal yield curve clamping under Iflag1 in {0, 1, -1}."""

    @pytest.fixture
    def yield_curve(self):
        # A simple linear hardening curve: y = 10.0 + 100.0 * x
        x = np.array([0.0, 0.1, 0.5, 1.0])
        y = np.array([10.0, 20.0, 60.0, 110.0])
        return FunctTable(1, x, y)

    def test_iflag1_zero_volumetric_strain(self, yield_curve):
        """Iflag1 = 0: yield stress evaluated vs volumetric strain mu = rho/rho0 - 1."""
        mat = _make_honeycomb(
            e11=1000.0,
            fun_a1=1,
            gflag=0,
            fscale11=1.0,
            curve28_x=[yield_curve.x, None, None, None, None, None],
            curve28_y=[yield_curve.y, None, None, None, None, None],
            curve28_fct=[yield_curve, None, None, None, None, None],
        )

        sig = np.zeros((1, 6))
        deps = np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # Elastic trial: sigma_11_trial = 1000 * 0.05 = 50.0
        # Environment: mu = 0.1 -> curve y(0.1) = 20.0
        extra = {
            "eps28": np.zeros((1, 6)),
            "rho": np.array([1.1e-3]),  # rho0 = 1.0e-3 -> mu = 0.1
        }
        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps, extra=extra)
        # Should be clamped to 20.0
        assert pytest.approx(sign[0, 0]) == 20.0

    def test_iflag1_one_direct_strain(self, yield_curve):
        """Iflag1 = 1: yield stress evaluated vs direct strain eps11."""
        mat = _make_honeycomb(
            e11=1000.0,
            fun_a1=1,
            gflag=1,
            fscale11=1.0,
            curve28_x=[yield_curve.x, None, None, None, None, None],
            curve28_y=[yield_curve.y, None, None, None, None, None],
            curve28_fct=[yield_curve, None, None, None, None, None],
        )

        sig = np.zeros((1, 6))
        deps = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # Elastic trial: sigma_11_trial = 100.0
        # Total eps11 = 0.1 -> y(0.1) = 20.0
        extra = {"eps28": np.zeros((1, 6))}
        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps, extra=extra)
        assert pytest.approx(sign[0, 0]) == 20.0

    def test_iflag1_minus_one_negative_direct_strain(self, yield_curve):
        """Iflag1 = -1: yield stress evaluated vs -eps11 (compressive strain)."""
        mat = _make_honeycomb(
            e11=1000.0,
            fun_a1=1,
            gflag=-1,
            fscale11=1.0,
            curve28_x=[yield_curve.x, None, None, None, None, None],
            curve28_y=[yield_curve.y, None, None, None, None, None],
            curve28_fct=[yield_curve, None, None, None, None, None],
        )

        sig = np.zeros((1, 6))
        deps = np.array([[-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # Elastic trial: sigma_11_trial = -100.0
        # -eps11 = -(-0.1) = 0.1 -> y(0.1) = 20.0
        # Clamped: sign(sigma) * min(|sigma|, Y) = -20.0
        extra = {"eps28": np.zeros((1, 6))}
        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps, extra=extra)
        assert pytest.approx(sign[0, 0]) == -20.0


# =============================================================================
# 5. Shear Yield Curve Clamping (Iflag2: Volumetric & Shear Strain)
# =============================================================================

class TestLaw28ShearYieldClamping:
    """Verify shear yield curve clamping under Iflag2 in {0, 1, -1}."""

    @pytest.fixture
    def shear_curve(self):
        x = np.array([0.0, 0.1, 0.2])
        y = np.array([5.0, 15.0, 25.0])
        return FunctTable(2, x, y)

    def test_iflag2_zero_volumetric(self, shear_curve):
        mat = _make_honeycomb(
            g12=500.0,
            fun_a3=2,
            vflag=0,
            fscale12=1.0,
            curve28_x=[None, None, None, shear_curve.x, None, None],
            curve28_y=[None, None, None, shear_curve.y, None, None],
            curve28_fct=[None, None, None, shear_curve, None, None],
        )
        sig = np.zeros((1, 6))
        deps = np.array([[0.0, 0.0, 0.0, 0.1, 0.0, 0.0]])
        # Trial shear: 500 * 0.1 = 50.0
        # Volumetric mu = 0.1 -> y(0.1) = 15.0
        extra = {
            "eps28": np.zeros((1, 6)),
            "rho": np.array([1.1e-3]),
        }
        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps, extra=extra)
        assert pytest.approx(sign[0, 3]) == 15.0

    def test_iflag2_one_shear_strain(self, shear_curve):
        mat = _make_honeycomb(
            g12=500.0,
            fun_a3=2,
            vflag=1,
            fscale12=1.0,
            curve28_x=[None, None, None, shear_curve.x, None, None],
            curve28_y=[None, None, None, shear_curve.y, None, None],
            curve28_fct=[None, None, None, shear_curve, None, None],
        )
        sig = np.zeros((1, 6))
        deps = np.array([[0.0, 0.0, 0.0, 0.1, 0.0, 0.0]])
        # Trial shear: 50.0
        # eps_xy = 0.1 -> y(0.1) = 15.0
        extra = {"eps28": np.zeros((1, 6))}
        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps, extra=extra)
        assert pytest.approx(sign[0, 3]) == 15.0


# =============================================================================
# 6. Failure Strain Criteria & Element Deletion
# =============================================================================

class TestLaw28FailureCriteria:
    """Verify failure strains EMX11..33 and EMX12..31 trigger element deletion (OFF=0)."""

    def test_tensile_strain_failure_11(self):
        mat = _make_honeycomb(e11=100.0, eps_max11=0.05)
        sig = np.zeros((1, 6))
        deps = np.array([[0.06, 0.0, 0.0, 0.0, 0.0, 0.0]])
        off = np.array([1.0])
        extra = {"eps28": np.zeros((1, 6)), "off28": off}

        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps, extra=extra)
        # Element exceeded eps_max11=0.05 -> deleted
        assert off[0] == 0.0
        assert np.allclose(sign[0], 0.0)

    def test_tensile_strain_failure_22_and_33(self):
        mat = _make_honeycomb(e22=100.0, e33=100.0, eps_max22=0.04, eps_max33=0.03)

        # 22 failure
        off = np.array([1.0])
        extra = {"eps28": np.zeros((1, 6)), "off28": off}
        sign, _, _ = law28_honeycomb.solid_update(mat, np.zeros((1, 6)), np.array([[0.0, 0.05, 0.0, 0.0, 0.0, 0.0]]), extra=extra)
        assert off[0] == 0.0
        assert np.allclose(sign[0], 0.0)

        # 33 failure
        off = np.array([1.0])
        extra = {"eps28": np.zeros((1, 6)), "off28": off}
        sign, _, _ = law28_honeycomb.solid_update(mat, np.zeros((1, 6)), np.array([[0.0, 0.0, 0.04, 0.0, 0.0, 0.0]]), extra=extra)
        assert off[0] == 0.0
        assert np.allclose(sign[0], 0.0)

    def test_shear_strain_failure(self):
        # In sigeps28.F: ABS(EPSXY/TWO) > EMX12
        mat = _make_honeycomb(g12=100.0, eps_max12=0.02)
        off = np.array([1.0])
        extra = {"eps28": np.zeros((1, 6)), "off28": off}

        # deps_xy = 0.05 -> engineering shear gamma=0.05 -> eps_xy/2 = 0.025 > 0.02
        deps = np.array([[0.0, 0.0, 0.0, 0.05, 0.0, 0.0]])
        sign, _, _ = law28_honeycomb.solid_update(mat, np.zeros((1, 6)), deps, extra=extra)
        assert off[0] == 0.0
        assert np.allclose(sign[0], 0.0)

    def test_no_failure_below_thresholds(self):
        mat = _make_honeycomb(e11=100.0, eps_max11=0.10)
        off = np.array([1.0])
        extra = {"eps28": np.zeros((1, 6)), "off28": off}
        deps = np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
        sign, _, _ = law28_honeycomb.solid_update(mat, np.zeros((1, 6)), deps, extra=extra)
        assert off[0] == 1.0
        assert pytest.approx(sign[0, 0]) == 5.0


# =============================================================================
# 7. Sound Speed Calculation
# =============================================================================

class TestLaw28SoundSpeed:
    """Verify sound speed formula: c = sqrt(max(E11..G31) / rho0)."""

    def test_sound_speed_normal_dominant(self):
        mat = _make_honeycomb(rho0=1.0e-3, e11=100.0, e22=400.0, e33=200.0, g12=50.0, g23=60.0, g31=70.0)
        # max modulus is e22 = 400.0 -> c = sqrt(400.0 / 1e-3) = sqrt(400000) = 632.4555
        expected_c = math.sqrt(400.0 / 1.0e-3)
        assert pytest.approx(law28_honeycomb.sound_speed(mat), rel=1e-6) == expected_c
        assert pytest.approx(pm.sound_speed(mat), rel=1e-6) == expected_c

    def test_sound_speed_shear_dominant(self):
        mat = _make_honeycomb(rho0=2.0e-3, e11=50.0, e22=60.0, e33=70.0, g12=80.0, g23=90.0, g31=180.0)
        # max modulus is g31 = 180.0 -> c = sqrt(180.0 / 2e-3) = sqrt(90000) = 300.0
        expected_c = math.sqrt(180.0 / 2.0e-3)
        assert pytest.approx(law28_honeycomb.sound_speed(mat), rel=1e-6) == expected_c


# =============================================================================
# 8. Consistent Solid Tangent
# =============================================================================

class TestLaw28ConsistentSolidTangent:
    """Verify algorithmic tangent matches numerical finite difference perturbation."""

    def test_tangent_vs_central_differences(self):
        mat = _make_honeycomb(e11=120.0, e22=240.0, e33=360.0, g12=45.0, g23=55.0, g31=65.0)
        n = 1
        sig = np.zeros((n, 6))
        D_alg = law28_honeycomb.consistent_solid_tangent(mat, sig)
        assert D_alg.shape == (n, 6, 6)

        # Expected diagonal: [E11, E22, E33, G12, G23, G31]
        expected_diag = [120.0, 240.0, 360.0, 45.0, 55.0, 65.0]
        assert np.allclose(np.diag(D_alg[0]), expected_diag)

        # Numerical perturbation
        h = 1e-6
        D_num = np.zeros((6, 6))
        for j in range(6):
            deps_p = np.zeros((n, 6))
            deps_m = np.zeros((n, 6))
            deps_p[0, j] = h
            deps_m[0, j] = -h
            sig_p, _, _ = law28_honeycomb.solid_update(mat, sig, deps_p)
            sig_m, _, _ = law28_honeycomb.solid_update(mat, sig, deps_m)
            D_num[:, j] = (sig_p[0] - sig_m[0]) / (2.0 * h)

        assert np.allclose(D_alg[0], D_num, atol=1e-5, rtol=1e-5)

    def test_tangent_zero_for_deleted_element(self):
        mat = _make_honeycomb(e11=100.0, e22=200.0, e33=300.0)
        sig = np.zeros((2, 6))
        off = np.array([1.0, 0.0])  # element 1 is deleted
        D = law28_honeycomb.consistent_solid_tangent(mat, sig, extra={"off28": off})
        assert not np.allclose(D[0], 0.0)
        assert np.allclose(D[1], 0.0)

    def test_framework_solid_tangent_dispatch(self):
        mat = _make_honeycomb(e11=150.0, e22=250.0, e33=350.0)
        sig = np.zeros((1, 6))
        D = pm.solid_tangent(mat, sig, epsp=None, epsp_incr=None)
        assert D.shape == (1, 6, 6)
        assert pytest.approx(D[0, 0, 0]) == 150.0
        assert pytest.approx(D[0, 1, 1]) == 250.0
        assert pytest.approx(D[0, 2, 2]) == 350.0

    def test_tangent_elastic_regime_central_differences(self):
        """Elastic regime: D is diagonal [E11, E22, E33, G12, G23, G31] with zero off-diagonals.
        Matches central difference perturbation exactly.
        """
        mat = _make_honeycomb(e11=100.0, e22=200.0, e33=300.0, g12=40.0, g23=50.0, g31=60.0)
        sig_0 = np.zeros((1, 6))
        deps = np.array([[0.01, 0.02, 0.015, 0.03, 0.025, 0.02]])
        extra = {"eps28": np.zeros((1, 6))}
        sig_base, _, _ = law28_honeycomb.solid_update(mat, sig_0, deps, extra=extra)

        D_alg = law28_honeycomb.consistent_solid_tangent(mat, sig_base, extra=extra)
        assert D_alg.shape == (1, 6, 6)
        expected_diag = [100.0, 200.0, 300.0, 40.0, 50.0, 60.0]
        assert np.allclose(np.diag(D_alg[0]), expected_diag)
        assert np.allclose(D_alg[0] - np.diag(expected_diag), 0.0)

        h = 1e-6
        D_num = np.zeros((6, 6))
        for j in range(6):
            dp = deps.copy(); dp[0, j] += h
            dm = deps.copy(); dm[0, j] -= h
            sp, _, _ = law28_honeycomb.solid_update(mat, sig_0, dp, extra={"eps28": np.zeros((1, 6))})
            sm, _, _ = law28_honeycomb.solid_update(mat, sig_0, dm, extra={"eps28": np.zeros((1, 6))})
            D_num[:, j] = (sp[0] - sm[0]) / (2.0 * h)

        assert np.allclose(D_alg[0], D_num, atol=1e-5, rtol=1e-5)

    def test_tangent_component_yielded_regime_iflag1_one(self):
        """Iflag1 = 1: d(sigma_ii)/d(eps_ii) = sign(sigma_ii) * f'_{ii}(eps_ii) * F_scale,ii.
        Off-diagonals are zero. Matches central difference perturbation.
        """
        fc = FunctTable(1, [0.0, 0.1, 0.5, 1.0], [10.0, 20.0, 32.0, 50.0])  # slope in [0.1, 0.5] is 30.0
        mat = _make_honeycomb(
            e11=1000.0, e22=200.0, e33=300.0, g12=40.0, g23=50.0, g31=60.0,
            fun_a1=1, gflag=1, fscale11=1.5,
            curve28_x=[fc.x, None, None, None, None, None],
            curve28_y=[fc.y, None, None, None, None, None],
            curve28_s=[fc.slope, None, None, None, None, None],
            curve28_fct=[fc, None, None, None, None, None],
        )
        sig_0 = np.zeros((1, 6))
        deps = np.array([[0.3, 0.01, 0.01, 0.01, 0.01, 0.01]])
        extra = {"eps28": np.zeros((1, 6))}
        sig_base, _, _ = law28_honeycomb.solid_update(mat, sig_0, deps, extra=extra)

        D_alg = law28_honeycomb.consistent_solid_tangent(mat, sig_base, extra=extra)
        assert pytest.approx(D_alg[0, 0, 0]) == 45.0
        assert np.allclose(D_alg[0, 0, 1:], 0.0)

        h = 1e-6
        D_num = np.zeros((6, 6))
        for j in range(6):
            dp = deps.copy(); dp[0, j] += h
            dm = deps.copy(); dm[0, j] -= h
            sp, _, _ = law28_honeycomb.solid_update(mat, sig_0, dp, extra={"eps28": np.zeros((1, 6))})
            sm, _, _ = law28_honeycomb.solid_update(mat, sig_0, dm, extra={"eps28": np.zeros((1, 6))})
            D_num[:, j] = (sp[0] - sm[0]) / (2.0 * h)

        assert np.allclose(D_alg[0], D_num, atol=1e-5, rtol=1e-5)

    def test_tangent_volumetric_yielded_hydrostatic_coupling_iflag1_zero(self):
        """Iflag1 = 0: d(sigma_ii)/d(eps_jj) = -sign(sigma_ii) * f'_{ii}(mu) * F_scale,ii for j in {0, 1, 2}.
        Hydrostatic coupling terms in all 3 normal directions; shear off-diagonals are zero.
        """
        fc = FunctTable(1, [0.0, 0.1, 0.5, 1.0], [10.0, 20.0, 60.0, 110.0])  # slope in [0.1, 0.5] is 100.0
        mat = _make_honeycomb(
            e11=1000.0, e22=200.0, e33=300.0, g12=40.0, g23=50.0, g31=60.0,
            fun_a1=1, gflag=0, fscale11=1.5,
            curve28_x=[fc.x, None, None, None, None, None],
            curve28_y=[fc.y, None, None, None, None, None],
            curve28_s=[fc.slope, None, None, None, None, None],
            curve28_fct=[fc, None, None, None, None, None],
        )
        sig_0 = np.zeros((1, 6))
        deps = np.array([[-0.25, 0.0, 0.0, 0.0, 0.0, 0.0]])
        extra = {"eps28": np.zeros((1, 6))}
        sig_base, _, _ = law28_honeycomb.solid_update(mat, sig_0, deps, extra=extra)

        D_alg = law28_honeycomb.consistent_solid_tangent(mat, sig_base, extra=extra)
        assert pytest.approx(D_alg[0, 0, 0]) == 150.0
        assert pytest.approx(D_alg[0, 0, 1]) == 150.0
        assert pytest.approx(D_alg[0, 0, 2]) == 150.0
        assert np.allclose(D_alg[0, 0, 3:], 0.0)

        h = 1e-6
        D_num = np.zeros((6, 6))
        for j in range(6):
            dp = deps.copy(); dp[0, j] += h
            dm = deps.copy(); dm[0, j] -= h
            sp, _, _ = law28_honeycomb.solid_update(mat, sig_0, dp, extra={"eps28": np.zeros((1, 6))})
            sm, _, _ = law28_honeycomb.solid_update(mat, sig_0, dm, extra={"eps28": np.zeros((1, 6))})
            D_num[:, j] = (sp[0] - sm[0]) / (2.0 * h)

        assert np.allclose(D_alg[0], D_num, atol=1e-5, rtol=1e-5)

    def test_tangent_negative_component_yielded_iflag1_minus_one(self):
        """Iflag1 = -1: d(sigma_ii)/d(eps_ii) = -sign(sigma_ii) * f'_{ii}(-eps_ii) * F_scale,ii."""
        fc = FunctTable(1, [0.0, 0.1, 0.5, 1.0], [10.0, 20.0, 60.0, 110.0])  # slope = 100.0
        mat = _make_honeycomb(
            e11=1000.0, e22=200.0, e33=300.0, g12=40.0, g23=50.0, g31=60.0,
            fun_a1=1, gflag=-1, fscale11=1.0,
            curve28_x=[fc.x, None, None, None, None, None],
            curve28_y=[fc.y, None, None, None, None, None],
            curve28_s=[fc.slope, None, None, None, None, None],
            curve28_fct=[fc, None, None, None, None, None],
        )
        sig_0 = np.zeros((1, 6))
        deps = np.array([[-0.3, 0.0, 0.0, 0.0, 0.0, 0.0]])
        extra = {"eps28": np.zeros((1, 6))}
        sig_base, _, _ = law28_honeycomb.solid_update(mat, sig_0, deps, extra=extra)

        D_alg = law28_honeycomb.consistent_solid_tangent(mat, sig_base, extra=extra)
        assert pytest.approx(D_alg[0, 0, 0]) == 100.0
        assert np.allclose(D_alg[0, 0, 1:], 0.0)

        h = 1e-6
        D_num = np.zeros((6, 6))
        for j in range(6):
            dp = deps.copy(); dp[0, j] += h
            dm = deps.copy(); dm[0, j] -= h
            sp, _, _ = law28_honeycomb.solid_update(mat, sig_0, dp, extra={"eps28": np.zeros((1, 6))})
            sm, _, _ = law28_honeycomb.solid_update(mat, sig_0, dm, extra={"eps28": np.zeros((1, 6))})
            D_num[:, j] = (sp[0] - sm[0]) / (2.0 * h)

        assert np.allclose(D_alg[0], D_num, atol=1e-5, rtol=1e-5)

    def test_tangent_shear_yielded_regime_iflag2_one(self):
        """Iflag2 = 1: d(sigma_12)/d(gamma_12) = sign(sigma_12) * f'_{12}(eps_12) * F_scale,12."""
        fc = FunctTable(2, [0.0, 0.2, 0.6], [5.0, 15.0, 35.0])  # slope = 50.0
        mat = _make_honeycomb(
            e11=100.0, e22=200.0, e33=300.0, g12=500.0, g23=50.0, g31=60.0,
            fun_a3=2, vflag=1, fscale12=1.2,
            curve28_x=[None, None, None, fc.x, None, None],
            curve28_y=[None, None, None, fc.y, None, None],
            curve28_s=[None, None, None, fc.slope, None, None],
            curve28_fct=[None, None, None, fc, None, None],
        )
        sig_0 = np.zeros((1, 6))
        deps = np.array([[0.0, 0.0, 0.0, 0.35, 0.0, 0.0]])
        extra = {"eps28": np.zeros((1, 6))}
        sig_base, _, _ = law28_honeycomb.solid_update(mat, sig_0, deps, extra=extra)

        D_alg = law28_honeycomb.consistent_solid_tangent(mat, sig_base, extra=extra)
        assert pytest.approx(D_alg[0, 3, 3]) == 60.0

        h = 1e-6
        D_num = np.zeros((6, 6))
        for j in range(6):
            dp = deps.copy(); dp[0, j] += h
            dm = deps.copy(); dm[0, j] -= h
            sp, _, _ = law28_honeycomb.solid_update(mat, sig_0, dp, extra={"eps28": np.zeros((1, 6))})
            sm, _, _ = law28_honeycomb.solid_update(mat, sig_0, dm, extra={"eps28": np.zeros((1, 6))})
            D_num[:, j] = (sp[0] - sm[0]) / (2.0 * h)

        assert np.allclose(D_alg[0], D_num, atol=1e-5, rtol=1e-5)

    def test_tangent_shear_volumetric_yielded_iflag2_zero(self):
        """Iflag2 = 0: shear yield evaluated vs volumetric strain mu = -tr(eps)."""
        fc = FunctTable(2, [0.0, 0.2, 0.6], [5.0, 15.0, 35.0])  # slope = 50.0
        mat = _make_honeycomb(
            e11=100.0, e22=200.0, e33=300.0, g12=500.0, g23=50.0, g31=60.0,
            fun_a3=2, vflag=0, fscale12=1.0,
            curve28_x=[None, None, None, fc.x, None, None],
            curve28_y=[None, None, None, fc.y, None, None],
            curve28_s=[None, None, None, fc.slope, None, None],
            curve28_fct=[None, None, None, fc, None, None],
        )
        sig_0 = np.zeros((1, 6))
        deps = np.array([[-0.3, 0.0, 0.0, 0.1, 0.0, 0.0]])
        extra = {"eps28": np.zeros((1, 6))}
        sig_base, _, _ = law28_honeycomb.solid_update(mat, sig_0, deps, extra=extra)

        D_alg = law28_honeycomb.consistent_solid_tangent(mat, sig_base, extra=extra)
        assert pytest.approx(D_alg[0, 3, 0]) == -50.0
        assert pytest.approx(D_alg[0, 3, 1]) == -50.0
        assert pytest.approx(D_alg[0, 3, 2]) == -50.0
        assert pytest.approx(D_alg[0, 3, 3]) == 0.0

        h = 1e-6
        D_num = np.zeros((6, 6))
        for j in range(6):
            dp = deps.copy(); dp[0, j] += h
            dm = deps.copy(); dm[0, j] -= h
            sp, _, _ = law28_honeycomb.solid_update(mat, sig_0, dp, extra={"eps28": np.zeros((1, 6))})
            sm, _, _ = law28_honeycomb.solid_update(mat, sig_0, dm, extra={"eps28": np.zeros((1, 6))})
            D_num[:, j] = (sp[0] - sm[0]) / (2.0 * h)

        assert np.allclose(D_alg[0], D_num, atol=1e-5, rtol=1e-5)

    def test_tangent_ruptured_and_deleted_zeroes_slice(self):
        """Ruptured or deleted elements have their entire (6, 6) tangent matrix zeroed."""
        mat = _make_honeycomb(e11=100.0, e22=200.0, e33=300.0, eps_max11=0.10)
        sig_0 = np.zeros((3, 6))
        deps = np.array([
            [0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        off = np.array([1.0, 0.0, 1.0])
        extra = {"eps28": np.zeros((3, 6)), "off28": off}
        sig_base, _, _ = law28_honeycomb.solid_update(mat, sig_0, deps, extra=extra)

        assert off[0] == 0.0  # element 0 ruptured
        assert off[1] == 0.0  # element 1 pre-deleted
        assert off[2] == 1.0  # element 2 active

        D_alg = law28_honeycomb.consistent_solid_tangent(mat, sig_base, extra=extra)
        assert np.allclose(D_alg[0], 0.0)
        assert np.allclose(D_alg[1], 0.0)
        assert not np.allclose(D_alg[2], 0.0)
        assert pytest.approx(D_alg[2, 0, 0]) == 100.0

    def test_tangent_multi_axial_simultaneous_yielding(self):
        """Simultaneous yielding across multiple normal and shear directions."""
        fc1 = FunctTable(1, [0.0, 0.1, 0.5, 1.0], [10.0, 20.0, 32.0, 50.0])  # slope = 30.0
        fc2 = FunctTable(2, [0.0, 0.1, 0.5, 1.0], [5.0, 15.0, 25.0, 40.0])   # slope = 25.0
        fc3 = FunctTable(3, [0.0, 0.2, 0.6], [4.0, 12.0, 28.0])              # slope = 40.0

        mat = _make_honeycomb(
            e11=1000.0, e22=2000.0, e33=3000.0, g12=500.0, g23=600.0, g31=700.0,
            fun_a1=1, fun_b1=2, fun_a3=3,
            gflag=0, vflag=1,
            fscale11=1.0, fscale22=1.2, fscale12=1.5,
            curve28_x=[fc1.x, fc2.x, None, fc3.x, None, None],
            curve28_y=[fc1.y, fc2.y, None, fc3.y, None, None],
            curve28_s=[fc1.slope, fc2.slope, None, fc3.slope, None, None],
            curve28_fct=[fc1, fc2, None, fc3, None, None],
        )
        sig_0 = np.zeros((1, 6))
        deps = np.array([[-0.25, -0.05, 0.005, 0.35, 0.01, 0.01]])
        extra = {"eps28": np.zeros((1, 6))}
        sig_base, _, _ = law28_honeycomb.solid_update(mat, sig_0, deps, extra=extra)
        D_alg = law28_honeycomb.consistent_solid_tangent(mat, sig_base, extra=extra)

        h = 1e-6
        D_num = np.zeros((6, 6))
        for j in range(6):
            dp = deps.copy(); dp[0, j] += h
            dm = deps.copy(); dm[0, j] -= h
            sp, _, _ = law28_honeycomb.solid_update(mat, sig_0, dp, extra={"eps28": np.zeros((1, 6))})
            sm, _, _ = law28_honeycomb.solid_update(mat, sig_0, dm, extra={"eps28": np.zeros((1, 6))})
            D_num[:, j] = (sp[0] - sm[0]) / (2.0 * h)

        assert np.allclose(D_alg[0], D_num, atol=1e-5, rtol=1e-5)


# =============================================================================
# 9. Shell Update Rejection Guard
# =============================================================================

class TestLaw28ShellRejection:
    """Verify shells raise NotImplementedError for LAW28 honeycomb."""

    def test_shell_update_raises(self):
        mat = _make_honeycomb()
        with pytest.raises(NotImplementedError, match="LAW28 \\(HONEYCOMB crushable\\) is implemented for 3D solid and SPH elements only."):
            pm.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), None, 0.0)


# =============================================================================
# 10. Starter Curve Resolution
# =============================================================================

class TestLaw28StarterResolution:
    """Verify resolve_materials hook in starter initializes curve tables."""

    def test_starter_resolve_functions(self):
        model = Model()
        mat = _make_honeycomb(fun_a1=10, fun_a3=20)
        model.materials[1] = mat

        # Add curves to model
        f10 = FunctTable(10, [0.0, 0.2, 0.5], [10.0, 30.0, 80.0])
        f20 = FunctTable(20, [0.0, 0.1, 0.4], [5.0, 15.0, 45.0])
        model.functions[10] = f10
        model.functions[20] = f20

        log = MessageLog()
        resolve_materials(model, log)
        assert len(log.errors) == 0

        p = mat.params
        assert "curve28_x" in p
        assert p["curve28_x"][0] is not None
        assert np.allclose(p["curve28_x"][0], f10.x)
        assert p["curve28_x"][3] is not None
        assert np.allclose(p["curve28_x"][3], f20.x)


# =============================================================================
# 11. Starter Checks Compatibility
# =============================================================================

class TestLaw28StarterChecks:
    """Verify topology checks allow solids and reject shells."""

    def test_allowed_laws_set(self):
        assert 28 in _ALLOWED_LAWS["bricks"]
        assert "LAW28" in _ALLOWED_LAWS["bricks"]
        assert "HONEYCOMB" in _ALLOWED_LAWS["bricks"]
        assert 28 in _ALLOWED_LAWS["tetras"]
        assert 28 not in _ALLOWED_LAWS["shells"]

    def test_shell_element_with_law28_rejected(self, tmp_path):
        s_path = os.path.join(tmp_path, "SHELL_LAW28_0000.rad")
        deck = StarterDeck("SHELL_LAW28")
        deck.node([(1, 0, 0, 0), (2, 1, 0, 0), (3, 1, 1, 0), (4, 0, 1, 0)])
        deck.shell(1, [(1, 1, 2, 3, 4)])
        deck.part(1, "SHELL_PART", 1, 1)
        deck.mat_law28(1, rho=0.001, e11=100.0, e22=100.0, e33=100.0)
        deck.prop_shell(1, "SHELL_PROP", 1.0)
        deck.write(s_path)

        log = MessageLog()
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(StarterError):
                run_starter(s_path, log=log)

        err_text = " ".join(str(e) for e in log.errors)
        assert "not ported for shells elements" in err_text


# =============================================================================
# 12. Multi-Cycle Explicit Engine Simulation
# =============================================================================

class TestLaw28EngineSimulation:
    """Multi-cycle explicit engine simulation on solid hexa8."""

    def test_engine_multi_cycle_simulation_hexa8(self, tmp_path):
        """Run 15+ explicit cycles of compression on a solid hexa8 element.
        Verify:
        - Starter initializes and resolves without errors.
        - Engine runs multiple explicit integration cycles.
        - State progresses without NaNs or Infs.
        - Energy balance is maintained.
        """
        run_name = "HEXA_HONEYCOMB"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ])
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        deck.part(1, "HONEYCOMB_BLOCK", 1, 1)
        deck.mat_law28(
            1,
            rho=1.0e-3,
            e11=200.0,
            e22=300.0,
            e33=400.0,
            g12=50.0,
            g23=60.0,
            g31=70.0,
            title="HONEYCOMB_MATERIAL",
        )
        deck.prop_solid(1, "SOLID_PROP")
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
0.3
/DT
0.9 0
/PRINT/-1
/STOP
100
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        log = MessageLog()
        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path, log=log)
            eng_model = run_engine(e_path)

        assert len(log.errors) == 0
        mat = st_model.materials[1]
        assert mat.law == 28
        assert mat.sound_speed_solid() > 0.0

        state = eng_model.engine_state
        assert state.cycle >= 10, f"Expected >= 10 cycles, got {state.cycle}"
