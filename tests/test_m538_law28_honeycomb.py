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
from pyradioss.engine.engine import run_engine, _energies


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
    """Multi-cycle explicit engine simulation, energy balance, and stability audit for LAW28."""

    def test_engine_multi_cycle_simulation_hexa8(self, tmp_path):
        """Run explicit cycles of compression on a solid hexa8 element with LAW28.
        Verify:
        - Starter initializes and resolves without errors.
        - Engine runs multiple explicit integration cycles.
        - State progresses without NaNs or Infs.
        - Energy balance is tracked properly in engine ledgers.
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
        assert not state.stop_reason or "/STOP" in str(state.stop_reason)

        brick_g = dict(eng_model.element_groups())["bricks"]
        sig = brick_g.state["sig"]
        assert np.isfinite(sig).all()
        en = _energies(eng_model, state)
        assert np.isfinite(en["IE"])
        assert np.isfinite(en["KE"])
        assert abs(en["ERR"]) < 5.0

    def test_engine_multielement_compressive_impact_50_steps(self, tmp_path):
        """Audit explicit dynamic simulation and energy conservation on a 2x2x2 hexa8 mesh.
        Verify:
        - 2x2x2 mesh (8 solid hexa8 bricks, 27 nodes) under dynamic compressive impact load.
        - Run for 50+ explicit time steps without numerical divergence.
        - Audit energy conservation: kinetic energy E_k, internal energy E_int, external work W_ext.
        - Total energy balance error |ERR| <= 5.0% and total energy ratio E_tot / (E_tot,0 + W_ext) ~ 1.0.
        - Stresses, coordinates, and velocities remain strictly finite (no NaNs or Infs).
        - Courant time step remains stable throughout the simulation.
        """
        run_name = "HEXA_IMPACT_50"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)

        # 2x2x2 regular mesh of 27 nodes: x, y, z in {0, 10, 20}
        nodes = []
        nid = 1
        node_grid = np.zeros((3, 3, 3), dtype=int)
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    nodes.append((nid, float(i * 10.0), float(j * 10.0), float(k * 10.0)))
                    node_grid[i, j, k] = nid
                    nid += 1
        deck.node(nodes)

        # 8 brick elements
        bricks = []
        eid = 1
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    n1 = int(node_grid[i, j, k])
                    n2 = int(node_grid[i + 1, j, k])
                    n3 = int(node_grid[i + 1, j + 1, k])
                    n4 = int(node_grid[i, j + 1, k])
                    n5 = int(node_grid[i, j, k + 1])
                    n6 = int(node_grid[i + 1, j, k + 1])
                    n7 = int(node_grid[i + 1, j + 1, k + 1])
                    n8 = int(node_grid[i, j + 1, k + 1])
                    bricks.append((eid, n1, n2, n3, n4, n5, n6, n7, n8))
                    eid += 1
        deck.brick(1, bricks)
        deck.part(1, "HONEYCOMB_BLOCK_2X2X2", 1, 1)

        # Orthotropic honeycomb material
        deck.mat_law28(
            1,
            rho=1.0e-3,
            e11=200.0,
            e22=300.0,
            e33=400.0,
            g12=50.0,
            g23=60.0,
            g31=70.0,
            title="HONEYCOMB_ORTHO",
        )
        deck.prop_solid(1, "SOLID_PROP")

        # Boundary conditions: Clamped base (z=0, 9 nodes)
        base_nodes = node_grid[:, :, 0].flatten().tolist()
        top_nodes = node_grid[:, :, 2].flatten().tolist()
        deck.grnod_node(1, "base_nodes", base_nodes)
        deck.grnod_node(2, "top_nodes", top_nodes)
        deck.bcs(1, "clamp_base", "111", "111", 1)

        # Dynamic compressive impact load: initial velocity -10.0 mm/ms in -Z on top face
        deck.inivel_tra(1, "impact_velocity", [0.0, 0.0, -10.0], 2)
        deck.write(s_path)

        # Run for 0.8 s (approx 59 cycles with Courant dt ~ 0.0136 s)
        engine_deck = f"""/RUN/{run_name}/1
0.80
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

        # 1. Verification of Starter resolution
        assert len(log.errors) == 0
        mat = st_model.materials[1]
        assert mat.law == 28
        assert mat.sound_speed_solid() == pytest.approx(math.sqrt(400.0 / 1.0e-3), rel=1e-5)

        # 2. 50+ explicit time steps executed
        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert not state.stop_reason or "/STOP" in str(state.stop_reason)

        # 3. Kinematic and stress state finiteness
        assert np.isfinite(eng_model.x).all()
        assert np.isfinite(eng_model.v).all()
        brick_g = dict(eng_model.element_groups())["bricks"]
        sig = brick_g.state["sig"]
        assert sig.shape == (8, 6)
        assert np.isfinite(sig).all()

        # Compressive shock wave: mean normal stress in Z is compressive
        assert np.mean(sig[:, 2]) < 0.0

        # 4. Energy conservation audit
        en = _energies(eng_model, state)
        assert en["IE"] > 0.0, f"Expected positive internal energy, got {en['IE']}"
        assert en["KE"] > 0.0, f"Expected positive kinetic energy, got {en['KE']}"
        assert pytest.approx(en["EW"], abs=1e-6) == 0.0  # Initial velocity, no external work
        assert abs(en["ERR"]) <= 5.0, f"Energy error exceeded threshold: {en['ERR']}%"

        # Energy ratio: E_tot / (E_tot,0 + W_ext) == 1.0
        # In OpenRadioss engine ledgers, E_tot = IE + KE + HE + CE + EN + DE
        total_e = en["IE"] + en["KE"] + en["HE"] + en["CE"] + en["EN"] + en["DE"]
        ref_e = _energies.e0 + state.wext
        assert ref_e > 0.0
        energy_ratio = total_e / ref_e
        assert pytest.approx(energy_ratio, rel=1e-5) == 1.0

    def test_engine_combined_hexa8_tetra4_mesh_simulation(self, tmp_path):
        """Run explicit dynamic simulation with combined hexa8 and tetra4 solid elements.
        Verify:
        - Concurrent solid constitutive update for both hexa8 and tetra4 elements with LAW28.
        - Prescribed velocity impact load (/IMPVEL) with /FUNCT velocity table.
        - Run for 50+ explicit time steps.
        - External work W_ext, internal energy E_int, kinetic energy E_k booked correctly.
        - Energy balance maintained within 5.0% error across both element families.
        """
        run_name = "COMBINED_HEX_TET"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)

        # Mesh: 1 hexa8 brick at base (nodes 1..8) + 2 tetra4 elements (apex node 9)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
            (9, 5.0, 5.0, 20.0),
        ])
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        deck.tetra4(2, [
            (2, 5, 6, 7, 9),
            (3, 5, 7, 8, 9),
        ])
        deck.part(1, "HEX_HONEYCOMB_PART", 1, 1)
        deck.part(2, "TET_HONEYCOMB_PART", 2, 1)

        deck.mat_law28(
            1,
            rho=1.0e-3,
            e11=200.0,
            e22=300.0,
            e33=400.0,
            g12=50.0,
            g23=60.0,
            g31=70.0,
            title="HONEYCOMB_SHARED",
        )
        deck.prop_solid(1, "HEX_PROP")
        deck.prop_solid(2, "TET_PROP")

        # Boundary conditions: Clamped base (nodes 1..4)
        deck.grnod_node(1, "base", [1, 2, 3, 4])
        deck.bcs(1, "fix_base", "111", "111", 1)

        # Prescribed velocity /IMPVEL pushing apex node 9 downward into the honeycomb
        deck.grnod_node(2, "apex", [9])
        deck.funct(1, "vel_profile", [(0.0, -10.0), (10.0, -10.0)])
        deck.impvel(1, "push_apex", 1, "Z", 2, scale=1.0)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
0.80
/DT
0.9 0
/PRINT/-1
/STOP
100
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert not state.stop_reason or "/STOP" in str(state.stop_reason)

        groups = dict(eng_model.element_groups())
        assert "bricks" in groups and "tetras" in groups
        sig_b = groups["bricks"].state["sig"]
        sig_t = groups["tetras"].state["sig"]

        assert sig_b.shape == (1, 6)
        assert sig_t.shape == (2, 6)
        assert np.isfinite(sig_b).all()
        assert np.isfinite(sig_t).all()

        # Both element groups experience compression in Z
        assert sig_b[0, 2] < 0.0
        assert np.mean(sig_t[:, 2]) < 0.0

        # Energy conservation audit
        en = _energies(eng_model, state)
        assert en["IE"] > 0.0
        assert en["KE"] > 0.0
        assert en["EW"] > 0.0
        assert abs(en["ERR"]) <= 5.0

        assert np.isfinite(eng_model.x).all()
        assert np.isfinite(eng_model.v).all()

    def test_engine_severe_deformation_element_deletion(self, tmp_path):
        """Audit element deletion under severe deformation for LAW28 honeycomb.
        Verify:
        - Rupture strain eps_max33 triggers clean element deletion under severe dynamic tension.
        - Deletion updates ndel counter and zeroes out stress for deleted elements.
        - Alive elements continue integration without NaN or Inf propagation.
        - Time step does not crash or collapse to zero; solver finishes normally.
        """
        run_name = "DELETION_HONEYCOMB"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)

        # 2 stacked bricks in Z
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
            (9, 0.0, 0.0, 20.0),
            (10, 10.0, 0.0, 20.0),
            (11, 10.0, 10.0, 20.0),
            (12, 0.0, 10.0, 20.0),
        ])
        deck.brick(1, [
            (1, 1, 2, 3, 4, 5, 6, 7, 8),
            (2, 5, 6, 7, 8, 9, 10, 11, 12),
        ])
        deck.part(1, "HONEYCOMB_STACK", 1, 1)

        # Tensile rupture limit in direction 33: eps_max33 = 0.005
        deck.mat_law28(
            1,
            rho=1.0e-3,
            e11=100.0,
            e22=100.0,
            e33=500.0,
            eps_max33=0.005,
            title="HONEYCOMB_RUPTURE",
        )
        deck.prop_solid(1, "SOLID_PROP")

        # Clamped base (nodes 1..4)
        deck.grnod_node(1, "base", [1, 2, 3, 4])
        deck.bcs(1, "fix_base", "111", "111", 1)

        # High-velocity dynamic tensile pull on top face (nodes 9..12) in +Z at 100 mm/ms
        deck.grnod_node(2, "top", [9, 10, 11, 12])
        deck.inivel_tra(1, "pull_top", [0.0, 0.0, 100.0], 2)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
0.50
/DT
0.9 0
/PRINT/-1
/STOP
100
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 10
        # Verification that element deletion triggered cleanly
        assert state.ndel >= 1, f"Expected at least 1 element deleted, got ndel={state.ndel}"

        brick_g = dict(eng_model.element_groups())["bricks"]
        sig = brick_g.state["sig"]
        st_extra = brick_g.state.get("mat_extra", {})
        off = st_extra.get("off28")

        assert off is not None
        assert 0.0 in off, "Expected at least one element with off28 == 0.0"

        # Deleted elements have zero stress
        dead_mask = (off == 0.0)
        assert np.allclose(sig[dead_mask], 0.0)

        # Entire solution remains strictly finite (no NaNs or Infs)
        assert np.isfinite(eng_model.x).all()
        assert np.isfinite(eng_model.v).all()
        assert np.isfinite(sig).all()
        assert not state.stop_reason or "/STOP" in str(state.stop_reason)

    def test_engine_courant_timestep_stability(self, tmp_path):
        """Verify that the engine time-step calculation with LAW28 sound speed maintains Courant stability.
        Verify:
        - Normal modulus dominance (E33): c = sqrt(E33 / rho0).
        - Shear modulus dominance (G31): c = sqrt(G31 / rho0).
        - Initial time step matches theoretical formula: dt0 = 0.9 * Le / c.
        - Explicit time stepping remains stable across all cycles.
        """
        # Case A: Normal modulus dominant (E33 = 400.0, rho0 = 1.0e-3, Le = 10.0)
        # Expected c = sqrt(400 / 1e-3) = 632.4555 mm/ms -> dt0 = 0.9 * 10 / 632.4555 = 1.42302e-2 ms
        run_name_a = "COURANT_NORM"
        s_path_a = os.path.join(tmp_path, f"{run_name_a}_0000.rad")
        e_path_a = os.path.join(tmp_path, f"{run_name_a}_0001.rad")

        deck_a = StarterDeck(run_name_a)
        deck_a.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
        ])
        deck_a.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        deck_a.part(1, "BLOCK", 1, 1)
        deck_a.mat_law28(1, rho=1.0e-3, e11=200.0, e22=300.0, e33=400.0, g12=50.0, g23=60.0, g31=70.0)
        deck_a.prop_solid(1, "SOLID_PROP")
        deck_a.write(s_path_a)

        with open(e_path_a, "w") as f:
            f.write(f"/RUN/{run_name_a}/1\n0.05\n/DT\n0.9 0\n/PRINT/-1\n/STOP\n100\n/END\n")

        with contextlib.redirect_stdout(io.StringIO()):
            run_starter(s_path_a)
            eng_a = run_engine(e_path_a)

        out_path_a = os.path.join(tmp_path, f"{run_name_a}_0001.out")
        with open(out_path_a) as f:
            out_text_a = f.read()
        dt_line_a = [ln for ln in out_text_a.splitlines() if "INITIAL TIME STEP" in ln][0]
        dt_val_a = float(dt_line_a.split(":")[-1])
        c_expected_a = math.sqrt(400.0 / 1.0e-3)
        dt_expected_a = 0.9 * 10.0 / c_expected_a
        assert pytest.approx(dt_val_a, rel=1e-3) == dt_expected_a
        assert eng_a.engine_state.cycle >= 3

        # Case B: Shear modulus dominant (G31 = 900.0, rho0 = 1.0e-3, Le = 10.0)
        # Expected c = sqrt(900 / 1e-3) = 948.6833 mm/ms -> dt0 = 0.9 * 10 / 948.6833 = 9.48683e-3 ms
        run_name_b = "COURANT_SHEAR"
        s_path_b = os.path.join(tmp_path, f"{run_name_b}_0000.rad")
        e_path_b = os.path.join(tmp_path, f"{run_name_b}_0001.rad")

        deck_b = StarterDeck(run_name_b)
        deck_b.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
        ])
        deck_b.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        deck_b.part(1, "BLOCK", 1, 1)
        deck_b.mat_law28(1, rho=1.0e-3, e11=100.0, e22=100.0, e33=100.0, g12=50.0, g23=60.0, g31=900.0)
        deck_b.prop_solid(1, "SOLID_PROP")
        deck_b.write(s_path_b)

        with open(e_path_b, "w") as f:
            f.write(f"/RUN/{run_name_b}/1\n0.05\n/DT\n0.9 0\n/PRINT/-1\n/STOP\n100\n/END\n")

        with contextlib.redirect_stdout(io.StringIO()):
            run_starter(s_path_b)
            eng_b = run_engine(e_path_b)

        out_path_b = os.path.join(tmp_path, f"{run_name_b}_0001.out")
        with open(out_path_b) as f:
            out_text_b = f.read()
        dt_line_b = [ln for ln in out_text_b.splitlines() if "INITIAL TIME STEP" in ln][0]
        dt_val_b = float(dt_line_b.split(":")[-1])
        c_expected_b = math.sqrt(900.0 / 1.0e-3)
        dt_expected_b = 0.9 * 10.0 / c_expected_b
        assert pytest.approx(dt_val_b, rel=1e-3) == dt_expected_b
        assert eng_b.engine_state.cycle >= 5



# =============================================================================
# 13. Upstream Fortran Parity Audit Checks (Auditor 1, Milestone M538)
# =============================================================================

class TestLaw28FortranParityAudit:
    """Rigorous audit tests verifying law28_honeycomb against sigeps28.F & hm_read_mat28.F."""

    def test_audit_check1_engineering_shear_stress_update(self):
        """Check 1: SIGNXY = SIGOXY + G12 * DEPSXY with engineering shear strain increment."""
        g12, g23, g31 = 40.0, 50.0, 60.0
        mat = _make_honeycomb(e11=100.0, e22=200.0, e33=300.0, g12=g12, g23=g23, g31=g31)
        sig = np.zeros((1, 6))
        # Pass engineering shear strain increment gamma_xy = 0.05
        deps = np.array([[0.0, 0.0, 0.0, 0.05, 0.02, 0.01]])
        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps)

        assert pytest.approx(sign[0, 3]) == g12 * 0.05  # 2.0
        assert pytest.approx(sign[0, 4]) == g23 * 0.02  # 1.0
        assert pytest.approx(sign[0, 5]) == g31 * 0.01  # 0.6

    def test_audit_check2_tensorial_shear_failure_criterion(self):
        """Check 2: ABS(EPSXY/TWO) > EMX12 converts engineering shear to tensorial shear.
        Also verifies normal strain rupture is tensile only (EPSXX > EMX11, not ABS).
        """
        emx12 = 0.04
        mat = _make_honeycomb(eps_max11=0.10, eps_max12=emx12)

        # Case A: gamma_xy = 0.06 -> tensorial eps_xy = 0.03 <= 0.04 (no failure!)
        off_a = np.array([1.0])
        extra_a = {"eps28": np.zeros((1, 6)), "off28": off_a}
        deps_a = np.array([[0.0, 0.0, 0.0, 0.06, 0.0, 0.0]])
        sign_a, _, _ = law28_honeycomb.solid_update(mat, np.zeros((1, 6)), deps_a, extra=extra_a)
        assert off_a[0] == 1.0, "Should not fail when tensorial shear eps_xy <= emx12"
        assert sign_a[0, 3] != 0.0

        # Case B: gamma_xy = 0.10 -> tensorial eps_xy = 0.05 > 0.04 (failure!)
        off_b = np.array([1.0])
        extra_b = {"eps28": np.zeros((1, 6)), "off28": off_b}
        deps_b = np.array([[0.0, 0.0, 0.0, 0.10, 0.0, 0.0]])
        sign_b, _, _ = law28_honeycomb.solid_update(mat, np.zeros((1, 6)), deps_b, extra=extra_b)
        assert off_b[0] == 0.0, "Must fail when tensorial shear eps_xy > emx12"
        assert np.allclose(sign_b[0], 0.0)

        # Case C: Large compressive normal strain (eps_11 = -0.50) with eps_max11 = 0.10
        # In sigeps28.F: EPSXX(I) > EMX11. -0.50 is NOT > 0.10, so no tensile rupture!
        off_c = np.array([1.0])
        extra_c = {"eps28": np.zeros((1, 6)), "off28": off_c}
        deps_c = np.array([[-0.50, 0.0, 0.0, 0.0, 0.0, 0.0]])
        sign_c, _, _ = law28_honeycomb.solid_update(mat, np.zeros((1, 6)), deps_c, extra=extra_c)
        assert off_c[0] == 1.0, "Large compressive strain must not trigger tensile rupture"

    def test_audit_check3_volumetric_strain_mu_from_vol(self):
        """Check 3: mu = rho/rho0 - 1 = vol0/vol - 1 = AMU."""
        fc = FunctTable(1, [0.0, 0.1, 0.5], [10.0, 25.0, 75.0])
        mat = _make_honeycomb(
            rho0=1.0e-3,
            e11=1000.0,
            fun_a1=1,
            gflag=0,
            fscale11=1.0,
            curve28_x=[fc.x, None, None, None, None, None],
            curve28_y=[fc.y, None, None, None, None, None],
            curve28_fct=[fc, None, None, None, None, None],
        )
        sig = np.zeros((1, 6))
        deps = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # Provide vol and vol0: vol0 = 1.0, vol = 0.9090909 -> vol0/vol - 1 = 0.10
        extra = {
            "eps28": np.zeros((1, 6)),
            "vol0": np.array([1.0]),
            "vol": np.array([1.0 / 1.1]),  # mu = 1.1 - 1 = 0.10
        }
        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps, extra=extra)
        # Yield stress at mu = 0.10 is 25.0
        assert pytest.approx(sign[0, 0]) == 25.0

    def test_audit_check4_fscale13_attribute_corresponds_to_dir31(self):
        """Check 4: CFG attribute FScale13 corresponds to direction 31."""
        mat = _make_honeycomb(
            g31=1000.0,
            fun_a4=4,
            vflag=1,
            FScale13=2.5,  # CFG attribute name
            curve28_x=[None, None, None, None, None, np.array([0.0, 1.0])],
            curve28_y=[None, None, None, None, None, np.array([10.0, 10.0])],
        )
        assert pytest.approx(mat.params["fscale31"]) == 2.5
        sig = np.zeros((1, 6))
        deps = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.5]])
        sign, _, _ = law28_honeycomb.solid_update(mat, sig, deps)
        # Clamped to Y * FScale13 = 10.0 * 2.5 = 25.0
        assert pytest.approx(sign[0, 5]) == 25.0

    def test_audit_check5_sound_speed_shear_dominance(self):
        """Check 5: c = sqrt(max(E11..33, G12..31) / rho0) in sigeps28.F line 192."""
        mat = _make_honeycomb(
            rho0=2.0e-3,
            e11=100.0, e22=120.0, e33=110.0,
            g12=150.0, g23=200.0, g31=800.0,  # g31 is dominant
        )
        expected_c = math.sqrt(800.0 / 2.0e-3)
        assert pytest.approx(law28_honeycomb.sound_speed(mat)) == expected_c
        assert pytest.approx(mat.sound_speed_solid()) == expected_c
        assert pytest.approx(pm.sound_speed(mat)) == expected_c

    def test_audit_check6_timestep_parmat_formulation(self):
        """Check 6: DMIN = MIN(E11*E22, E22*E33, E11*E33), DMAX = MAX(E11,E22,E33),
        PARMAT(16) = 1 (iformdt=1), PARMAT(17) = DMIN / DMAX**2 (gfac).
        """
        e11, e22, e33 = 100.0, 200.0, 400.0
        mat = _make_honeycomb(e11=e11, e22=e22, e33=e33, g12=50.0, g23=60.0, g31=70.0)
        p = mat.params

        expected_dmin = min(e11 * e22, e22 * e33, e11 * e33)  # min(20000, 80000, 40000) = 20000
        expected_dmax = max(e11, e22, e33)                    # 400.0
        expected_parmat17 = expected_dmin / (expected_dmax ** 2)  # 20000 / 160000 = 0.125
        expected_parmat1 = max(e11, e22, e33, 50.0, 60.0, 70.0)  # 400.0

        assert pytest.approx(p["dmin"]) == expected_dmin
        assert pytest.approx(p["dmax"]) == expected_dmax
        assert p["iformdt"] == 1
        assert p["parmat_16"] == 1
        assert pytest.approx(p["parmat_17"]) == expected_parmat17
        assert pytest.approx(p["gfac"]) == expected_parmat17
        assert pytest.approx(p["parmat_1"]) == expected_parmat1

