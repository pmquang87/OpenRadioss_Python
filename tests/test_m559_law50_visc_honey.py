"""Comprehensive unit test suite for Milestone M559: /MAT/LAW50 (/MAT/VISC_HONEY, /MAT/HYP_FOAM).

Rate-Dependent Viscoelastic Honeycomb Material Model.
Upstream Fortran origins:
- ``starter/source/materials/mat/mat050/hm_read_mat50.F90`` (card parsing, defaults, parameters)
- ``engine/source/materials/mat/mat050/sigeps50s.F90`` (constitutive update, filtering, compaction, failure)
- ``C:\\OpenRadioss\\hm_cfg_files\\config\\CFG\\radioss2025\\MAT\\mat_law50.cfg`` (CFG schema)

Covers:
1. Schema & Synonym Registration (CfgCatalogue, /MAT/LAW50, /MAT/VISC_HONEY, /MAT/HYP_FOAM)
2. Deck Writer & Card Parsing Roundtrip (StarterDeck.mat_law50, mat_visc_honey)
3. Orthotropic Elastic Response in All 6 Directions (11, 22, 33, 12, 23, 31)
4. Normal Yield Function Clamping with Gflag in {0, 1, -1}
5. Shear Yield Function Clamping with Vflag in {0, 1, -1}
6. Strain Rate Filtering & Multi-Curve Interpolation (Irate=1 and Irate=2)
7. Tensile & Shear Failure Criteria and Element Deletion (eps_max11..33, eps_max12..31)
8. Compaction Transition & Compacted J2 Plasticity Radial Return
9. Sound Speed Calculation in Uncompacted and Compacted States
10. Consistent Solid Algorithmic Tangent vs Numerical Finite Differences
11. Shell Update Rejection Guard (NotImplementedError)
12. Curve Resolution Hook (/FUNCT to FunctTable binding)
13. Vectorized Multi-Element Batch Evaluation
"""

from __future__ import annotations

import math
import os
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law50_visc_honey
import pyradioss.materials as pm
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY, catalogue
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable


# =============================================================================
# Helpers & Builders
# =============================================================================

def _make_law50(
    rho0: float = 1.0e-3,
    refer_rho: float = 0.0,
    ea: float = 100.0,
    eb: float = 200.0,
    ec: float = 300.0,
    gab: float = 40.0,
    gbc: float = 50.0,
    gca: float = 60.0,
    asrate: float = 0.0,
    gflag: int = 0,
    vflag: int = 0,
    irate: int = 2,
    eps_max11: float = 1e30,
    eps_max22: float = 1e30,
    eps_max33: float = 1e30,
    eps_max12: float = 1e30,
    eps_max23: float = 1e30,
    eps_max31: float = 1e30,
    yfun11=None,
    sfac11=None,
    eps11=None,
    yfun22=None,
    sfac22=None,
    eps22=None,
    yfun33=None,
    sfac33=None,
    eps33=None,
    yfun12=None,
    sfac12=None,
    eps12=None,
    yfun23=None,
    sfac23=None,
    eps23=None,
    yfun31=None,
    sfac31=None,
    eps31=None,
    ecomp: float = 0.0,
    et: float = 0.0,
    sigy: float = 0.0,
    pr: float = 0.0,
    vcomp: float = 0.0,
    **kw,
) -> Material:
    """Helper to build a Material configured for LAW50."""
    rec = {
        "id": 1,
        "title": "LAW50_TEST",
        "params": {
            "MAT_RHO": rho0,
            "Refer_Rho": refer_rho,
            "MAT_EA": ea,
            "MAT_EB": eb,
            "MAT_EC": ec,
            "MAT_GAB": gab,
            "MAT_GBC": gbc,
            "MAT_GCA": gca,
            "MAT_asrate": asrate,
            "Gflag": gflag,
            "Vflag": vflag,
            "Irate": irate,
            "MAT_EPS_max11": eps_max11,
            "MAT_EPS_max22": eps_max22,
            "MAT_EPS_max33": eps_max33,
            "MAT_EPS_max12": eps_max12,
            "MAT_EPS_max23": eps_max23,
            "MAT_EPS_max31": eps_max31,
            "MAT_ECOMP": ecomp,
            "MAT_ET": et,
            "MAT_SIGY": sigy,
            "MAT_PR": pr,
            "MAT_VCOMP": vcomp,
            **kw,
        },
    }
    # Populate curve lists if provided
    for dir_name, yf, sf, ep in [
        ("11", yfun11, sfac11, eps11),
        ("22", yfun22, sfac22, eps22),
        ("33", yfun33, sfac33, eps33),
        ("12", yfun12, sfac12, eps12),
        ("23", yfun23, sfac23, eps23),
        ("31", yfun31, sfac31, eps31),
    ]:
        if yf is not None:
            for i, val in enumerate(yf[:5]):
                rec["params"][f"MAT_YFUN{dir_name}_{i+1}"] = val
        if sf is not None:
            for i, val in enumerate(sf[:5]):
                rec["params"][f"MAT_SFAC{dir_name}_{i+1}"] = val
        if ep is not None:
            for i, val in enumerate(ep[:5]):
                rec["params"][f"MAT_EPS{dir_name}_{i+1}"] = val

    return law50_visc_honey.build_law50(rec)


# =============================================================================
# 1. Schema & Registry Tests
# =============================================================================

class TestLaw50RegistryAndSchema:
    """Verify registration, synonyms, and CFG catalogue mapping."""

    def test_registry_registration(self):
        pm.register_materials()
        for key in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
            assert key in MAT_PHYSICS_REGISTRY, f"{key} missing in MAT_PHYSICS_REGISTRY"
            fn = MAT_PHYSICS_REGISTRY[key]
            assert callable(fn)

    def test_synonym_lookup(self):
        cat = catalogue()
        schema_visc = cat.schema("VISC_HONEY")
        assert schema_visc is not None, "VISC_HONEY schema missing"
        assert schema_visc.law_number == 50
        assert "VISC_HONEY" in schema_visc.law_names
        assert "LAW50" in schema_visc.law_names

    def test_dataclass_creation(self):
        p = law50_visc_honey.Law50Params(
            rho0=1.2e-3, ea=100.0, eb=200.0, ec=300.0,
            gab=40.0, gbc=50.0, gca=60.0,
            ecomp=500.0, pr=0.25, sigy=50.0, et=10.0, vcomp=0.2,
        )
        assert p.icompact == 1
        assert pytest.approx(p.gcomp) == 500.0 / 1.25
        assert pytest.approx(p.bulk) == 500.0 / (3.0 * (1.0 - 2.0 * 0.25))
        assert pytest.approx(p.E) == 500.0
        assert pytest.approx(p.G) == 200.0

        p_uncomp = law50_visc_honey.Law50Params(
            rho0=1.2e-3, ea=100.0, eb=200.0, ec=300.0,
            gab=40.0, gbc=50.0, gca=60.0,
        )
        assert p_uncomp.icompact == 0
        assert pytest.approx(p_uncomp.E) == 300.0
        assert pytest.approx(p_uncomp.G) == 60.0


# =============================================================================
# 2. Deck Writer & Card Parsing Roundtrip
# =============================================================================

class TestLaw50DeckWriterAndParsing:
    """Verify StarterDeck.mat_law50 and roundtrip card parsing."""

    def test_deck_writer_roundtrip(self, tmp_path):
        deck = StarterDeck("LAW50_ROUNDTRIP")
        deck.mat_law50(
            mat_id=10,
            rho=0.002,
            ea=150.0,
            eb=250.0,
            ec=350.0,
            gab=45.0,
            gbc=55.0,
            gca=65.0,
            asrate=100.0,
            gflag=1,
            eps_max11=0.25,
            eps_max22=0.30,
            eps_max33=0.35,
            yfun11=[101, 102],
            sfac11=[1.5, 2.0],
            eps11=[0.0, 10.0],
            vflag=-1,
            eps_max12=0.40,
            eps_max23=0.45,
            eps_max31=0.50,
            yfun12=[201],
            sfac12=[1.2],
            eps12=[0.0],
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
        assert m.law == 50
        p = m.params
        assert pytest.approx(p["rho0"]) == 0.002
        assert pytest.approx(p["ea"]) == 150.0
        assert pytest.approx(p["eb"]) == 250.0
        assert pytest.approx(p["ec"]) == 350.0
        assert pytest.approx(p["gab"]) == 45.0
        assert pytest.approx(p["gbc"]) == 55.0
        assert pytest.approx(p["gca"]) == 65.0
        assert pytest.approx(p["asrate"]) == 100.0
        assert p["gflag"] == 1
        assert p["vflag"] == -1
        assert pytest.approx(p["eps_max11"]) == 0.25
        assert pytest.approx(p["eps_max22"]) == 0.30
        assert pytest.approx(p["eps_max33"]) == 0.35
        assert pytest.approx(p["eps_max12"]) == 0.40
        assert pytest.approx(p["eps_max23"]) == 0.45
        assert pytest.approx(p["eps_max31"]) == 0.50
        assert p["yfun11"][0] == 101
        assert p["yfun11"][1] == 102
        assert pytest.approx(p["sfac11"][0]) == 1.5
        assert pytest.approx(p["eps11"][1]) == 10.0
        assert p["yfun12"][0] == 201
        assert pytest.approx(p["sfac12"][0]) == 1.2


# =============================================================================
# 3. Orthotropic Elastic Response in All 6 Directions
# =============================================================================

class TestLaw50OrthotropicElasticity:
    """Verify uncoupled orthotropic elastic response in all 6 directions."""

    def test_elastic_normal_directions(self):
        mat = _make_law50(ea=100.0, eb=200.0, ec=300.0, gab=40.0, gbc=50.0, gca=60.0)
        n = 3
        sig = np.zeros((n, 6))
        deps = np.zeros((n, 6))

        # Element 0: pure eps11
        deps[0, 0] = 0.01
        # Element 1: pure eps22
        deps[1, 1] = 0.02
        # Element 2: pure eps33
        deps[2, 2] = 0.03

        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps)

        # Expected: sigma_11 = 100 * 0.01 = 1.0, others zero
        assert pytest.approx(sign[0, 0]) == 1.0
        assert np.allclose(sign[0, 1:], 0.0)

        # Expected: sigma_22 = 200 * 0.02 = 4.0, others zero
        assert pytest.approx(sign[1, 1]) == 4.0
        assert np.allclose(sign[1, [0, 2, 3, 4, 5]], 0.0)

        # Expected: sigma_33 = 300 * 0.03 = 9.0, others zero
        assert pytest.approx(sign[2, 2]) == 9.0
        assert np.allclose(sign[2, [0, 1, 3, 4, 5]], 0.0)

    def test_elastic_shear_directions(self):
        mat = _make_law50(ea=100.0, eb=200.0, ec=300.0, gab=40.0, gbc=50.0, gca=60.0)
        n = 3
        sig = np.zeros((n, 6))
        deps = np.zeros((n, 6))

        # Element 0: pure shear 12
        deps[0, 3] = 0.01
        # Element 1: pure shear 23
        deps[1, 4] = 0.02
        # Element 2: pure shear 31
        deps[2, 5] = 0.03

        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps)

        # In OpenRadioss engine sigeps50s.F90:
        # sig12 = sig12 + G12 * deps12
        assert pytest.approx(sign[0, 3]) == 40.0 * 0.01
        assert np.allclose(sign[0, [0, 1, 2, 4, 5]], 0.0)

        assert pytest.approx(sign[1, 4]) == 50.0 * 0.02
        assert np.allclose(sign[1, [0, 1, 2, 3, 5]], 0.0)

        assert pytest.approx(sign[2, 5]) == 60.0 * 0.03
        assert np.allclose(sign[2, [0, 1, 2, 3, 4]], 0.0)

    def test_elastic_unloading_reversible(self):
        mat = _make_law50(ea=100.0, eb=200.0, ec=300.0, gab=40.0, gbc=50.0, gca=60.0)
        n = 1
        sig = np.zeros((n, 6))
        extra = {}

        # Loading step
        deps_load = np.array([[0.005, 0.004, 0.003, 0.002, 0.001, 0.0005]])
        sig1, _, _ = law50_visc_honey.solid_update(mat, sig, deps_load, extra=extra)

        # Unloading step with exact opposite increment
        deps_unload = -deps_load
        sig2, _, _ = law50_visc_honey.solid_update(mat, sig1, deps_unload, extra=extra)

        # Stress should return to 0
        assert np.allclose(sig2, 0.0, atol=1e-12)


# =============================================================================
# 4. Normal Yield Clamping with Gflag
# =============================================================================

class TestLaw50NormalYieldClamping:
    """Verify normal yield clamping with gflag in {0, 1, -1}."""

    def test_normal_yield_direct_strain_gflag_1(self):
        tbl11 = FunctTable(11, x=np.array([0.0, 1.0]), y=np.array([5.0, 5.0]))
        mat = _make_law50(ea=100.0, eb=100.0, ec=100.0, gflag=1)
        mat.params["tables"] = [tbl11, None, None, None, None, None]

        sig = np.zeros((1, 6))
        # Elastic trial would be 100 * 0.1 = 10.0 > 5.0
        deps = np.zeros((1, 6))
        deps[0, 0] = 0.1

        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps)
        assert pytest.approx(sign[0, 0]) == 5.0

        # Symmetrical compressive clamping
        deps[0, 0] = -0.1
        sign_c, _, _ = law50_visc_honey.solid_update(mat, sig, deps)
        assert pytest.approx(sign_c[0, 0]) == -5.0

    def test_normal_yield_volumetric_strain_gflag_0(self):
        tbl = FunctTable(10, x=np.array([0.0, 0.05, 0.10]), y=np.array([2.0, 4.0, 8.0]))
        mat = _make_law50(ea=100.0, eb=100.0, ec=100.0, gflag=0)
        mat.params["tables"] = [tbl, tbl, tbl, None, None, None]

        sig = np.zeros((1, 6))
        # Pass volumetric compression mu = 0.10 -> Yield at mu=0.10 is 8.0
        deps = np.array([[0.02, 0.03, 0.05, 0.0, 0.0, 0.0]])
        extra = {"amu": np.array([0.10])}
        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps, extra=extra)
        assert pytest.approx(sign[0, 0]) == 2.0
        assert pytest.approx(sign[0, 1]) == 3.0
        assert pytest.approx(sign[0, 2]) == 5.0

        # Now apply a huge increment to exceed yield
        deps2 = np.array([[0.1, 0.1, 0.1, 0.0, 0.0, 0.0]])
        sign2, _, _ = law50_visc_honey.solid_update(mat, sign, deps2, extra=extra)
        assert sign2[0, 0] > 0.0

    def test_normal_yield_eng_volumetric_gflag_minus_1(self):
        # gflag = -1: measure is negative strain
        tbl = FunctTable(10, x=np.array([-1.0, 0.0, 1.0]), y=np.array([3.0, 3.0, 3.0]))
        mat = _make_law50(ea=100.0, eb=100.0, ec=100.0, gflag=-1)
        mat.params["tables"] = [tbl, None, None, None, None, None]

        sig = np.zeros((1, 6))
        deps = np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]])
        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps)
        # Trial 20.0 clamped to 3.0
        assert pytest.approx(sign[0, 0]) == 3.0


# =============================================================================
# 5. Shear Yield Clamping with Vflag
# =============================================================================

class TestLaw50ShearYieldClamping:
    """Verify shear yield clamping with vflag in {0, 1, -1}."""

    def test_shear_yield_direct_strain_vflag_1(self):
        tbl12 = FunctTable(12, x=np.array([0.0, 1.0]), y=np.array([2.5, 2.5]))
        mat = _make_law50(gab=50.0, gbc=50.0, gca=50.0, vflag=1)
        mat.params["tables"] = [None, None, None, tbl12, None, None]

        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        # deps12 = 0.1 -> trial = 50 * 0.1 = 5.0 > 2.5
        deps[0, 3] = 0.1

        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps)
        assert pytest.approx(sign[0, 3]) == 2.5

    def test_shear_yield_volumetric_strain_vflag_0(self):
        tbl12 = FunctTable(12, x=np.array([0.0, 0.2]), y=np.array([1.0, 4.0]))
        mat = _make_law50(gab=50.0, vflag=0)
        mat.params["tables"] = [None, None, None, tbl12, None, None]

        sig = np.zeros((1, 6))
        # Pass volumetric strain amu = 0.1 directly
        extra = {"amu": np.array([0.1])}
        deps = np.zeros((1, 6))
        deps[0, 3] = 0.1  # Trial 50 * 0.1 = 5.0 > 2.5
        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps, extra=extra)
        assert pytest.approx(sign[0, 3]) == 2.5


# =============================================================================
# 6. Strain Rate Filtering & Multi-Curve Interpolation
# =============================================================================

class TestLaw50RateFilteringAndTables:
    """Verify rate filtering and rate-dependent table lookup."""

    def test_rate_filtering_asrate(self):
        mat = _make_law50(asrate=100.0, irate=2)
        sig = np.zeros((1, 6))
        deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
        dt = 0.005
        extra = {}

        law50_visc_honey.solid_update(mat, sig, deps, dt=dt, extra=extra)
        # Initial filtered rate was 0.0. Current raw rate = 0.01 / 0.005 = 2.0.
        # Filtered rate = 0.5 * 2.0 + 0.5 * 0.0 = 1.0.
        uvar50 = extra["uvar50"]
        assert pytest.approx(uvar50[0, 0]) == 1.0

        # Next step with same raw rate:
        law50_visc_honey.solid_update(mat, sig, deps, dt=dt, extra=extra)
        # Filtered rate = 0.5 * 2.0 + 0.5 * 1.0 = 1.5.
        assert pytest.approx(extra["uvar50"][0, 0]) == 1.5

    def test_rate_dependent_table_interpolation(self):
        c1 = FunctTable(1, x=np.array([0.0, 1.0]), y=np.array([10.0, 10.0]))
        c2 = FunctTable(2, x=np.array([0.0, 1.0]), y=np.array([20.0, 20.0]))

        mat = _make_law50(ea=1000.0, asrate=0.0, irate=2, gflag=1)
        mat.params["tables"] = [
            [(0.0, 1.0, c1), (100.0, 1.0, c2)],
            None, None, None, None, None
        ]

        sig = np.zeros((1, 6))
        # dt = 0.01, deps = 0.5 -> strain rate = 50.0 (halfway between 0 and 100)
        # Interpolated yield = 15.0
        deps = np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps, dt=0.01)
        assert pytest.approx(sign[0, 0]) == 15.0

    def test_common_equivalent_strain_rate_irate_1(self):
        mat = _make_law50(asrate=1000.0, irate=1)
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        deps[0, 3] = 0.02
        dt = 0.01
        extra = {}

        law50_visc_honey.solid_update(mat, sig, deps, dt=dt, extra=extra)
        rates = extra["uvar50"][0]
        # In sigeps50s.F90 line 257: uvar(i,1) = epsd(i) = sqrt(0.5 * 2.0^2) = sqrt(2.0)
        assert pytest.approx(rates[0]) == math.sqrt(2.0)


# =============================================================================
# 7. Tensile & Shear Failure Criteria and Deletion
# =============================================================================

class TestLaw50FailureCriteria:
    """Verify tensile and shear failure strain criteria and stress zeroing."""

    def test_normal_tensile_failure(self):
        mat = _make_law50(ea=100.0, eps_max11=0.05)
        sig = np.zeros((1, 6))
        extra = {}

        # Step 1: eps11 = 0.03 < 0.05 -> intact (off=1.0)
        deps1 = np.array([[0.03, 0.0, 0.0, 0.0, 0.0, 0.0]])
        sign1, _, _ = law50_visc_honey.solid_update(mat, sig, deps1, extra=extra)
        assert pytest.approx(sign1[0, 0]) == 3.0
        assert extra["off50"][0] == 1.0

        # Step 2: eps11 becomes 0.03 + 0.03 = 0.06 > 0.05 -> failed (off=0.0)!
        deps2 = np.array([[0.03, 0.0, 0.0, 0.0, 0.0, 0.0]])
        sign2, _, _ = law50_visc_honey.solid_update(mat, sign1, deps2, extra=extra)
        assert np.allclose(sign2, 0.0)
        assert extra["off50"][0] == 0.0

    def test_shear_failure(self):
        mat = _make_law50(gab=50.0, eps_max12=0.04)
        sig = np.zeros((1, 6))
        extra = {}

        # deps12 = 0.10 -> engineering shear strain = 0.10 / 2 = 0.05 > 0.04 -> failure (off=0.0)
        deps = np.zeros((1, 6))
        deps[0, 3] = 0.10

        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps, extra=extra)
        assert np.allclose(sign, 0.0)
        assert extra["off50"][0] == 0.0

    def test_failed_element_stays_failed(self):
        mat = _make_law50(ea=100.0, eps_max11=0.02)
        sig = np.zeros((1, 6))
        # off50 = 0.0 indicates a previously deleted/failed element
        extra = {"off50": np.array([0.0])}

        deps = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps, extra=extra)
        assert np.allclose(sign, 0.0)
        assert extra["off50"][0] == 0.0


# =============================================================================
# 8. Compaction Transition & Compacted J2 Plasticity
# =============================================================================

class TestLaw50CompactionAndJ2Plasticity:
    """Verify compaction transition and compacted J2 plasticity."""

    def test_compaction_moduli_interpolation(self):
        mat = _make_law50(
            ea=100.0, eb=100.0, ec=100.0,
            gab=40.0, gbc=40.0, gca=40.0,
            ecomp=1000.0, pr=0.25, sigy=50.0, et=0.0, vcomp=0.2,
        )
        sig = np.zeros((1, 6))
        extra = {}
        # Compressive strain: deps < 0 -> volumetric compression mu > 0 -> rvol < 1 -> beta > 0
        deps = np.array([[-0.1, -0.2, -0.2, 0.0, 0.0, 0.0]])
        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps, extra=extra)
        assert extra["compacted"][0] == 0.0
        # Stiffened moduli should yield higher absolute stress than base 100 * 0.1 = 10.0
        assert abs(sign[0, 0]) > 100.0 * 0.1

    def test_full_compaction_trigger_and_latch(self):
        mat = _make_law50(
            ea=100.0, eb=100.0, ec=100.0,
            gab=40.0, gbc=40.0, gca=40.0,
            ecomp=1000.0, pr=0.25, sigy=100.0, et=10.0, vcomp=0.5,
        )
        sig = np.zeros((1, 6))
        extra = {}
        # Large compressive volumetric strain: deps_v = -1.5 -> mu = 1.5 -> rvol = 1/2.5 = 0.4 <= 0.5
        deps = np.array([[-0.5, -0.5, -0.5, 0.0, 0.0, 0.0]])
        sign, _, _ = law50_visc_honey.solid_update(mat, sig, deps, extra=extra)
        assert extra["compacted"][0] == 1.0

        # Subsequent step with expansion should KEEP compacted = 1
        deps2 = np.array([[0.1, 0.1, 0.1, 0.0, 0.0, 0.0]])
        sign2, _, _ = law50_visc_honey.solid_update(mat, sign, deps2, extra=extra)
        assert extra["compacted"][0] == 1.0

    def test_compacted_j2_plasticity_radial_return(self):
        mat = _make_law50(
            ea=100.0, eb=100.0, ec=100.0,
            gab=40.0, gbc=40.0, gca=40.0,
            ecomp=1000.0, pr=0.0, sigy=20.0, et=10.0, vcomp=0.8,
        )
        extra = {
            "compacted": np.array([1.0]),
            "eps50": np.array([[0.5, 0.5, 0.5, 0.0, 0.0, 0.0]]),
            "uvar50": np.zeros((1, 6)),
            "off50": np.array([1.0]),
        }
        sig = np.zeros((1, 6))

        deps = np.zeros((1, 6))
        deps[0, 3] = 0.1

        sign, epsp, _ = law50_visc_honey.solid_update(mat, sig, deps, extra=extra)

        s12 = sign[0, 3]
        von_mises = math.sqrt(3.0 * s12 * s12)
        assert epsp[0] > 0.0
        # In sigeps50s.F90 lines 390-391: yld = sigy + hcomp * eplas_old, stress is clamped to yld
        assert pytest.approx(von_mises, rel=1e-5) == 20.0


# =============================================================================
# 9. Sound Speed Calculation
# =============================================================================

class TestLaw50SoundSpeed:
    """Verify acoustic sound speed calculation in uncompacted and compacted regimes."""

    def test_uncompacted_sound_speed(self):
        mat = _make_law50(rho0=2.0e-3, ea=200.0, eb=300.0, ec=500.0, gab=100.0, gbc=120.0, gca=150.0)
        c = law50_visc_honey.sound_speed_solid(mat)
        assert pytest.approx(c) == 500.0

    def test_compacted_sound_speed(self):
        mat = _make_law50(
            rho0=2.0e-3, ea=100.0, eb=100.0, ec=100.0,
            ecomp=1000.0, pr=0.25, sigy=50.0, vcomp=0.2,
        )
        # In sigeps50s.F90: SOUNDSP = SQRT(MAX(E11,E22,E33,G12,G23,G31)/RHO)
        # When compacted: ecomp=1000.0, gcomp = 1000 / 1.25 = 800.0 -> max is 1000.0
        # c = sqrt(1000.0 / 0.002) = sqrt(500000)
        c = law50_visc_honey.sound_speed_solid(mat, compacted=True)
        assert pytest.approx(c) == math.sqrt(1000.0 / 0.002)

    def test_sound_speed_dispatch_via_pm(self):
        mat = _make_law50(rho0=1.0e-3, ea=400.0)
        c = pm.sound_speed(mat)
        assert pytest.approx(c) == math.sqrt(400.0 / 1.0e-3)


# =============================================================================
# 10. Consistent Solid Tangent vs Finite Differences
# =============================================================================

class TestLaw50ConsistentTangent:
    """Verify algorithmic consistent tangent matches numerical perturbation."""

    def test_elastic_tangent_comparison(self):
        mat = _make_law50(ea=150.0, eb=250.0, ec=350.0, gab=45.0, gbc=55.0, gca=65.0)
        sig = np.zeros((1, 6))
        c_algo = law50_visc_honey.consistent_solid_tangent(mat, sig)
        assert c_algo.shape == (1, 6, 6)

        assert pytest.approx(c_algo[0, 0, 0]) == 150.0
        assert pytest.approx(c_algo[0, 1, 1]) == 250.0
        assert pytest.approx(c_algo[0, 2, 2]) == 350.0
        assert pytest.approx(c_algo[0, 3, 3]) == 45.0
        assert pytest.approx(c_algo[0, 4, 4]) == 55.0
        assert pytest.approx(c_algo[0, 5, 5]) == 65.0

        # Numerical perturbation check
        h = 1e-7
        c_num = np.zeros((6, 6))
        for j in range(6):
            deps_p = np.zeros((1, 6))
            deps_m = np.zeros((1, 6))
            deps_p[0, j] = h
            deps_m[0, j] = -h
            sig_p, _, _ = law50_visc_honey.solid_update(mat, sig, deps_p)
            sig_m, _, _ = law50_visc_honey.solid_update(mat, sig, deps_m)
            c_num[:, j] = (sig_p[0] - sig_m[0]) / (2.0 * h)

        assert np.allclose(c_algo[0], c_num, atol=1e-5)

    def test_compacted_elastic_tangent(self):
        mat = _make_law50(
            ea=100.0, eb=100.0, ec=100.0,
            ecomp=1000.0, pr=0.25, sigy=1e10, vcomp=0.5,
        )
        extra = {"compacted": np.array([1.0])}
        sig = np.zeros((1, 6))
        c_algo = law50_visc_honey.consistent_solid_tangent(mat, sig, extra=extra)

        h = 1e-7
        c_num = np.zeros((6, 6))
        for j in range(6):
            deps_p = np.zeros((1, 6))
            deps_m = np.zeros((1, 6))
            deps_p[0, j] = h
            deps_m[0, j] = -h
            extra_p = {"compacted": np.array([1.0])}
            extra_m = {"compacted": np.array([1.0])}
            sig_p, _, _ = law50_visc_honey.solid_update(mat, sig, deps_p, extra=extra_p)
            sig_m, _, _ = law50_visc_honey.solid_update(mat, sig, deps_m, extra=extra_m)
            c_num[:, j] = (sig_p[0] - sig_m[0]) / (2.0 * h)

        assert np.allclose(c_algo[0], c_num, atol=1e-4)


# =============================================================================
# 11. Shell Update Rejection Guard
# =============================================================================

class TestLaw50ShellGuard:
    """Verify shell_update raises NotImplementedError (solid-only law)."""

    def test_shell_update_raises(self):
        mat = _make_law50()
        sig = np.zeros((1, 5))
        deps = np.zeros((1, 5))
        with pytest.raises(NotImplementedError, match="LAW50"):
            law50_visc_honey.shell_update(mat, sig, deps)

        with pytest.raises(NotImplementedError, match="LAW50"):
            pm.shell_update(mat, sig, deps)


# =============================================================================
# 12. Curve Resolution Hook
# =============================================================================

class TestLaw50CurveResolution:
    """Verify resolve hook binds function IDs to FunctTable instances."""

    def test_resolve_curves_from_model(self):
        mat = _make_law50(yfun11=[101, 102], yfun12=[201])
        model = Model()
        tbl1 = FunctTable(101, x=np.array([0.0, 1.0]), y=np.array([10.0, 20.0]))
        tbl2 = FunctTable(102, x=np.array([0.0, 1.0]), y=np.array([30.0, 40.0]))
        tbl3 = FunctTable(201, x=np.array([0.0, 1.0]), y=np.array([50.0, 60.0]))

        model.functions[101] = tbl1
        model.functions[102] = tbl2
        model.functions[201] = tbl3

        law50_visc_honey.resolve(mat, model)
        tables = mat.params.get("tables")
        assert tables is not None
        assert len(tables) == 6

        # Direction 11 has 2 curves:
        assert isinstance(tables[0], list)
        assert len(tables[0]) == 2
        assert tables[0][0] is tbl1
        assert tables[0][1] is tbl2

        # Direction 12 has 1 curve:
        assert tables[3] is tbl3


# =============================================================================
# 13. Vectorized Multi-Element Batch Evaluation
# =============================================================================

class TestLaw50VectorizedBatch:
    """Verify multi-element batch processing with diverse strain states."""

    def test_vectorized_batch_execution(self):
        n = 50
        mat = _make_law50(
            ea=100.0, eb=200.0, ec=300.0,
            gab=40.0, gbc=50.0, gca=60.0,
            asrate=50.0,
        )
        sig = np.zeros((n, 6))
        deps = np.random.uniform(-0.01, 0.01, size=(n, 6))
        extra = {}

        sign, epsp, c = law50_visc_honey.solid_update(mat, sig, deps, dt=0.001, extra=extra)

        assert sign.shape == (n, 6)
        assert epsp.shape == (n,)
        assert c.shape == (n,)
        assert extra["eps50"].shape == (n, 6)
        assert extra["uvar50"].shape == (n, 6)
        assert extra["off50"].shape == (n,)
        assert extra["compacted"].shape == (n,)
        assert not np.any(np.isnan(sign))
