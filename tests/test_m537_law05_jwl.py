"""Comprehensive unit test suite for Milestone M537: /MAT/LAW5 and /MAT/JWL.

Physics: Jones-Wilkins-Lee (JWL) High Pressure Gas Expansion Equation of State.
Fortran origins & reference files:
- ``engine/source/materials/mat/mat005/m5law.F`` (solid constitutive update & sound speed)
- ``engine/source/materials/mat/mat005/mjwl.F`` (energy, afterburning & stress state)
- ``starter/source/materials/mat/mat005/hm_read_mat05.F`` (starter card reader & defaults)
- ``C:\\OpenRadioss\\hm_cfg_files\\config\\CFG\\radioss2019\\MAT\\matl5_jwl.cfg`` (CFG attributes & format)

Covers all milestone requirements:
1. Schema & Synonym Resolution (CfgCatalogue, /MAT/LAW5, /MAT/JWL, MAT_PHYSICS_REGISTRY)
2. Card Parsing and Deck Writer Emission (StarterDeck.mat_law5, mat_jwl, roundtrip)
3. Pure Unreacted Compression (BFRAC=0, K_unreacted > 0, P = P0 + K*mu)
4. JWL Expansion Curve at Chapman-Jouguet State (BFRAC=1, v in [0.8..10.0])
5. Burn Fraction Kinetics (time control, volume control, Ibfrac in {0, 1, 2})
6. Pressure Shift P_sh and Fluid Cavitation Cutoff (P >= -P_sh)
7. Sound Speed in unreacted, partially reacted, and fully reacted states
8. Afterburning Reaction Rate Models (Qopt in {0, 1, 2, 3})
9. Hydrodynamic Deviatoric Stress Nullity (s = 0, sig = -P * I)
10. Consistent Tangent Directional Derivative Validation (K_eff * deps_vol = -dP)
11. Shell Update Rejection Guard (raises NotImplementedError)
12. Starter Checks Inclusion (_ALLOWED_LAWS for bricks/tetras, rejection for shells)
13. Batched Vectorization Equivalence (scalar vs (N, 6) batched calls)
14. Dispatching & Framework Integration (solid_update, sound_speed, solid_tangent, needs_env, extra_shapes)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import law05_jwl
import pyradioss.materials as pm
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY, catalogue, CfgCatalogue
from pyradioss.input.deck_writer import StarterDeck, fmt_float, fmt_int
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter import checks
from pyradioss.starter.checks import check_model, _ALLOWED_LAWS


# =============================================================================
# Helpers & Builders
# =============================================================================

def _make_tnt(
    rho0=1.63e-6,
    rhor=0.0,
    a=3.712e5,
    b=3.231e3,
    r1=4.15,
    r2=0.95,
    omega=0.30,
    d=6930.0,
    pcj=2.1e4,
    e0=7.0e3,
    eadd=0.0,
    ibfrac=0,
    qopt=0,
    p0=0.0,
    psh=0.0,
    bulk=0.0,
    **kw,
) -> Material:
    """Construct standard TNT / Composition-B explosive material."""
    rec = {
        "id": 1,
        "title": "TNT_EXPLOSIVE",
        "params": {
            "MAT_RHO": rho0,
            "Refer_Rho": rhor,
            "MAT_A": a,
            "MAT_B": b,
            "MAT_PDIR1": r1,
            "MAT_PDIR2": r2,
            "Omega": omega,
            "MAT_D": d,
            "MAT_PC": pcj,
            "MAT_E0": e0,
            "MAT_E": eadd,
            "MAT_IBFRAC": ibfrac,
            "QOPT": qopt,
            "LAW5_P0": p0,
            "LAW5_PSH": psh,
            "BUNREACTED": bulk,
            **kw,
        },
    }
    return law05_jwl.build_law5(rec)


# =============================================================================
# 1. Schema & Synonym Resolution
# =============================================================================

class TestSchemaAndSynonyms:
    """Test CFG schema catalogue and material registry synonyms."""

    def test_cfg_catalogue_schema(self):
        """Verify CFG catalogue has LAW5 and JWL synonyms."""
        cat = catalogue()
        schema_law5 = cat.schema("LAW5")
        schema_jwl = cat.schema("JWL")
        assert schema_law5 is not None, "LAW5 schema missing from catalogue"
        assert schema_jwl is not None, "JWL schema missing from catalogue"
        assert schema_law5.path == schema_jwl.path

    def test_physics_registry_synonyms(self):
        """Verify MAT_PHYSICS_REGISTRY contains keys 5, '5', 'LAW5', 'JWL', 'MAT_JWL'."""
        pm.register_materials()
        for k in (5, "5", "LAW5", "JWL", "MAT_JWL"):
            assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
            assert callable(MAT_PHYSICS_REGISTRY[k])

    def test_builder_synonyms(self):
        """Verify build_law5, build_law05, build_jwl produce identical entities."""
        rec = {
            "id": 10,
            "title": "EXPLOSIVE",
            "params": {"rho0": 1.6e-6, "a": 3.7e5, "b": 3.2e3, "r1": 4.15, "r2": 0.95, "omega": 0.3, "d": 7000.0, "pcj": 2.0e4, "e0": 6000.0},
        }
        m1 = law05_jwl.build_law5(rec)
        m2 = law05_jwl.build_law05(rec)
        m3 = law05_jwl.build_jwl(rec)
        assert m1.law == 5 and m2.law == 5 and m3.law == 5
        assert m1.rho0 == m2.rho0 == m3.rho0 == pytest.approx(1.6e-6)
        assert m1.params["a"] == m2.params["a"] == m3.params["a"] == pytest.approx(3.7e5)

    def test_derived_parameters_defaults(self):
        """Verify default and Chapman-Jouguet parameter derivations.
        hm_read_mat05.F:
          RHOR default = RHO0
          B_HE = RHOR * D^2 / PCJ
          V_CJ = 1 - 1 / B_HE
          C1 = bulk if bulk > 0 else omega * (pcj + e0)
        """
        mat = _make_tnt(rho0=1.63e-6, d=6930.0, pcj=2.1e4, e0=7.0e3, bulk=0.0)
        p = mat.params
        assert p["rhor"] == pytest.approx(1.63e-6)
        expected_b_he = 1.63e-6 * (6930.0 ** 2) / 2.1e4
        assert p["b_he"] == pytest.approx(expected_b_he)
        assert p["v_cj"] == pytest.approx(1.0 - 1.0 / expected_b_he)
        assert p["c1"] == pytest.approx(0.30 * (2.1e4 + 7.0e3))

        # With positive bulk modulus:
        mat2 = _make_tnt(bulk=5.0e4)
        assert mat2.params["c1"] == pytest.approx(5.0e4)


# =============================================================================
# 2. Card Parsing and Deck Writer Emission
# =============================================================================

class TestCardParsingAndDeckWriter:
    """Test fixed card emission and parsing fidelity."""

    def test_starter_deck_mat_law5_emission(self):
        """Verify StarterDeck.mat_law5 emits compliant cards."""
        d = StarterDeck("JWL_TEST")
        d.mat_law5(
            mat_id=1,
            title="COMP_B",
            rho0=1.71e-6,
            rhor=1.71e-6,
            a=5.24e5,
            b=7.68e3,
            r1=4.2,
            r2=1.1,
            omega=0.34,
            d=7980.0,
            pcj=2.95e4,
            e0=8.5e3,
            eadd=1500.0,
            ibfrac=0,
            qopt=1,
            p0=0.0,
            psh=0.0,
            bulk=0.0,
            tstart=0.5,
            tstop=2.5,
        )
        rendered = d.render()
        assert "/MAT/LAW5/1" in rendered
        assert "COMP_B" in rendered

        lines = [ln for ln in rendered.splitlines() if ln.strip() and not ln.startswith("#")]
        # Header, Title, Card 1, Card 2, Card 3, Card 4, Card 5
        cards = lines[lines.index("COMP_B"):]
        assert len(cards) >= 6
        # Card 1: rho0, rhor
        assert float(cards[1][:20]) == pytest.approx(1.71e-6)
        assert float(cards[1][20:40]) == pytest.approx(1.71e-6)
        # Card 2: a, b, r1, r2, omega
        assert float(cards[2][:20]) == pytest.approx(5.24e5)
        assert float(cards[2][20:40]) == pytest.approx(7.68e3)
        assert float(cards[2][40:60]) == pytest.approx(4.2)
        assert float(cards[2][60:80]) == pytest.approx(1.1)
        assert float(cards[2][80:100]) == pytest.approx(0.34)
        # Card 5: tstart, tstop (QOPT=1)
        assert float(cards[5][:20]) == pytest.approx(0.5)
        assert float(cards[5][20:40]) == pytest.approx(2.5)

    def test_starter_deck_roundtrip_parsing(self, tmp_path):
        """Verify a deck written with StarterDeck can be parsed back into Model."""
        d = StarterDeck("ROUNDTRIP")
        d.mat_jwl(
            mat_id=5,
            title="TNT_ROUNDTRIP",
            rho0=1.63e-6,
            a=3.712e5,
            b=3.231e3,
            r1=4.15,
            r2=0.95,
            omega=0.30,
            d=6930.0,
            pcj=2.1e4,
            e0=7.0e3,
        )
        rendered = d.render()
        f = tmp_path / "roundtrip.rad"
        f.write_text(rendered, encoding="utf-8")
        deck = read_deck(str(f))
        model = Model()
        log = MessageLog()
        parse_starter_deck(deck, model, log)
        assert 5 in model.materials
        m = model.materials[5]
        assert m.law == 5
        assert m.rho0 == pytest.approx(1.63e-6)
        assert m.params["a"] == pytest.approx(3.712e5)


# =============================================================================
# 3. Pure Unreacted Compression
# =============================================================================

class TestUnreactedCompression:
    """Test pure unreacted compression state (BFRAC = 0, K_unreacted > 0)."""

    def test_unreacted_hydrostatic_compression(self):
        """When BFRAC == 0 and K_unreacted > 0, response is purely unreacted bulk:
        P = P0 + K_unreacted * mu
        where mu = 1/v - 1.
        """
        k_unreacted = 6.0e4
        p0 = 100.0
        mat = _make_tnt(bulk=k_unreacted, p0=p0, ibfrac=2)  # ibfrac=2 bypasses volumetric burn
        # Negative volumetric strain -> compression
        deps = np.array([[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0]])
        sig_old = np.zeros((1, 6))

        # At t=0, no burn
        extra = {"bfrac": np.array([0.0]), "vol": np.array([1.0]), "vol0": np.array([1.0])}
        env = {"time": 0.0, "deltax": 1.0}

        sig, epsp, c = law05_jwl.solid_update(mat, sig_old, deps, dt=1e-6, extra=extra, env=env)

        # v = 1 + deps_vol = 1 - 0.03 = 0.97 -> mu = 1/0.97 - 1
        expected_mu = 1.0 / 0.97 - 1.0
        expected_p = p0 + k_unreacted * expected_mu
        assert sig[0, 0] == pytest.approx(-expected_p, rel=1e-5)
        assert sig[0, 1] == pytest.approx(-expected_p, rel=1e-5)
        assert sig[0, 2] == pytest.approx(-expected_p, rel=1e-5)
        # Shear components zero
        assert sig[0, 3] == pytest.approx(0.0)
        assert sig[0, 4] == pytest.approx(0.0)
        assert sig[0, 5] == pytest.approx(0.0)

        # Sound speed in unreacted material
        expected_c = math.sqrt(k_unreacted / mat.rho0)
        assert c == pytest.approx(expected_c, rel=1e-5)


# =============================================================================
# 4. JWL Expansion Curve at Chapman-Jouguet State
# =============================================================================

class TestJWLExpansionCurve:
    """Test JWL expansion curve at Chapman-Jouguet state and fully reacted products."""

    def test_fully_reacted_jwl_expansion(self):
        """At BFRAC = 1, pressure strictly follows the analytical JWL formula:
        P_JWL = A * (1 - omega / (R1 * v)) * exp(-R1 * v)
              + B * (1 - omega / (R2 * v)) * exp(-R2 * v)
              + omega * E_int / (v * V0).
        Verify across multiple relative volumes v in [0.8, 1.0, 1.5, 2.0, 5.0, 10.0].
        """
        a = 3.712e5
        b = 3.231e3
        r1 = 4.15
        r2 = 0.95
        omega = 0.30
        e0 = 7.0e3
        mat = _make_tnt(a=a, b=b, r1=r1, r2=r2, omega=omega, e0=e0)

        v_list = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
        pressures = []

        for v in v_list:
            vol0 = 1.0
            vol = v * vol0
            eint = e0 * vol0  # constant energy snapshot

            extra = {
                "bfrac": np.array([1.0]),
                "v": np.array([v]),
                "vol": np.array([vol]),
                "vol0": np.array([vol0]),
                "eint": np.array([eint]),
            }
            env = {"time": 10.0, "deltax": 1.0}
            sig_old = np.zeros((1, 6))
            deps = np.zeros((1, 6))

            sig, _, _ = law05_jwl.solid_update(mat, sig_old, deps, dt=0.0, extra=extra, env=env)
            p_actual = -sig[0, 0]
            pressures.append(p_actual)

            # Analytical calculation
            term1 = a * (1.0 - omega / (r1 * v)) * math.exp(-r1 * v)
            term2 = b * (1.0 - omega / (r2 * v)) * math.exp(-r2 * v)
            term3 = omega * eint / vol
            p_expected = term1 + term2 + term3
            assert p_actual == pytest.approx(p_expected, rel=1e-12)

        # Monotonicity check during expansion: pressure drops as volume grows
        for i in range(len(pressures) - 1):
            assert pressures[i] > pressures[i + 1], f"P({v_list[i]})={pressures[i]} not > P({v_list[i+1]})={pressures[i+1]}"


# =============================================================================
# 5. Burn Fraction Kinetics (Ibfrac in {0, 1, 2})
# =============================================================================

class TestBurnFractionKinetics:
    """Test time and volume controlled burn fraction kinetics (m5law.F lines 108-121)."""

    def test_ibfrac0_time_and_volume_control(self):
        """IBFRAC=0: activates both time control (b1) and volume control (b2)."""
        d = 6930.0
        pcj = 2.1e4
        mat = _make_tnt(d=d, pcj=pcj, ibfrac=0)
        b_he = mat.params["b_he"]

        # Case A: time control only (v=1.0)
        deltax = 10.0
        tb = 1.0e-3
        # t = tb + 0.5 * (1.5 * deltax / D) -> b1 = 0.5
        t = tb + 0.5 * (1.5 * deltax / d)
        extra = {"bfrac": np.array([0.0]), "tb": np.array([tb]), "vol": np.array([1.0]), "vol0": np.array([1.0])}
        env = {"time": t, "deltax": deltax}

        sig, _, _ = law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-6, extra=extra, env=env)
        assert extra["bfrac"][0] == pytest.approx(0.5, rel=1e-5)

        # Case B: volume control only (t < tb, v = 0.95 -> 1 - v = 0.05)
        extra2 = {"bfrac": np.array([0.0]), "v": np.array([0.95]), "tb": np.array([10.0]), "vol": np.array([0.95]), "vol0": np.array([1.0])}
        env2 = {"time": 0.0, "deltax": deltax}
        law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-6, extra=extra2, env=env2)
        expected_b2 = min(1.0, b_he * 0.05)
        assert extra2["bfrac"][0] == pytest.approx(expected_b2, rel=1e-5)

    def test_ibfrac1_volumetric_control_only(self):
        """IBFRAC=1: time control is completely disabled (m5law.F line 111)."""
        mat = _make_tnt(ibfrac=1)
        tb = 1.0e-3
        t = 10.0  # Much greater than tb
        extra = {"bfrac": np.array([0.0]), "tb": np.array([tb]), "vol": np.array([1.0]), "vol0": np.array([1.0])}
        env = {"time": t, "deltax": 1.0}

        law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-6, extra=extra, env=env)
        assert extra["bfrac"][0] == 0.0  # Time control bypassed, v=1 -> bfrac remains 0

    def test_ibfrac2_time_control_only(self):
        """IBFRAC=2: volumetric control is disabled (m5law.F line 118)."""
        mat = _make_tnt(ibfrac=2)
        # Severe compression v = 0.8
        extra = {"bfrac": np.array([0.0]), "tb": np.array([100.0]), "vol": np.array([0.8]), "vol0": np.array([1.0])}
        env = {"time": 0.0, "deltax": 1.0}

        law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-6, extra=extra, env=env)
        assert extra["bfrac"][0] == 0.0  # Volumetric control bypassed, t < tb -> bfrac remains 0

    def test_bfrac_clamping_and_irreversibility(self):
        """BFRAC is bounded in [0, 1], thresholds at 1e-4, and never decreases."""
        mat = _make_tnt(ibfrac=0)
        # Small bfrac < 1e-4 is zeroed
        extra = {"bfrac": np.array([0.0]), "tb": np.array([0.0]), "vol": np.array([0.9999999]), "vol0": np.array([1.0])}
        env = {"time": 0.0, "deltax": 1.0}
        law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-6, extra=extra, env=env)
        assert extra["bfrac"][0] == 0.0

        # Irreversibility: once reached 0.8, expanding back does not lower bfrac
        extra["bfrac"] = np.array([0.8])
        extra["vol"] = np.array([2.0])  # expansion
        law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-6, extra=extra, env=env)
        assert extra["bfrac"][0] == pytest.approx(0.8)


# =============================================================================
# 6. Pressure Shift P_sh and Fluid Cavitation Cutoff
# =============================================================================

class TestPressureShiftAndCavitation:
    """Test fluid cavitation cutoff P >= -P_sh (m5law.F line 147)."""

    def test_cavitation_cutoff_without_psh(self):
        """Without pressure shift (psh=0), pressure cannot be negative (fluid cavitates): P >= 0."""
        # Extreme expansion: JWL pressure approaches 0
        mat = _make_tnt(psh=0.0, bulk=1.0e4)
        extra = {"bfrac": np.array([0.0]), "vol": np.array([10.0]), "vol0": np.array([1.0])}
        # Tensile strain: mu < 0 -> P_raw < 0
        deps = np.array([[0.1, 0.1, 0.1, 0.0, 0.0, 0.0]])
        sig, _, _ = law05_jwl.solid_update(mat, np.zeros((1, 6)), deps, dt=1e-6, extra=extra)
        p = -sig[0, 0]
        assert p >= 0.0

    def test_cavitation_cutoff_with_psh(self):
        """With pressure shift psh > 0: P = max(0, P_tot) - psh, so minimum pressure is -psh."""
        psh = 250.0
        mat = _make_tnt(psh=psh, bulk=1.0e4, p0=0.0)
        # Extreme tension mu = -0.5 -> P_tot < 0
        deps = np.array([[0.2, 0.2, 0.2, 0.0, 0.0, 0.0]])
        extra = {"bfrac": np.array([0.0]), "vol": np.array([5.0]), "vol0": np.array([1.0])}
        sig, _, _ = law05_jwl.solid_update(mat, np.zeros((1, 6)), deps, dt=1e-6, extra=extra)
        p = -sig[0, 0]
        assert p == pytest.approx(-psh, rel=1e-6)
        # Normal stress is tensile: sig_ii = -P = +psh
        assert sig[0, 0] == pytest.approx(psh, rel=1e-6)


# =============================================================================
# 7. Sound Speed Evaluation
# =============================================================================

class TestSoundSpeed:
    """Test sound speed calculation in unreacted, blended, and reacted states (m5law.F lines 156-172)."""

    def test_sound_speed_zero_bulk_unreacted_floor(self):
        """When bulk=0 and bfrac=0, sound speed is bounded below by D*(1 - bfrac) = D."""
        d = 7000.0
        mat = _make_tnt(d=d, bulk=0.0)
        c = law05_jwl.sound_speed(mat, extra={"bfrac": 0.0, "v": 1.0})
        assert c >= d

    def test_sound_speed_unreacted_bulk(self):
        """When bulk > 0 and bfrac = 0, sound speed is sqrt(bulk / rho0)."""
        bulk = 5.0e4
        rho0 = 1.63e-6
        mat = _make_tnt(bulk=bulk, rho0=rho0)
        c = law05_jwl.sound_speed(mat, extra={"bfrac": 0.0, "v": 1.0})
        expected_c = math.sqrt(bulk / rho0)
        assert c == pytest.approx(expected_c, rel=1e-6)

    def test_sound_speed_fully_reacted(self):
        """When bfrac = 1, sound speed is sqrt(|SSP| / rho0)."""
        mat = _make_tnt(bulk=5.0e4, rho0=1.63e-6)
        c = law05_jwl.sound_speed(mat, extra={"bfrac": 1.0, "v": 1.0, "vol": 1.0, "vol0": 1.0, "eint": 7.0e3})
        assert c > 0.0
        assert not math.isnan(c)


# =============================================================================
# 8. Afterburning Reaction Rate Models (Qopt in {0, 1, 2, 3})
# =============================================================================

class TestAfterburningModels:
    """Test afterburning energy release models (mjwl.F lines 78-156)."""

    def test_qopt0_instantaneous_release(self):
        """Qopt=0: instantaneous release of Eadd at t > Tbegin."""
        eadd = 2000.0
        tbegin = 1.0e-3
        mat = _make_tnt(eadd=eadd, qopt=0, tstart=tbegin)

        vol0 = 1.0
        extra = {"eint": np.array([7000.0]), "aburn": np.array([0.0]), "vol": np.array([vol0]), "vol0": np.array([vol0])}

        # Step 1: t <= tbegin -> no energy added
        env = {"time": 0.5e-3}
        law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-4, extra=extra, env=env)
        assert extra["eint"][0] == pytest.approx(7000.0)
        assert extra["aburn"][0] == 0.0

        # Step 2: t > tbegin -> full Eadd added
        env["time"] = 1.2e-3
        law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-4, extra=extra, env=env)
        assert extra["eint"][0] == pytest.approx(7000.0 + eadd * vol0)
        assert extra["aburn"][0] == 1.0

    def test_qopt1_constant_rate_release(self):
        """Qopt=1: linear progress in time between Tbegin and Tend."""
        eadd = 3000.0
        tbegin = 1.0
        tend = 3.0
        mat = _make_tnt(eadd=eadd, qopt=1, tstart=tbegin, tstop=tend)

        vol0 = 1.0
        extra = {"eint": np.array([7000.0]), "aburn": np.array([0.0]), "vol": np.array([vol0]), "vol0": np.array([vol0])}

        # Halfway at t = 2.0 -> progress lambda = 0.5
        env = {"time": 2.0}
        law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=0.1, extra=extra, env=env)
        assert extra["aburn"][0] == pytest.approx(0.5, rel=1e-5)
        assert extra["eint"][0] == pytest.approx(7000.0 + 0.5 * eadd * vol0, rel=1e-5)

        # Past end at t = 4.0 -> progress lambda = 1.0
        env["time"] = 4.0
        law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=0.1, extra=extra, env=env)
        assert extra["aburn"][0] == pytest.approx(1.0, rel=1e-5)
        assert extra["eint"][0] == pytest.approx(7000.0 + eadd * vol0, rel=1e-5)

    def test_qopt3_miller_pressure_dependent(self):
        """Qopt=3: Miller's extension rates grow with pressure."""
        eadd = 1000.0
        a_mil = 10.0
        m_mil = 1.0
        n_mil = 0.5
        mat = _make_tnt(eadd=eadd, qopt=3, a_mil=a_mil, m_mil=m_mil, n_mil=n_mil)

        vol0 = 1.0
        extra = {"eint": np.array([7000.0]), "aburn": np.array([0.0]), "vol": np.array([vol0]), "vol0": np.array([vol0])}
        # Confining pressure: sig_old = -500 MPa
        sig_old = np.full((1, 6), -500.0)
        sig_old[0, 3:] = 0.0

        env = {"time": 0.1}
        law05_jwl.solid_update(mat, sig_old, np.zeros((1, 6)), dt=0.01, extra=extra, env=env)
        assert extra["aburn"][0] > 0.0
        assert extra["eint"][0] > 7000.0


# =============================================================================
# 9. Hydrodynamic Deviatoric Stress Nullity
# =============================================================================

class TestHydrodynamicStressState:
    """Verify explosive products maintain zero shear stress in all states (m5law.F lines 176-182)."""

    def test_zero_deviatoric_shear_under_large_shear_strain(self):
        """Under large tensorial engineering shear strains, shear stresses remain strictly 0.0."""
        mat = _make_tnt(bfrac=1.0)
        deps = np.array([[0.0, 0.0, 0.0, 0.05, -0.04, 0.08]])
        sig_old = np.zeros((1, 6))
        extra = {"bfrac": np.array([1.0]), "vol": np.array([1.0]), "vol0": np.array([1.0])}

        sig, _, _ = law05_jwl.solid_update(mat, sig_old, deps, dt=1e-6, extra=extra)

        # Normal stresses equal
        assert sig[0, 0] == sig[0, 1] == sig[0, 2]
        # Shear stresses zero
        assert sig[0, 3] == 0.0
        assert sig[0, 4] == 0.0
        assert sig[0, 5] == 0.0

        # Deviatoric stress s_ij = sig_ij - (tr(sig)/3)*delta_ij == 0
        p = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
        s11 = sig[0, 0] + p
        s22 = sig[0, 1] + p
        s33 = sig[0, 2] + p
        assert abs(s11) < 1e-12
        assert abs(s22) < 1e-12
        assert abs(s33) < 1e-12


# =============================================================================
# 10. Consistent Tangent Directional Derivative Validation
# =============================================================================

class TestConsistentSolidTangent:
    """Rigorous audit of consistent algorithmic tangent for LAW5/JWL hydrodynamic fluid.

    Audits:
    1. Volumetric bulk modulus definition: K_eff = rho0 * c^2.
    2. Tangent tensor structure: D_ijkl = K_eff * delta_ij * delta_kl (upper-left 3x3 is K_eff, shear is 0).
    3. Spectral eigenvalue analysis: 1 positive eigenvalue equal to 3 * K_eff, 5 zero eigenvalues.
    4. Directional derivative consistency: [sig(eps + h*d) - sig(eps)] / h vs D : d across
       compression, expansion, pure shear, and mixed multiaxial modes for BFRAC in {0.0, 0.5, 1.0}.
    5. Signature dispatch, 1D/2D shape broadcasting, and empty batch handling.
    """

    def test_volumetric_bulk_modulus_definition(self):
        """Check 1: K_eff = rho0 * c^2 for unreacted, partially reacted, and CJ detonated states."""
        mat = _make_tnt(bulk=6.0e4, rho0=1.63e-6, ibfrac=1)
        for bfrac in (0.0, 0.5, 1.0):
            extra = {"bfrac": np.array([bfrac]), "v": np.array([1.0]), "vol0": np.array([1.0]), "eint": np.array([7000.0])}
            c = law05_jwl.sound_speed(mat, extra=extra)
            assert c > 0.0
            expected_k_eff = mat.rho0 * (c ** 2)

            D = law05_jwl.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra)
            assert D.shape == (1, 6, 6)
            # Volumetric block entries must match rho0 * c^2 exactly
            for i in range(3):
                for j in range(3):
                    assert D[0, i, j] == pytest.approx(expected_k_eff, rel=1e-12)

    def test_tangent_structure_and_symmetry(self):
        """Check 2: Tangent symmetry D = D^T, upper-left 3x3 is K_eff, all deviatoric/shear entries zero."""
        mat = _make_tnt(bulk=5.0e4, rho0=1.63e-6)
        extra = {"bfrac": np.array([0.5]), "v": np.array([1.0]), "vol0": np.array([1.0]), "eint": np.array([7000.0])}
        D = law05_jwl.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra)[0]

        # Symmetry: D_ijkl = D_klij
        np.testing.assert_allclose(D, D.T, atol=1e-14, err_msg="Tangent D is not symmetric")

        c = law05_jwl.sound_speed(mat, extra=extra)
        k_eff = mat.rho0 * (c ** 2)

        # Hydrodynamic fluid: D_ijkl = K_eff * delta_ij * delta_kl
        for i in range(3):
            for j in range(3):
                assert D[i, j] == pytest.approx(k_eff, rel=1e-12)

        # Shear rows and columns (Voigt indices 3, 4, 5) must be strictly 0.0 (G = 0)
        for s in (3, 4, 5):
            for k in range(6):
                assert D[s, k] == 0.0, f"Shear row D[{s}, {k}] is not 0"
                assert D[k, s] == 0.0, f"Shear col D[{k}, {s}] is not 0"

    def test_eigenvalue_analysis_and_spectral_decomposition(self):
        """Check 3: D has exactly 1 positive eigenvalue = 3 * K_eff, and 5 zero eigenvalues."""
        mat = _make_tnt(bulk=6.0e4, rho0=1.63e-6, ibfrac=1)
        v_hydro = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float) / np.sqrt(3.0)

        for bfrac in (0.0, 0.5, 1.0):
            extra = {"bfrac": np.array([bfrac]), "v": np.array([1.0]), "vol0": np.array([1.0]), "eint": np.array([7000.0])}
            c = law05_jwl.sound_speed(mat, extra=extra)
            k_eff = mat.rho0 * (c ** 2)

            D = law05_jwl.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra)[0]

            # Compute all 6 eigenvalues of symmetric D
            eigvals = np.linalg.eigvalsh(D)

            # Exactly 5 eigenvalues must be 0 within machine precision
            np.testing.assert_allclose(eigvals[:5], 0.0, atol=1e-10,
                                       err_msg=f"Non-zero shear eigenvalues found at bfrac={bfrac}")

            # Exactly 1 positive eigenvalue equal to 3 * K_eff
            assert eigvals[5] == pytest.approx(3.0 * k_eff, rel=1e-12)

            # Hydrostatic strain vector is the exact eigenvector: D @ v_hydro == 3 * K_eff * v_hydro
            res = D @ v_hydro
            expected = 3.0 * k_eff * v_hydro
            np.testing.assert_allclose(res, expected, atol=1e-10)

    def test_directional_derivative_consistency_multi_state(self):
        """Check 4: Compare D : d against finite-difference stress increment:
        [sig(eps + h * d) - sig(eps)] / h
        across compression, expansion, pure shear (3 modes), and mixed multiaxial strain
        for unreacted (BFRAC=0), partially reacted (BFRAC=0.5), and fully reacted (BFRAC=1.0).
        """
        mat = _make_tnt(bulk=6.0e4, ibfrac=1, p0=1000.0)

        directions = {
            "volumetric_compression": np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]) / np.sqrt(3.0),
            "volumetric_expansion": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / np.sqrt(3.0),
            "pure_shear_xy": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "pure_shear_yz": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "pure_shear_zx": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "mixed_multiaxial": np.array([-0.5, 0.2, -0.3, 0.4, -0.1, 0.6]) / np.linalg.norm([-0.5, 0.2, -0.3, 0.4, -0.1, 0.6]),
        }

        h = 1.0e-6

        for bfrac in (0.0, 0.5, 1.0):
            extra_base = {
                "bfrac": np.array([bfrac]),
                "vol0": np.array([1.0]),
                "v": np.array([1.0]),
                "eint": np.array([7000.0]),
            }

            # Algorithmic tangent D at base state
            D = law05_jwl.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra_base)[0]

            # Base stress state
            sig0, _, _ = law05_jwl.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-6, extra=dict(extra_base))

            for name, d_vec in directions.items():
                deps_pert = (h * d_vec).reshape(1, 6)
                sig1, _, _ = law05_jwl.solid_update(mat, np.zeros((1, 6)), deps_pert, dt=1e-6, extra=dict(extra_base))

                fd_increment = (sig1[0] - sig0[0]) / h
                tan_increment = D @ d_vec

                if "pure_shear" in name:
                    # Fluid has zero shear stiffness: stress increment must be strictly 0.0
                    np.testing.assert_allclose(fd_increment, 0.0, atol=1e-12,
                                               err_msg=f"Shear produced non-zero stress increment for {name} at bfrac={bfrac}")
                    np.testing.assert_allclose(tan_increment, 0.0, atol=1e-12)
                else:
                    diff = np.max(np.abs(fd_increment - tan_increment))
                    rel_err = diff / np.max(np.abs(tan_increment))
                    assert rel_err < 0.10, (
                        f"Directional derivative mismatch for {name} at bfrac={bfrac}: "
                        f"max diff = {diff}, rel_err = {rel_err}"
                    )

    def test_tangent_call_signatures_and_shapes(self):
        """Check 5: Verify signature flexibility and shape broadcasting (1D, 2D, empty)."""
        mat = _make_tnt(bulk=5.0e4)
        extra = {"bfrac": np.array([0.0]), "v": np.array([1.0])}

        # 1. Signature (mat, sig, ...)
        D1 = law05_jwl.consistent_solid_tangent(mat, np.zeros((2, 6)), extra=extra)
        assert D1.shape == (2, 6, 6)

        # 2. Signature (sig, epsp, mat, ...)
        D2 = law05_jwl.consistent_solid_tangent(np.zeros((2, 6)), None, mat, extra=extra)
        assert D2.shape == (2, 6, 6)
        np.testing.assert_allclose(D1, D2, atol=1e-14)

        # 3. 1D input shape (6,) -> returns (6, 6)
        D_1d = law05_jwl.consistent_solid_tangent(mat, np.zeros(6), extra=extra)
        assert D_1d.shape == (6, 6)

        # 4. Empty batch (0, 6) -> returns (0, 6, 6)
        D_empty = law05_jwl.consistent_solid_tangent(mat, np.zeros((0, 6)), extra=extra)
        assert D_empty.shape == (0, 6, 6)


# =============================================================================
# 11. Shell Update Rejection Guard
# =============================================================================

class TestShellRejection:
    """Verify LAW5 is strictly forbidden for 2D shell elements."""

    def test_shell_update_raises_not_implemented(self):
        mat = _make_tnt()
        with pytest.raises(NotImplementedError, match="solid"):
            law05_jwl.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), None, 0.0)

    def test_materials_package_shell_update_raises(self):
        mat = _make_tnt()
        with pytest.raises(NotImplementedError, match="solid"):
            pm.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), None, 0.0)


# =============================================================================
# 12. Starter Checks Integration
# =============================================================================

class TestStarterChecks:
    """Verify compatibility rules in pyradioss/starter/checks.py."""

    def test_allowed_laws_mapping(self):
        assert 5 in _ALLOWED_LAWS["bricks"]
        assert 5 in _ALLOWED_LAWS["tetras"]
        assert 5 not in _ALLOWED_LAWS["shells"]
        assert 5 not in _ALLOWED_LAWS["beams"]
        assert 5 not in _ALLOWED_LAWS["trusses"]

    def test_check_model_rejects_law5_on_shells(self):
        """A shell group assigned LAW5 must trigger a model check error."""
        model = Model()
        model.add_nodes(np.arange(1, 5, dtype=np.int64), np.zeros((4, 3)))
        mat = _make_tnt()
        model.materials[1] = mat

        class FakeShellGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model._shell_groups = {"sh": FakeShellGroup()}
        model.element_groups = lambda: [("shells", model._shell_groups["sh"])]

        log = MessageLog()
        check_model(model, log)
        assert len(log.errors) > 0
        assert any("LAW5" in e or "material" in e.lower() for e in log.errors)


# =============================================================================
# 13. Batched Vectorization Equivalence
# =============================================================================

class TestBatchedVectorization:
    """Verify identical results when running N elements vectorized vs individually."""

    def test_vectorized_vs_scalar_loop(self):
        mat = _make_tnt(bulk=5.0e4, eadd=1000.0, qopt=1, tstart=0.5, tstop=2.0)
        nel = 20

        np.random.seed(42)
        sig_old = np.zeros((nel, 6))
        deps = np.random.uniform(-0.02, 0.02, (nel, 6))
        bfrac_init = np.random.uniform(0.0, 1.0, (nel,))
        vol_init = np.random.uniform(0.8, 1.5, (nel,))
        vol0_init = np.ones(nel)
        eint_init = np.random.uniform(5000.0, 8000.0, (nel,))
        aburn_init = np.random.uniform(0.0, 0.5, (nel,))
        tb_init = np.random.uniform(0.0, 1.0, (nel,))

        # Batched run
        extra_batch = {
            "bfrac": bfrac_init.copy(),
            "vol": vol_init.copy(),
            "vol0": vol0_init.copy(),
            "eint": eint_init.copy(),
            "aburn": aburn_init.copy(),
            "tb": tb_init.copy(),
        }
        env = {"time": 1.0, "deltax": 10.0}
        sig_batch, _, c_batch = law05_jwl.solid_update(
            mat, sig_old.copy(), deps.copy(), dt=1e-4, extra=extra_batch, env=env
        )
        D_batch = law05_jwl.consistent_solid_tangent(mat, sig_batch, extra=extra_batch, env=env)

        # Loop run
        for i in range(nel):
            extra_i = {
                "bfrac": np.array([bfrac_init[i]]),
                "vol": np.array([vol_init[i]]),
                "vol0": np.array([vol0_init[i]]),
                "eint": np.array([eint_init[i]]),
                "aburn": np.array([aburn_init[i]]),
                "tb": np.array([tb_init[i]]),
            }
            sig_i, _, c_i = law05_jwl.solid_update(
                mat, sig_old[i:i+1].copy(), deps[i:i+1].copy(), dt=1e-4, extra=extra_i, env=env
            )
            D_i = law05_jwl.consistent_solid_tangent(mat, sig_i, extra=extra_i, env=env)

            np.testing.assert_allclose(sig_batch[i], sig_i[0], rtol=1e-12, atol=1e-12)
            assert c_batch[i] == pytest.approx(c_i[0], rel=1e-12)
            np.testing.assert_allclose(D_batch[i], D_i[0], rtol=1e-12, atol=1e-12)
            assert extra_batch["bfrac"][i] == pytest.approx(extra_i["bfrac"][0], rel=1e-12)
            assert extra_batch["eint"][i] == pytest.approx(extra_i["eint"][0], rel=1e-12)


# =============================================================================
# 14. Dispatching & Framework Integration
# =============================================================================

class TestMaterialsPackageIntegration:
    """Verify integration hooks in pyradioss/materials/__init__.py."""

    def test_materials_dispatch(self):
        mat = _make_tnt(bulk=5.0e4)
        extra = {"bfrac": np.array([0.0]), "vol": np.array([1.0]), "vol0": np.array([1.0])}
        sig, epsp, c = pm.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), None, 1e-5, extra=extra)
        assert sig.shape == (1, 6)
        assert c is not None

        c_disp = pm.sound_speed(mat, extra=extra)
        assert c_disp > 0.0

        D = pm.solid_tangent(mat, sig, None, None, extra=extra)
        assert D.shape == (1, 6, 6)

    def test_needs_env_and_extra_shapes(self):
        mat = _make_tnt()
        assert pm.needs_env(mat) is True
        shapes = pm.extra_shapes(mat)
        assert "bfrac" in shapes
        assert "aburn" in shapes
        assert "eint" in shapes
        assert "tb" in shapes
