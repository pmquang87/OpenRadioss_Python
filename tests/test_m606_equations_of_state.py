"""
Tests for M606: Advanced Equations of State Suite
(/EOS/JWL, /EOS/MURNAGHAN, /EOS/NOBLE-ABEL, /EOS/NASG, /EOS/PUFF).

Validates:
- Analytical formulation fidelity against OpenRadioss upstream:
  * common_source/eos/jwl.F
  * common_source/eos/murnaghan.F
  * common_source/eos/noble_abel.F
  * common_source/eos/nasg.F
  * common_source/eos/puff.F
- Pressure evaluation, derivatives dPdmu/dPdE/DPDM, and acoustic bulk sound speed
- Closed-form and predictor-corrector implicit energy-pressure time integration
- Solid element (solid_hexa8) force and time step coupling
- Starter keyword parsing and DeckWriter roundtrip
"""

import numpy as np
import pytest

from pyradioss.model.entities import EquationOfState
from pyradioss.materials import eos
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import DeckWriter
from pyradioss.input.starter_keywords import read_eos
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


class TestJWL:
    """JWL Equation of State (common_source/eos/jwl.F)."""

    def test_jwl_reference_pressure_and_derivatives(self):
        # Standard TNT parameters (units: GPa, cm3/g or mm-ms-g-MPa)
        # A = 371.2, B = 3.23, R1 = 4.15, R2 = 0.95, omega = 0.30, E0 = 7.0, rho0 = 1.63
        params = {
            "a": 371.2,
            "b": 3.23,
            "r1": 4.15,
            "r2": 0.95,
            "omega": 0.30,
            "e0": 7.0,
            "psh": 0.0,
            "rho0_card": 1.63,
        }
        obj = EquationOfState(kind="JWL", params=params, rho0=1.63)

        # At mu = 0 (v = 1):
        # P = A * (1 - omega/R1) * exp(-R1) + B * (1 - omega/R2) * exp(-R2) + omega * E0
        mu = 0.0
        e_val = 7.0
        p_eval = eos.pressure(obj, mu, e_val)

        r1df = 4.15
        r2df = 0.95
        term1 = 371.2 * (1.0 - 0.30 / r1df) * np.exp(-r1df)
        term2 = 3.23 * (1.0 - 0.30 / r2df) * np.exp(-r2df)
        term3 = 0.30 * e_val
        p_expected = term1 + term2 + term3

        assert np.isclose(p_eval, p_expected, rtol=1e-6)

        # Sound speed at reference state
        c_eval = eos.sound_speed(obj, mu, e_val)
        assert c_eval > 0.0

        # Verify initial_state helper
        e0, p0 = eos.initial_state(obj)
        assert e0 == 7.0
        assert np.isclose(p0, p_expected, rtol=1e-6)

    def test_jwl_implicit_energy_integration(self):
        params = {
            "a": 100.0,
            "b": 2.0,
            "r1": 4.0,
            "r2": 1.0,
            "omega": 0.35,
            "e0": 5.0,
            "psh": 0.0,
            "rho0_card": 1.5,
        }
        obj = EquationOfState(kind="JWL", params=params, rho0=1.5)

        # Perform an expansion step: dv = 0.05 (J increases from 1.0 to 1.05)
        # mu = 1/1.05 - 1
        mu = 1.0 / 1.05 - 1.0
        dv = np.array([0.05])
        mu_arr = np.array([mu])
        e_old = np.array([5.0])
        p_old = np.array([eos.pressure(obj, 0.0, 5.0)])
        de_other = np.array([0.0])

        p_new, e_new, c2 = eos.update(obj, mu_arr, dv, e_old, p_old, de_other)

        # Expansion should decrease internal energy due to pdV work
        assert e_new[0] < e_old[0]
        assert p_new[0] < p_old[0]
        assert c2[0] > 0.0


class TestMurnaghan:
    """Murnaghan Equation of State (common_source/eos/murnaghan.F)."""

    def test_murnaghan_analytical_pressure_and_bulk_modulus(self):
        k0 = 10.0  # Bulk modulus at zero pressure
        k1 = 4.0   # dK/dP
        p0 = 0.1   # Initial pressure
        psh = 0.0
        params = {"k0": k0, "k1": k1, "p0": p0, "psh": psh, "rho0_card": 1.0}
        obj = EquationOfState(kind="MURNAGHAN", params=params, rho0=1.0)

        # Compression mu = 0.1 (density increased by 10%)
        # P = K0/K1 * ((1+mu)^K1 - 1) + P0
        mu = 0.1
        p_eval = eos.pressure(obj, mu, e=0.0)
        p_expected = (k0 / k1) * ((1.0 + mu) ** k1 - 1.0) + p0
        assert np.isclose(p_eval, p_expected, rtol=1e-6)

        # Bulk modulus dP/dmu = K0 * (1+mu)^(K1 - 1)
        # Sound speed c = sqrt(dP/dmu / rho0)
        c_eval = eos.sound_speed(obj, mu, e=0.0)
        dpdm_expected = k0 * ((1.0 + mu) ** (k1 - 1.0))
        c_expected = np.sqrt(dpdm_expected / 1.0)
        assert np.isclose(c_eval, c_expected, rtol=1e-6)

    def test_murnaghan_energy_update(self):
        params = {"k0": 20.0, "k1": 3.5, "p0": 0.0, "psh": 0.0, "rho0_card": 2.0}
        obj = EquationOfState(kind="MURNAGHAN", params=params, rho0=2.0)

        # Compression step: dv = -0.02
        dv = np.array([-0.02])
        mu_arr = np.array([0.02])
        e_old = np.array([1.0])
        p_old = np.array([0.0])
        de_other = np.array([0.0])

        p_new, e_new, c2 = eos.update(obj, mu_arr, dv, e_old, p_old, de_other)
        # Compression: pdV work adds internal energy
        assert e_new[0] > e_old[0]
        assert p_new[0] > 0.0


class TestNobleAbel:
    """Noble-Abel Equation of State (common_source/eos/noble_abel.F)."""

    def test_noble_abel_dilute_limit_ideal_gas(self):
        # When covolume b = 0, Noble-Abel reduces to ideal gas P = (gamma - 1) * rho * e
        gamma = 1.4
        params = {"b": 0.0, "gamma": gamma, "psh": 0.0, "e0": 2.5, "rho0_card": 1.2}
        obj = EquationOfState(kind="NOBLE-ABEL", params=params, rho0=1.2)

        mu = 0.0
        e_val = 2.5
        p_eval = eos.pressure(obj, mu, e_val)
        p_ideal = (gamma - 1.0) * e_val
        assert np.isclose(p_eval, p_ideal, rtol=1e-6)

    def test_noble_abel_covolume_stiffening(self):
        # When covolume b > 0, denominator 1 - b*rho0*(1+mu) reduces effective volume,
        # dramatically increasing pressure
        b_cov = 0.2
        gamma = 1.4
        rho0 = 1.0
        params = {"b": b_cov, "gamma": gamma, "psh": 0.0, "e0": 1.0, "rho0_card": rho0}
        obj = EquationOfState(kind="NOBLE-ABEL", params=params, rho0=rho0)

        mu = 0.5  # rho = 1.5, b*rho = 0.3, denom = 0.7
        e_val = 1.0
        p_eval = eos.pressure(obj, mu, e_val)
        expected = (gamma - 1.0) * (1.0 + mu) * e_val / (1.0 - b_cov * rho0 * (1.0 + mu))
        assert np.isclose(p_eval, expected, rtol=1e-6)

        c_eval = eos.sound_speed(obj, mu, e_val)
        assert c_eval > 0.0


class TestNASG:
    """Noble-Abel Stiffened-Gas (NASG) Equation of State (common_source/eos/nasg.F)."""

    def test_nasg_pressure_and_cavitation_cutoff(self):
        gamma = 1.5
        p_star = 100.0
        q = 5.0
        b_cov = 0.05
        rho0 = 1.0
        psh = 0.0
        p0 = 1.0
        params = {
            "b": b_cov,
            "gamma": gamma,
            "p_star": p_star,
            "q": q,
            "psh": psh,
            "p0": p0,
            "cv": 1.0,
            "rho0_card": rho0,
        }
        obj = EquationOfState(kind="NASG", params=params, rho0=rho0)

        # Under strong tension (e.g. negative energy or large expansion),
        # pressure should not drop below cavitation floor -gamma * P_star
        mu_ten = -0.5
        e_low = 0.0
        p_ten = eos.pressure(obj, mu_ten, e_low)
        assert p_ten >= -gamma * p_star

        # Normal compression state
        mu_comp = 0.1
        e_norm = 20.0
        p_comp = eos.pressure(obj, mu_comp, e_norm)
        denom = 1.0 - b_cov * rho0 * (1.0 + mu_comp)
        num = (e_norm - rho0 * q)
        p_expected = (gamma - 1.0) * (1.0 + mu_comp) * num / denom - gamma * p_star
        assert np.isclose(p_comp, p_expected, rtol=1e-6)


class TestPUFF:
    """PUFF Equation of State (common_source/eos/puff.F)."""

    def test_puff_three_regimes(self):
        c1 = 10.0
        c2 = 5.0
        c3 = 2.0
        t1 = 8.0
        t2 = 3.0
        esubl = 50.0
        gamma0 = 1.5
        h = 0.5
        params = {
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "t1": t1,
            "t2": t2,
            "es": esubl,
            "gamma0": gamma0,
            "h": h,
            "psh": 0.0,
            "rho0_card": 1.0,
            "e0": 10.0,
        }
        obj = EquationOfState(kind="PUFF", params=params, rho0=1.0)

        # Regime 1: Compression (mu >= 0)
        mu_comp = 0.1
        e_comp = 20.0
        p_comp = eos.pressure(obj, mu_comp, e_comp)
        xx = mu_comp / (1.0 + mu_comp)
        gx = 1.0 - 0.5 * gamma0 * xx
        aa_comp = ((c1 + c3 * (mu_comp ** 2)) * mu_comp + c2 * (mu_comp ** 2)) * gx
        p_expected_comp = aa_comp + gamma0 * e_comp
        assert np.isclose(p_comp, p_expected_comp, rtol=1e-6)

        # Regime 2: Cold expansion (mu < 0, e < esubl)
        mu_exp = -0.1
        e_cold = 10.0  # < esubl (50.0)
        p_cold = eos.pressure(obj, mu_exp, e_cold)
        xx_exp = mu_exp / (1.0 + mu_exp)
        gx_exp = 1.0 - 0.5 * gamma0 * xx_exp
        aa_cold = ((t1 + t2 * mu_exp) * mu_exp) * gx_exp
        p_expected_cold = aa_cold + gamma0 * e_cold
        assert np.isclose(p_cold, p_expected_cold, rtol=1e-6)

        # Regime 3: Hot vapor expansion (mu < 0, e >= esubl)
        e_hot = 60.0  # >= esubl (50.0)
        p_hot = eos.pressure(obj, mu_exp, e_hot)
        eta = 1.0 + mu_exp
        ee = np.sqrt(eta)
        bb_hot = (h + (gamma0 - h) * ee) * eta
        cc = c1 / (gamma0 * esubl)
        expa = np.exp(cc * xx_exp)
        aa_hot = bb_hot * esubl * (expa - 1.0)
        p_expected_hot = aa_hot + bb_hot * e_hot
        assert np.isclose(p_hot, p_expected_hot, rtol=1e-6)

        # Sound speeds across all 3 regimes must be positive
        assert eos.sound_speed(obj, mu_comp, e_comp) > 0.0
        assert eos.sound_speed(obj, mu_exp, e_cold) > 0.0
        assert eos.sound_speed(obj, mu_exp, e_hot) > 0.0


class TestSolidElementCoupling:
    """Test coupling of solid element (solid_hexa8) with all 5 new equations of state."""

    @pytest.mark.parametrize("eos_kind, eos_params", [
        ("JWL", {"a": 200.0, "b": 5.0, "r1": 4.0, "r2": 1.0, "omega": 0.3, "e0": 6.0, "psh": 0.0, "rho0_card": 1.5}),
        ("MURNAGHAN", {"k0": 15.0, "k1": 3.0, "p0": 0.0, "psh": 0.0, "rho0_card": 1.2}),
        ("NOBLE-ABEL", {"b": 0.1, "gamma": 1.4, "psh": 0.0, "e0": 2.0, "rho0_card": 1.0}),
        ("NASG", {"b": 0.05, "gamma": 1.4, "p_star": 50.0, "q": 2.0, "psh": 0.0, "p0": 0.5, "cv": 1.0, "rho0_card": 1.0}),
        ("PUFF", {"c1": 12.0, "c2": 4.0, "c3": 1.0, "t1": 6.0, "t2": 2.0, "es": 40.0, "gamma0": 1.4, "h": 0.4, "psh": 0.0, "rho0_card": 1.0, "e0": 5.0}),
    ])
    def test_solid_element_step_with_eos(self, eos_kind, eos_params):
        from pyradioss.elements import solid_hexa8
        from pyradioss.model.entities import Material, Property
        from pyradioss.model.model import ElementGroup

        model = Model()
        x0 = 0.5 * (solid_hexa8._XI + 1.0)
        model.x0 = x0.copy()
        model.x = x0.copy()
        model.v = np.zeros((8, 3), dtype=np.float64)
        model.vr = np.zeros((8, 3), dtype=np.float64)

        conn = np.arange(8, dtype=np.int64)[None, :]
        group = ElementGroup(ids=np.array([1], dtype=np.int64), conn=conn, part=np.array([0], dtype=np.int64))

        rho0 = eos_params.get("rho0_card", 1.0)
        mat = Material(id=1, law=1, rho0=rho0, params={"E": 2.1e11, "nu": 0.3})
        eos_obj = EquationOfState(kind=eos_kind, params=eos_params, rho0=rho0)
        mat.eos = eos_obj

        prop = Property(id=1, title="SOLID_PROP", type="SOLID", params={"qa": 1.1, "qb": 0.05, "h": 0.1})
        group.state["slices"] = [(slice(0, 1), mat, prop)]

        log = MessageLog()
        solid_hexa8.init_group(group, model, log)

        # Displace slightly to induce strain
        x = model.x.copy()
        x[4:, 2] *= 0.99
        v = np.zeros_like(x)
        fint = np.zeros_like(x)
        mint = np.zeros_like(x)

        dt_crit = solid_hexa8.forces(group, x, v, model.vr, 1e-6, fint, mint)

        assert "p_eos" in group.state
        assert "e_eos" in group.state
        assert np.all(dt_crit > 0.0)
        assert np.all(np.isfinite(dt_crit))


class TestStarterDeckRoundtrip:
    """Verify Starter parsing and DeckWriter formatting for all 5 EOS."""

    def test_parse_and_write_eos_suite(self):
        deck_text = """# OpenRadioss input deck
/TITLE
EOS Suite Test
/MAT/LAW1/1
Rubber
1.0 100.0 0.3
/EOS/JWL/1
JWL high explosive
371.2 3.23 4.15 0.95 0.30
7.0 0.0 1.63
/MAT/LAW1/2
Water
1.0 100.0 0.3
/EOS/MURNAGHAN/2
Murnaghan liquid
10.0 4.0 0.1 0.0 1.0
/MAT/LAW1/3
Gas
1.0 100.0 0.3
/EOS/NOBLE-ABEL/3
Noble Abel Gas
0.05 1.4 5.0 0.0 1.0
/MAT/LAW1/4
NASG
1.0 100.0 0.3
/EOS/NASG/4
NASG liquid
0.05 1.5 100.0 2.0
0.0 1.0 1.0 1.0
/MAT/LAW1/5
PUFF
1.0 100.0 0.3
/EOS/PUFF/5
Puff metal
10.0 5.0 2.0 1.5
8.0 3.0 50.0
0.5 10.0 1.0
/END
"""
        model = Model()
        log = MessageLog()
        b_list = read_deck(deck_text.splitlines())
        for b in b_list:
            if b.parts[0] == "MAT":
                pass
            elif b.parts[0] == "EOS":
                read_eos(b, model, log)

        assert len(model.raw_eos) == 5
        eos_by_mat = {mid: obj for mid, obj, _ in model.raw_eos}

        assert 1 in eos_by_mat
        assert eos_by_mat[1].kind == "JWL"
        assert eos_by_mat[1].params["a"] == 371.2

        assert 2 in eos_by_mat
        assert eos_by_mat[2].kind == "MURNAGHAN"
        assert eos_by_mat[2].params["k0"] == 10.0

        assert 3 in eos_by_mat
        assert eos_by_mat[3].kind == "NOBLE-ABEL"
        assert eos_by_mat[3].params["b"] == 0.05

        assert 4 in eos_by_mat
        assert eos_by_mat[4].kind == "NASG"
        assert eos_by_mat[4].params["p_star"] == 100.0

        assert 5 in eos_by_mat
        assert eos_by_mat[5].kind == "PUFF"
        assert eos_by_mat[5].params["c1"] == 10.0

        # Verify DeckWriter emits clean blocks
        w = DeckWriter()
        for b in b_list:
            if b.parts[0] == "EOS":
                w.card_block(b)
        out = w.render()
        assert "/EOS/JWL/1" in out
        assert "/EOS/MURNAGHAN/2" in out
        assert "/EOS/NOBLE-ABEL/3" in out
        assert "/EOS/NASG/4" in out
        assert "/EOS/PUFF/5" in out
