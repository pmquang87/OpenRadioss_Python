# -*- coding: utf-8 -*-
"""Tests for Milestone M589 Failure Models Wave 2:
- /FAIL/PUCK: Puck 3D composite failure criterion (Fiber & IFF modes A, B, C)
- /FAIL/COCKCROFT: Cockcroft-Latham ductile fracture criterion
- /FAIL/HC_DSSE: Hosford-Coulomb fracture locus with DSSE
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model import Model
from pyradioss.model.entities import FailureModel
import pyradioss.failure as failure
import pyradioss.failure.puck as puck_module
import pyradioss.failure.cockcroft as cockcroft_module
import pyradioss.failure.hc_dsse as hc_dsse_module


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    """Helper to parse a starter deck string."""
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


# ============================================================================
# 1. /FAIL/PUCK Tests
# ============================================================================

class TestPuckFailure:
    """Test Puck 3D composite failure criterion."""

    params = {
        "sigma_1t": 1200.0,
        "sigma_2t": 60.0,
        "sigma_12": 80.0,
        "sigma_1c": 900.0,
        "sigma_2c": 180.0,
        "p12_pos": 0.35,
        "p12_neg": 0.25,
        "p22_neg": 0.25,
        "tau_max": 1.0e-5,
        "fcut": 0.0,
        "ifail_sh": 1,
        "ifail_so": 1,
    }

    def test_fiber_tension_and_compression(self):
        """Longitudinal fiber mode f1 = sxx/s1t (tension) or -sxx/s1c (compression)."""
        fm = FailureModel(type="PUCK", ifail_sh=1, params=self.params)

        # Tensile fiber: sxx = 600 -> f1 = 0.5
        sig_tens = np.array([[600.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        dama = np.zeros(1)
        broken = failure.solid_step(fm, sig_tens, np.zeros(1), None, 1e-4, dama)
        assert not broken[0]
        assert pytest.approx(dama[0], abs=1e-5) == 0.5

        # Critical tensile fiber: sxx = 1200 -> f1 = 1.0 -> broken
        sig_crit_t = np.array([[1200.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        dama = np.zeros(1)
        broken = failure.solid_step(fm, sig_crit_t, np.zeros(1), None, 1e-4, dama)
        assert broken[0]
        assert pytest.approx(dama[0], abs=1e-5) == 1.0

        # Compressive fiber: sxx = -450 -> f1 = 450/900 = 0.5
        sig_comp = np.array([[-450.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        dama = np.zeros(1)
        broken = failure.solid_step(fm, sig_comp, np.zeros(1), None, 1e-4, dama)
        assert not broken[0]
        assert pytest.approx(dama[0], abs=1e-5) == 0.5

        # Critical compressive fiber: sxx = -900 -> f1 = 1.0 -> broken
        sig_crit_c = np.array([[-900.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        dama = np.zeros(1)
        broken = failure.solid_step(fm, sig_crit_c, np.zeros(1), None, 1e-4, dama)
        assert broken[0]
        assert pytest.approx(dama[0], abs=1e-5) == 1.0

    def test_iff_mode_a(self):
        """Inter-Fiber Failure Mode A under transverse tension (syy >= 0)."""
        # Pure transverse tension: syy = 60.0 (sigma_2t = 60.0), sxy = 0.0
        # fac = (1 - p12_pos * s2t/s12) * syy/s2t = (1 - 0.35 * 60/80) * 1.0
        # fa = sqrt(fac^2) + p12_pos * syy/s12 = 1 - 0.35*60/80 + 0.35*60/80 = 1.0
        sig = np.array([[0.0, 60.0, 0.0]])
        f1, fa, fb, fc = puck_module.compute_puck_modes(sig, self.params)
        assert pytest.approx(f1[0]) == 0.0
        assert pytest.approx(fa[0], abs=1e-5) == 1.0
        assert pytest.approx(fb[0]) == 0.0
        assert pytest.approx(fc[0]) == 0.0

        # Half transverse tension: syy = 30.0 -> fa = 0.5
        sig_half = np.array([[0.0, 30.0, 0.0]])
        _, fa_half, _, _ = puck_module.compute_puck_modes(sig_half, self.params)
        assert pytest.approx(fa_half[0], abs=1e-5) == 0.5

    def test_iff_mode_b(self):
        """Inter-Fiber Failure Mode B under moderate transverse compression (syy < 0)."""
        # Pure in-plane shear: syy = -0.001 (very small neg), sxy = 80.0 (sigma_12 = 80.0)
        # fb = [sqrt(sxy^2 + (pn12*syy)^2) + pn12*syy] / s12
        sig_shear = np.array([[0.0, -1e-6, 80.0]])
        _, _, fb, _ = puck_module.compute_puck_modes(sig_shear, self.params)
        assert pytest.approx(fb[0], abs=1e-4) == 1.0

        # Moderate compression + shear: syy = -40.0, sxy = 40.0
        # fb = [sqrt(40^2 + (0.25*(-40))^2) + 0.25*(-40)] / 80 = [sqrt(1600 + 100) - 10] / 80
        expected_fb = (np.sqrt(40.0**2 + (0.25 * (-40.0))**2) + 0.25 * (-40.0)) / 80.0
        sig_comb = np.array([[0.0, -40.0, 40.0]])
        _, _, fb_comb, _ = puck_module.compute_puck_modes(sig_comb, self.params)
        assert pytest.approx(fb_comb[0], abs=1e-5) == expected_fb

    def test_iff_mode_c(self):
        """Inter-Fiber Failure Mode C under high transverse compression (syy << 0)."""
        # Pure transverse compression: syy = -180.0 (sigma_2c = 180.0), sxy = 0.0
        # fc = (syy/s2c)^2 * (-s2c / syy) = 1.0 * (180.0 / 180.0) = 1.0!
        sig_comp2 = np.array([[0.0, -180.0, 0.0]])
        _, _, _, fc = puck_module.compute_puck_modes(sig_comp2, self.params)
        assert pytest.approx(fc[0], abs=1e-5) == 1.0

        # Critical failure via Mode C:
        fm = FailureModel(type="PUCK", ifail_sh=1, params=self.params)
        dama = np.zeros(1)
        broken = failure.shell_step(fm, sig_comp2, np.zeros(1), None, 1e-4, dama)
        assert broken[0]
        assert pytest.approx(dama[0], abs=1e-5) == 1.0

    def test_3d_solid_through_thickness_transverse(self):
        """Through-thickness stress szz triggers transverse modes direction 3 in solids."""
        fm = FailureModel(type="PUCK", ifail_sh=1, params=self.params)

        # szz = 60.0 (tensile transverse through thickness)
        sig_z = np.array([[0.0, 0.0, 60.0, 0.0, 0.0, 0.0]])
        dama = np.zeros(1)
        broken = failure.solid_step(fm, sig_z, np.zeros(1), None, 1e-4, dama)
        assert broken[0]
        assert pytest.approx(dama[0], abs=1e-5) == 1.0

        # szz = -180.0 (compressive transverse through thickness)
        sig_zc = np.array([[0.0, 0.0, -180.0, 0.0, 0.0, 0.0]])
        dama = np.zeros(1)
        broken = failure.solid_step(fm, sig_zc, np.zeros(1), None, 1e-4, dama)
        assert broken[0]
        assert pytest.approx(dama[0], abs=1e-5) == 1.0


# ============================================================================
# 2. /FAIL/COCKCROFT Tests
# ============================================================================

class TestCockcroftFailure:
    """Test Cockcroft-Latham ductile fracture criterion."""

    def test_uniaxial_tension_work_accumulation_and_deletion(self):
        """Plastic work damage accumulation under tension until element deletion."""
        c0 = 500.0  # W_crit
        fm = FailureModel(type="COCKCROFT", ifail_sh=1, params={"c0": c0, "alpha": 1.0})

        sig = np.array([[250.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        dama = np.zeros(1)

        # Cycle 1: deps_p = 0.5 -> dD = 250 * 0.5 / 500 = 0.25
        b1 = failure.solid_step(fm, sig, np.array([0.5]), None, 1e-4, dama)
        assert not b1[0]
        assert pytest.approx(dama[0], abs=1e-5) == 0.25

        # Cycle 2: deps_p = 1.0 -> dD = 250 * 1.0 / 500 = 0.50 -> D = 0.75
        b2 = failure.solid_step(fm, sig, np.array([1.0]), None, 1e-4, dama)
        assert not b2[0]
        assert pytest.approx(dama[0], abs=1e-5) == 0.75

        # Cycle 3: deps_p = 0.5 -> dD = 250 * 0.5 / 500 = 0.25 -> D = 1.0 -> broken!
        b3 = failure.solid_step(fm, sig, np.array([0.5]), None, 1e-4, dama)
        assert b3[0]
        assert pytest.approx(dama[0], abs=1e-5) == 1.0

    def test_hydrostatic_compression_zero_damage(self):
        """Hydrostatic compression has sigma_1 < 0 -> <sigma_1> = 0 -> zero damage."""
        c0 = 300.0
        fm = FailureModel(type="COCKCROFT", ifail_sh=1, params={"c0": c0, "alpha": 1.0})

        # All normal stresses compressive (-400.0)
        sig_hydro_c = np.array([[-400.0, -400.0, -400.0, 0.0, 0.0, 0.0]])
        dama = np.zeros(1)

        # Large plastic strain increment
        broken = failure.solid_step(fm, sig_hydro_c, np.array([5.0]), None, 1e-4, dama)
        assert not broken[0]
        assert dama[0] == 0.0  # Exactly 0 damage!

        # Also for shells: bi-axial compression in plane stress:
        sig_shell_c = np.array([[-200.0, -200.0, 0.0]])
        dama_sh = np.zeros(1)
        broken_sh = failure.shell_step(fm, sig_shell_c, np.array([3.0]), None, 1e-4, dama_sh)
        assert not broken_sh[0]
        assert dama_sh[0] == 0.0

    def test_normalized_cockcroft_latham(self):
        """Normalized Cockcroft-Latham D = int (<sigma_1> / sigma_vm) d(eps_p) / W_crit."""
        c0 = 2.0  # W_crit in terms of plastic strain
        fm = FailureModel(type="COCKCROFT", ifail_sh=1,
                          params={"c0": c0, "alpha": 1.0, "normalized": True})

        # Uniaxial tension: sxx = 300 -> sigma_1 = 300, sigma_vm = 300 -> ratio = 1.0
        sig_ut = np.array([[300.0, 0.0, 0.0]])
        dama = np.zeros(1)

        # Increment deps_p = 1.0 -> dD = 1.0 * 1.0 / 2.0 = 0.5
        b1 = failure.shell_step(fm, sig_ut, np.array([1.0]), None, 1e-4, dama)
        assert not b1[0]
        assert pytest.approx(dama[0], abs=1e-5) == 0.5

        # Increment deps_p = 1.0 -> dD = 0.5 -> D = 1.0 -> broken
        b2 = failure.shell_step(fm, sig_ut, np.array([1.0]), None, 1e-4, dama)
        assert b2[0]
        assert pytest.approx(dama[0], abs=1e-5) == 1.0

    def test_total_strain_mode(self):
        """Negative C0 uses total strain increment instead of plastic strain."""
        c0 = -200.0  # negative -> total strain mode
        fm = FailureModel(type="COCKCROFT", ifail_sh=1, params={"c0": c0, "alpha": 1.0})

        sig = np.array([[100.0, 0.0, 0.0]])
        # Total strain increment deps = [0.01, -0.003, 0.0]
        deps = np.array([[0.01, -0.003, 0.0]])
        dama = np.zeros(1)

        # solid/shell step calculates d_eeq from deps
        failure.shell_step(fm, sig, np.zeros(1), deps, 1e-4, dama)
        assert dama[0] > 0.0


# ============================================================================
# 3. /FAIL/HC_DSSE Tests
# ============================================================================

class TestHosfordCoulombDSSE:
    """Test Hosford-Coulomb fracture locus and DSSE necking."""

    def test_analytical_fracture_locus_states(self):
        """Verify fracture strain at pure shear, uniaxial tension, and equibiaxial tension."""
        a = 1.5
        b = 0.45   # uniaxial tensile fracture strain
        c = 0.12
        n_f = 1.0

        # 1. Uniaxial tension (eta = 1/3, theta_bar = 1.0): eps_f = b exactly!
        eps_ut = hc_dsse_module.compute_hc_fracture_strain(1.0 / 3.0, 1.0, a, b, c, n_f)
        assert pytest.approx(eps_ut, abs=1e-6) == b

        # 2. Equibiaxial tension (eta = 2/3, theta_bar = -1.0): eps_f = b exactly!
        eps_ebt = hc_dsse_module.compute_hc_fracture_strain(2.0 / 3.0, -1.0, a, b, c, n_f)
        assert pytest.approx(eps_ebt, abs=1e-6) == b

        # 3. Pure shear (eta = 0.0, theta_bar = 0.0):
        # g_HC = 1/sqrt(3) * (1 + 2^(a-1))^(1/a)
        ghc_shear = (1.0 / np.sqrt(3.0)) * ((1.0 + 2.0**(a - 1.0)) ** (1.0 / a))
        expected_shear = b * ((1.0 + c) / ghc_shear) ** (1.0 / n_f)
        eps_shear = hc_dsse_module.compute_hc_fracture_strain(0.0, 0.0, a, b, c, n_f)
        assert pytest.approx(eps_shear, rel=1e-5) == expected_shear

    def test_stress_tensor_invariants_shell(self):
        """Verify shell_step computes triaxiality and Lode parameter from stress tensor."""
        a, b, c, d, n_f = 1.2, 0.5, 0.1, 0.5, 1.0
        fm = FailureModel(type="HC_DSSE", ifail_sh=1, params={
            "a_hc_dsse": a, "b_hc_dsse": b, "c_hc_dsse": c, "d_hc_dsse": d, "n_f": n_f
        })

        # Pure shear stress: [0, 0, 100] -> eps_f should equal eps_shear
        sig_shear = np.array([[0.0, 0.0, 100.0]])
        dama = np.zeros(1)
        failure.shell_step(fm, sig_shear, np.array([0.1]), None, 1e-4, dama)
        eps_f_expected = hc_dsse_module.compute_hc_fracture_strain(0.0, 0.0, a, b, c, n_f)
        assert pytest.approx(dama[0], rel=1e-5) == 0.1 / eps_f_expected

        # Uniaxial tension: [200, 0, 0] -> eps_f should equal b = 0.5
        sig_ut = np.array([[200.0, 0.0, 0.0]])
        dama_ut = np.zeros(1)
        failure.shell_step(fm, sig_ut, np.array([0.25]), None, 1e-4, dama_ut)
        assert pytest.approx(dama_ut[0], abs=1e-5) == 0.25 / b  # 0.25 / 0.5 = 0.5

    def test_physical_parameter_fitting(self):
        """Verify I_flag=1 fits physical tests data (shear, UT, PST, necking) per hm_read_fail_hc_dsse.F."""
        eps_shear = 0.60
        eps_ut = 0.40
        eps_pst = 0.25
        eps_inst = 0.35
        n_f = 1.0

        a_fit, b_fit, c_fit, d_fit, nf_fit = hc_dsse_module.fit_physical_parameters(
            eps_shear, eps_ut, eps_pst, eps_inst, n_f
        )

        # Uniaxial tensile fracture strain b_fit must equal eps_ut
        assert pytest.approx(b_fit, abs=1e-6) == eps_ut

        # Re-evaluating fracture strain at UT should equal eps_ut
        eval_ut = hc_dsse_module.compute_hc_fracture_strain(1.0 / 3.0, 1.0, a_fit, b_fit, c_fit, nf_fit)
        assert pytest.approx(eval_ut, abs=1e-5) == eps_ut

        # Fitted parameters match Fortran hm_read_fail_hc_dsse.F lines 2553-2580:
        assert pytest.approx(a_fit, abs=1e-3) == 1.4974
        assert pytest.approx(c_fit, abs=1e-4) == 0.08235
        assert pytest.approx(d_fit, abs=1e-3) == 1.9704

    def test_dsse_instability(self):
        """Verify DSSE necking evaluation for eta > 1/3."""
        b = 0.4
        d = 1.5
        # For eta <= 1/3, DSSE is inactive (huge strain)
        assert hc_dsse_module.compute_dsse_strain(0.2, b, d) > 1e10
        # For eta = 0.5 (biaxial region), DSSE strain is finite and positive
        dsse_05 = hc_dsse_module.compute_dsse_strain(0.5, b, d)
        assert dsse_05 > 0.0 and dsse_05 < 1e5

    def test_compressive_cutoff(self):
        """Under high triaxial compression eta < -1/3, eps_f = 100.0 (no fracture)."""
        eps_comp = hc_dsse_module.compute_hc_fracture_strain(-0.6, 0.0, 1.5, 0.4, 0.1, 1.0)
        assert eps_comp == 100.0


# ============================================================================
# 4. Starter Keyword Parsing Tests (Free & Fixed Formats)
# ============================================================================

class TestStarterKeywordParsing:
    """Test deck parsing for all three models in free and fixed format."""

    def test_puck_fixed_format(self, tmp_path: Path):
        c1 = f"{1200.0:>20.4f}{60.0:>20.4f}{80.0:>20.4f}{900.0:>20.4f}{180.0:>20.4f}"
        c2 = f"{0.35:>20.4f}{0.25:>20.4f}{0.25:>20.4f}{1.0e-5:>20.5e}{1:>10d}{1:>10d}"
        c3 = f"{500.0:>20.4f}"
        deck = f"""# RADIOSS STARTER
/BEGIN
PUCK_FIXED_TEST
/FAIL/PUCK/101
{c1}
{c2}
{c3}
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 101 in model.fail_pucks

        puck = model.fail_pucks[101]
        assert puck.mat_id == 101
        assert pytest.approx(puck.sigma_1t) == 1200.0
        assert pytest.approx(puck.sigma_2t) == 60.0
        assert pytest.approx(puck.sigma_12) == 80.0
        assert pytest.approx(puck.sigma_1c) == 900.0
        assert pytest.approx(puck.sigma_2c) == 180.0
        assert pytest.approx(puck.p12_pos) == 0.35
        assert pytest.approx(puck.p12_neg) == 0.25
        assert pytest.approx(puck.p22_neg) == 0.25
        assert pytest.approx(puck.tau_max) == 1.0e-5
        assert puck.ifail_sh == 1
        assert puck.ifail_so == 1
        assert pytest.approx(puck.fcut) == 500.0

    def test_puck_free_format(self, tmp_path: Path):
        deck = """# RADIOSS STARTER
/BEGIN
PUCK_FREE_TEST
/FAIL/PUCK/102
1200.0 60.0 80.0 900.0 180.0
0.35 0.25 0.25 1.0e-5 2 1
250.0
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 102 in model.fail_pucks
        puck = model.fail_pucks[102]
        assert puck.mat_id == 102
        assert pytest.approx(puck.sigma_1t) == 1200.0
        assert puck.ifail_sh == 2

    def test_cockcroft_fixed_format(self, tmp_path: Path):
        c1 = f"{350.0:>20.4f}{0.85:>20.4f}{2:>10d}"
        deck = f"""# RADIOSS STARTER
/BEGIN
COCKCROFT_FIXED_TEST
/FAIL/COCKCROFT/201
{c1}
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 201 in model.fail_cockcrofts

        cc = model.fail_cockcrofts[201]
        assert cc.mat_id == 201
        assert pytest.approx(cc.c0) == 350.0
        assert pytest.approx(cc.alpha) == 0.85
        assert cc.failip == 2

    def test_cockcroft_free_format(self, tmp_path: Path):
        deck = """# RADIOSS STARTER
/BEGIN
COCKCROFT_FREE_TEST
/FAIL/COCKCROFT/202
-420.0 1.0 1
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 202 in model.fail_cockcrofts
        cc = model.fail_cockcrofts[202]
        assert pytest.approx(cc.c0) == -420.0
        assert pytest.approx(cc.alpha) == 1.0
        assert cc.failip == 1

    def test_hc_dsse_fixed_format(self, tmp_path: Path):
        c1 = f"{1:>10d}{0.0:>20.4f}{0:>10d}"
        c2 = f"{1.35:>20.4f}{0.48:>20.4f}{0.11:>20.4f}{0.45:>20.4f}{1.0:>20.4f}"
        deck = f"""# RADIOSS STARTER
/BEGIN
HC_DSSE_FIXED_TEST
/FAIL/HC_DSSE/301
{c1}
{c2}
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 301 in model.fail_hc_dsses

        hc = model.fail_hc_dsses[301]
        assert hc.mat_id == 301
        assert hc.ifail_sh == 1
        assert hc.iflag == 0
        assert pytest.approx(hc.a_hc_dsse) == 1.35
        assert pytest.approx(hc.b_hc_dsse) == 0.48
        assert pytest.approx(hc.c_hc_dsse) == 0.11
        assert pytest.approx(hc.d_hc_dsse) == 0.45
        assert pytest.approx(hc.n_f) == 1.0

    def test_hc_dsse_free_format_with_physical_fitting(self, tmp_path: Path):
        deck = """# RADIOSS STARTER
/BEGIN
HC_DSSE_FREE_PHYSICAL_TEST
/FAIL/HC_DSSE/302
2 0.0 1
0.55 0.38 0.22 0.30 1.0
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 302 in model.fail_hc_dsses
        hc = model.fail_hc_dsses[302]
        assert hc.mat_id == 302
        assert hc.ifail_sh == 2
        assert hc.iflag == 1
        assert pytest.approx(hc.a_hc_dsse) == 0.55
        assert pytest.approx(hc.b_hc_dsse) == 0.38
