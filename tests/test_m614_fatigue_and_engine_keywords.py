"""
Test Suite for Milestone M614 Components 5 & 6:
Extended Fatigue Models & freimpl.F Engine Keyword Parsing / Round-Trip Deck Serialization.

Physics references:
- freimpl.F: C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\input\\freimpl.F
- Steinberg (1988): Vibration Analysis for Electronic Equipment.
- Zhao & Baker (1992): On the probability density function of rainflow stress range for stationary Gaussian processes.
- Ramberg & Osgood (1943), Coffin (1954), Manson (1953), Morrow (1968), Smith-Watson-Topper (1970).
- Neuber (1961), Molski & Glinka (1981).
"""

import math
import numpy as np
import pytest

from pyradioss.common.npcompat import trapezoid as np_trapezoid

from pyradioss.implicit.spectral_fatigue import (
    # Mean stress
    goodman_correction,
    gerber_correction,
    soderberg_correction,
    morrow_correction,
    swt_correction,
    walker_correction,
    mean_stress_correction,
    effective_sn_coefficient,
    # Steinberg
    steinberg_damage,
    # Zhao-Baker
    zhao_baker_coefficients,
    zhao_baker_range_pdf,
    zhao_baker_damage,
    # Strain-Life & Notch Plasticity
    ramberg_osgood_strain,
    ramberg_osgood_stress,
    ramberg_osgood_cyclic_strain,
    ramberg_osgood_cyclic_stress,
    coffin_manson_strain,
    coffin_manson_life,
    neuber_notch_analysis,
    neuber_strain_life,
    glinka_notch_analysis,
    glinka_strain_life,
    # Multi-slope S-N
    MultiSlopeSN,
    # Summary
    fatigue_summary,
)
from pyradioss.model.model import EngineControls
from pyradioss.input.deck_reader import read_deck
from pyradioss.common.messages import MessageLog
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.input.deck_writer import write_engine_deck


# ============================================================================
# 1. Extended Fatigue Models Tests
# ============================================================================

class TestMeanStressCorrections:
    def test_goodman(self):
        sa = 100.0
        sm = 50.0
        su = 400.0
        # Seq = 100 / (1 - 50/400) = 100 / (7/8) = 800 / 7 = 114.2857
        seq = goodman_correction(sa, sm, su)
        assert pytest.approx(seq, rel=1e-5) == 800.0 / 7.0

        # Compressive mean stress with ignore_compressive=True (default)
        assert goodman_correction(sa, -50.0, su) == sa
        # With ignore_compressive=False
        assert goodman_correction(sa, -50.0, su, ignore_compressive=False) < sa

    def test_gerber(self):
        sa = 100.0
        sm = 100.0
        su = 400.0
        # Seq = 100 / (1 - (100/400)^2) = 100 / (1 - 1/16) = 100 / (15/16) = 1600 / 15
        seq = gerber_correction(sa, sm, su)
        assert pytest.approx(seq, rel=1e-5) == 1600.0 / 15.0

    def test_soderberg(self):
        sa = 100.0
        sm = 60.0
        sy = 300.0
        # Seq = 100 / (1 - 60/300) = 100 / (0.8) = 125.0
        seq = soderberg_correction(sa, sm, sy)
        assert pytest.approx(seq, rel=1e-5) == 125.0

    def test_morrow(self):
        sa = 100.0
        sm = 50.0
        sigf = 500.0
        # Seq = 100 / (1 - 50/500) = 100 / 0.9 = 111.111...
        seq = morrow_correction(sa, sm, sigf)
        assert pytest.approx(seq, rel=1e-5) == 100.0 / 0.9

    def test_swt(self):
        sa = 100.0
        sm = 50.0
        # Seq = sqrt((100 + 50) * 100) = sqrt(15000) = 122.474487
        seq = swt_correction(sa, sm)
        assert pytest.approx(seq, rel=1e-5) == math.sqrt(150.0 * 100.0)

    def test_walker(self):
        sa = 100.0
        sm = 50.0
        # Smax = 150, Smin = -50 -> R = -50/150 = -1/3 -> (1 - R)/2 = 2/3
        # Seq = (Smax)^(1-gamma) * Sa^gamma = 150^0.5 * 100^0.5 = 122.474487
        seq = walker_correction(sa, sm, gamma=0.5)
        r = (-50.0) / 150.0
        expected = 150.0 * (((1.0 - r) / 2.0) ** 0.5)
        assert pytest.approx(seq, rel=1e-5) == expected
        # At gamma = 0.5, Walker is identical to SWT
        assert pytest.approx(seq, rel=1e-5) == swt_correction(sa, sm)

    def test_mean_stress_dispatcher_and_effective_sn(self):
        sa = 100.0
        sm = 40.0
        su = 400.0
        seq_goodman = mean_stress_correction(sa, sm, method="goodman", su=su)
        assert pytest.approx(seq_goodman, rel=1e-5) == goodman_correction(sa, sm, su)

        # C_eff for Goodman: C_eff = C * (1 - sm/su)^m
        c = 1e12
        m = 3.0
        c_eff = effective_sn_coefficient(c, m, sm, su=su, method="goodman")
        expected_c_eff = c * ((1.0 - sm / su) ** m)
        assert pytest.approx(c_eff, rel=1e-5) == expected_c_eff


class TestSteinberg3Band:
    def test_steinberg_damage_analytic(self):
        # S-N: N = C * S^(-m)
        # S_i = 1*sigma, 2*sigma, 3*sigma
        # fractions = 0.683, 0.271, 0.0433
        sigma = 20.0
        nu_p = 50.0
        t_dur = 3600.0
        c = 1e15
        m = 3.5

        d_total, d_rate, bands = steinberg_damage(sigma, nu_p, t_dur, c, m)

        n_cycles = nu_p * t_dur
        expected_d = 0.0
        for s_factor, frac in [(1.0, 0.683), (2.0, 0.271), (3.0, 0.0433)]:
            s_val = s_factor * sigma
            n_i = frac * n_cycles
            n_cap_i = c * (s_val ** (-m))
            expected_d += n_i / n_cap_i

        assert pytest.approx(d_total, rel=1e-6) == expected_d
        assert pytest.approx(d_rate, rel=1e-6) == expected_d / t_dur
        assert len(bands) == 3
        assert pytest.approx(bands[0]["damage"] + bands[1]["damage"] + bands[2]["damage"], rel=1e-6) == d_total

    def test_steinberg_in_fatigue_summary(self):
        # Create a simple narrowband PSD
        f = np.linspace(1.0, 100.0, 200)
        psd = np.exp(-((f - 30.0) / 5.0) ** 2)
        summary = fatigue_summary(f, psd, t_dur=1000.0, c=1e15, m=3.0)
        assert "steinberg" in summary
        assert summary["steinberg"]["damage"] > 0.0
        assert summary["steinberg"]["damage_rate"] > 0.0
        assert len(summary["steinberg"]["bands"]) == 3


class TestZhaoBakerModel:
    def test_zhao_baker_narrowband_limit(self):
        # When alpha2 -> 1, Zhao-Baker converges to Narrowband (Rayleigh)
        alpha2 = 0.9999
        a, b, w = zhao_baker_coefficients(alpha2)
        # As alpha2 -> 1, w -> 0 (pure Rayleigh component)
        assert pytest.approx(w, abs=1e-3) == 0.0

        sigma = 15.0
        nu_p = 40.0
        t_dur = 1000.0
        c = 1e14
        m = 3.0

        d_zb, d_rate_zb, e_sm = zhao_baker_damage(sigma, alpha2, nu_p, t_dur, c, m)

        # Theoretical Rayleigh moment: E[S^m] = (2*sqrt(2)*sigma)^m * Gamma(1 + m/2)
        expected_esm_rayleigh = (2.0 * math.sqrt(2.0) * sigma) ** m * math.gamma(1.0 + 0.5 * m)
        assert pytest.approx(e_sm, rel=1e-3) == expected_esm_rayleigh

    def test_zhao_baker_pdf_integration(self):
        # Numerical integration of Zhao-Baker PDF should integrate to 1.0
        alpha2 = 0.75
        sigma = 10.0
        s_vals = np.linspace(0.001, 100.0, 5000)
        pdf_vals = zhao_baker_range_pdf(s_vals, sigma, alpha2)
        integral = np_trapezoid(pdf_vals, s_vals)
        assert pytest.approx(integral, rel=1e-3) == 1.0

    def test_zhao_baker_in_fatigue_summary(self):
        f = np.linspace(1.0, 100.0, 200)
        psd = 1.0 / (1.0 + (f / 20.0) ** 2)
        summary = fatigue_summary(f, psd, t_dur=1000.0, c=1e15, m=3.0)
        assert "zhao_baker" in summary
        assert summary["zhao_baker"]["damage"] > 0.0
        assert summary["zhao_baker"]["e_sm"] > 0.0


class TestStrainLifeAndNotch:
    def test_ramberg_osgood_inversion(self):
        e = 200000.0
        k = 1000.0
        n = 0.15
        sigma_target = 650.0

        # Forward: compute strain from stress
        eps = ramberg_osgood_strain(sigma_target, e, k, n)
        # Inverse: solve stress from strain
        sigma_calc = ramberg_osgood_stress(eps, e, k, n)
        assert pytest.approx(sigma_calc, rel=1e-6) == sigma_target

    def test_ramberg_osgood_cyclic_inversion(self):
        e = 210000.0
        kp = 1100.0
        np_val = 0.12
        dsig_target = 800.0

        deps = ramberg_osgood_cyclic_strain(dsig_target, e, kp, np_val)
        dsig_calc = ramberg_osgood_cyclic_stress(deps, e, kp, np_val)
        assert pytest.approx(dsig_calc, rel=1e-6) == dsig_target

    def test_coffin_manson_life(self):
        e = 200000.0
        sigf = 950.0
        b = -0.09
        epsf = 0.45
        c = -0.55

        # Pick a target life 2N_f = 20000 reversals (10000 cycles)
        revs = 20000.0
        target_life = 10000.0
        strain_amp = coffin_manson_strain(revs, e, sigf, b, epsf, c)

        # Solve for life given strain amplitude
        solved_life = coffin_manson_life(strain_amp, e, sigf, b, epsf, c)
        assert pytest.approx(solved_life, rel=1e-4) == target_life

        # Mean stress decreases life (Morrow / SWT)
        life_morrow = coffin_manson_life(strain_amp, e, sigf, b, epsf, c, mean_stress=100.0, method="morrow")
        assert life_morrow < target_life

        life_swt = coffin_manson_life(strain_amp, e, sigf, b, epsf, c, mean_stress=100.0, method="swt")
        assert life_swt < target_life

    def test_neuber_and_glinka_notch(self):
        kt = 2.5
        ds = 200.0  # nominal elastic stress range
        e = 200000.0
        kp = 1000.0
        np_val = 0.15

        # Neuber: Kt^2 * dS * de = dsig * deps
        dsig_n, deps_n = neuber_notch_analysis(kt, ds, e, kp, np_val)
        lhs_neuber = (kt ** 2) * (ds ** 2) / e
        rhs_neuber = dsig_n * deps_n
        assert pytest.approx(lhs_neuber, rel=1e-5) == rhs_neuber

        # Glinka ESED: W_notch = W_nominal
        dsig_g, deps_g = glinka_notch_analysis(kt, ds, e, kp, np_val)
        assert dsig_g > 0.0
        assert deps_g > 0.0
        # Glinka gives slightly lower strain than Neuber (known mechanics fact)
        assert deps_g < deps_n


class TestMultiSlopeSN:
    def test_multi_slope_life_and_damage(self):
        # Knee at (200 MPa, 1e6 cycles), m1 = 3.0, m2 = 5.0, endurance = 100 MPa
        sn = MultiSlopeSN(s_knee=200.0, n_knee=1e6, m1=3.0, m2=5.0, s_endurance=100.0)

        # S > S_knee
        # S = 400 MPa -> N = 1e6 * (400/200)^(-3) = 1e6 / 8 = 1.25e5
        n_high = sn.life(400.0)
        assert pytest.approx(n_high, rel=1e-6) == 1.25e5

        # S_endurance < S < S_knee
        # S = 150 MPa -> N = 1e6 * (150/200)^(-5) = 1e6 * (4/3)^5 = 4.21399e6
        n_mid = sn.life(150.0)
        assert pytest.approx(n_mid, rel=1e-6) == 1e6 * ((200.0 / 150.0) ** 5.0)

        # S <= S_endurance -> infinite life
        n_low = sn.life(80.0)
        assert math.isinf(n_low)

        # Spectral damage
        sigma = 40.0
        nu_p = 25.0
        t_dur = 1000.0
        d_spec = sn.spectral_damage(sigma, nu_p, t_dur)
        assert d_spec > 0.0


# ============================================================================
# 2. Engine Keyword Parsing & Deck Serialization Tests
# ============================================================================

class TestEngineKeywordsAndSerialization:
    def test_parse_freimpl_all_cards(self):
        engine_deck = """#RADIOSS ENGINE
/RUN/TEST_M614/1
0.5
/PRINT/-50
/DT
0.9 1.0e-7
/IMPL/LINE
/IMPL/LINE/INTER/3
/IMPL/LINE/SCAUC
/IMPL/NONL/KTANG
/IMPL/NONL/SMDIS
/IMPL/NONL/SOLVI
/IMPL/NONL/PITER/4
/IMPL/NONL/1
50 12 1.0e-5 2.0e-4
/IMPL/SOLV/3
1 100 2 1.0e-6
/IMPL/SBCS/MSGLV/2
/IMPL/SBCS/ORDER/1
/IMPL/SBCS/OUTCO
/IMPL/MUMPS/MSGLV/1
/IMPL/MUMPS/ORDER/METIS
/IMPL/MUMPS/OUTCO
/IMPL/NCYCL/STOP
500
/IMPL/RREF/INTER/2
/IMPL/RREF/LIMIT
0.01 0.99
/IMPL/DIVER/TOL
1.0e5
/IMPL/DIVER/4
/IMPL/GSTIF/OFF
/IMPL/PSTIF/OFF
/IMPL/SHPOF
/IMPL/SPRIN/NONL
/IMPL/MONVO/OFF
/IMPL/CONTR/DT/STOP
1.0e-6 0.01
/IMPL/CONTR/SHEL
0.05
/IMPL/CONTR/INTER
0.02
/IMPL/PRINT/LINE/2
/IMPL/PRINT/NONL/3
/IMPL/PRINT/STIF
1.0e-4 5 1
/IMPL/CHECK
/IMPL/QSTAT/DTSCA
1.25
/IMPL/AUTOS/ALL
/IMPL/SPRB
/IMPL/FATIG/STEINBERG
/IMPL/FATIG/ZHAO_BAKER
/IMPL/FATIG/MEAN/GOODMAN
500.0 350.0 600.0 0.5
/IMPL/FATIG/EN
210000.0 950.0 -0.09 0.45 -0.55 1100.0 0.15
/IMPL/FATIG/NOTCH/NEUBER
2.2 210000.0 1100.0 0.15
"""
        blocks = read_deck(engine_deck)
        log = MessageLog()
        ec = parse_engine_deck(blocks, log)

        # Verify /RUN & basics
        assert ec.run_name == "TEST_M614"
        assert ec.t_end == 0.5
        assert ec.print_cycles == 50
        assert ec.dt_scale == 0.9
        assert ec.dt_min == 1.0e-7

        # Verify /IMPL/LINE
        assert ec.implicit is True
        assert ec.impl_line is True
        assert ec.impl_line_ilintf == 3
        assert ec.impl_line_iscau == 1

        # Verify /IMPL/NONL
        assert ec.impl_nonl is True
        assert ec.impl_nonl_ikt == 1
        assert ec.impl_nonl_smdisp == 1
        assert ec.impl_nonl_solvnfo == 1
        assert ec.impl_nonl_ipupd == 4
        assert ec.impl_nonl_insolv == 1
        assert ec.impl_nonl_n_lim == 50
        assert ec.impl_nonl_nitol == 12
        assert ec.impl_nonl_n_tole == 1.0e-5
        assert ec.impl_nonl_n_tolf == 2.0e-4

        # Verify /IMPL/SOLV
        assert ec.impl_solv is True
        assert ec.impl_solv_isolv == 3
        assert ec.impl_solv_iprec == 1
        assert ec.impl_solv_l_lim == 100
        assert ec.impl_solv_itol == 2
        assert ec.impl_solv_l_tol == 1.0e-6
        assert ec.impl_solv_mumpsd == 100

        # Verify /IMPL/SBCS
        assert ec.impl_sbcs is True
        assert ec.impl_sbcs_msg_lvl == 2
        assert ec.impl_sbcs_b_order == 1
        assert ec.impl_sbcs_b_mcore == 1

        # Verify /IMPL/MUMPS
        assert ec.impl_mumps is True
        assert ec.impl_mumps_m_msg == 1
        assert ec.impl_mumps_m_order == 5
        assert ec.impl_mumps_m_ocore == 1

        # Verify /IMPL/NCYCL/STOP
        assert ec.impl_ncycl_stop == 500

        # Verify /IMPL/RREF
        assert ec.impl_rref == 2
        assert ec.impl_rref_irefi == 2
        assert ec.impl_rref_rf_min == 0.01
        assert ec.impl_rref_rf_max == 0.99

        # Verify /IMPL/DIVER
        assert ec.impl_diver is True
        assert ec.impl_tol_div == 1.0e5
        assert ec.impl_ndiver == 4

        # Verify /IMPL/GSTIF & /IMPL/PSTIF
        assert ec.impl_gstif is True
        assert ec.impl_gstif_ikg == 0
        assert ec.impl_pstif is True
        assert ec.impl_pstif_ikpres == 0

        # Verify /IMPL/SHPOF, /IMPL/SPRIN, /IMPL/MONVO
        assert ec.impl_shpproj_ikproj == -1
        assert ec.impl_sprin_isprn == 1
        assert ec.impl_monvo_impmv == 0

        # Verify /IMPL/CONTR
        assert ec.impl_contr is True
        assert ec.impl_contr_dt_stop == (1.0e-6, 0.01)
        assert ec.impl_contr_kz_tol == 0.05
        assert ec.impl_contr_sk_int == 0.02

        # Verify /IMPL/PRINT
        assert ec.impl_print is True
        assert ec.impl_print_line == 2
        assert ec.impl_print_nonl == 3
        assert ec.impl_print_stif_tol == 1.0e-4
        assert ec.impl_print_stif_nc == 5
        assert ec.impl_print_stif_it == 1

        # Verify /IMPL/CHECK
        assert ec.impl_check == 1

        # Verify /IMPL/QSTAT & /IMPL/AUTOS & /IMPL/SPRB
        assert ec.impl_qstat == 1
        assert ec.impl_qstat_scal_dtq == 1.25
        assert ec.impl_autos == 2
        assert ec.impl_sprb is True

        # Verify Extended Fatigue Cards
        assert ec.impl_fatig_steinberg is True
        assert ec.impl_fatig_zhao_baker is True
        assert ec.impl_fatig_mean_method == "GOODMAN"
        assert ec.impl_fatig_mean_ult == 500.0
        assert ec.impl_fatig_mean_yield == 350.0
        assert ec.impl_fatig_mean_sigf == 600.0
        assert ec.impl_fatig_mean_gamma == 0.5

        assert ec.impl_fatig_en is True
        assert ec.impl_fatig_en_e == 210000.0
        assert ec.impl_fatig_en_sigf == 950.0
        assert ec.impl_fatig_en_b == -0.09
        assert ec.impl_fatig_en_epsf == 0.45
        assert ec.impl_fatig_en_c == -0.55
        assert ec.impl_fatig_en_kp == 1100.0
        assert ec.impl_fatig_en_np == 0.15

        assert ec.impl_fatig_notch is True
        assert ec.impl_fatig_notch_method == "NEUBER"
        assert ec.impl_fatig_notch_kt == 2.2
        assert ec.impl_fatig_notch_e == 210000.0
        assert ec.impl_fatig_notch_kp == 1100.0
        assert ec.impl_fatig_notch_np == 0.15

    def test_roundtrip_deck_serialization(self):
        # Create EngineControls with representative options
        ec_orig = EngineControls(
            run_name="ROUNDTRIP",
            t_end=1.5,
            dt_scale=0.85,
            dt_min=1.0e-8,
            th_dt=0.01,
            anim_dt=0.05,
            print_cycles=25,
            implicit=True,
            impl_dt=0.005,
            impl_line=True,
            impl_line_ilintf=3,
            impl_solv=True,
            impl_solv_isolv=2,
            impl_solv_iprec=1,
            impl_solv_l_lim=50,
            impl_solv_itol=1,
            impl_solv_l_tol=1.0e-5,
            impl_sbcs=True,
            impl_sbcs_msg_lvl=1,
            impl_sbcs_b_order=2,
            impl_sbcs_b_mcore=1,
            impl_mumps=True,
            impl_mumps_m_msg=2,
            impl_mumps_m_order=5,
            impl_mumps_m_ocore=1,
            impl_ncycl_stop=1200,
            impl_rref=2,
            impl_rref_rf_min=0.02,
            impl_rref_rf_max=0.98,
            impl_diver=True,
            impl_tol_div=5000.0,
            impl_ndiver=2,
            impl_gstif=True,
            impl_gstif_ikg=0,
            impl_pstif=True,
            impl_pstif_ikpres=0,
            impl_shpproj_ikproj=-1,
            impl_sprin_isprn=1,
            impl_monvo_impmv=0,
            impl_contr=True,
            impl_contr_dt_stop=(1.0e-7, 0.05),
            impl_contr_kz_tol=0.1,
            impl_contr_sk_int=0.05,
            impl_print=True,
            impl_print_line=1,
            impl_print_nonl=2,
            impl_check=1,
            impl_fatig_steinberg=True,
            impl_fatig_zhao_baker=True,
            impl_fatig_mean_method="GERBER",
            impl_fatig_mean_ult=450.0,
            impl_fatig_mean_yield=300.0,
            impl_fatig_mean_sigf=550.0,
            impl_fatig_mean_gamma=0.6,
            impl_fatig_en=True,
            impl_fatig_en_e=200000.0,
            impl_fatig_en_sigf=900.0,
            impl_fatig_en_b=-0.08,
            impl_fatig_en_epsf=0.4,
            impl_fatig_en_c=-0.6,
            impl_fatig_en_kp=1000.0,
            impl_fatig_en_np=0.14,
            impl_fatig_notch=True,
            impl_fatig_notch_method="GLINKA",
            impl_fatig_notch_kt=2.5,
            impl_fatig_notch_e=200000.0,
            impl_fatig_notch_kp=1000.0,
            impl_fatig_notch_np=0.14,
        )

        # 1. Serialize to engine deck text
        deck_text = write_engine_deck(ec_orig)
        assert "#RADIOSS ENGINE" in deck_text
        assert "/RUN/ROUNDTRIP/1" in deck_text
        assert "/IMPL/FATIG/STEINBERG" in deck_text

        # 2. Parse back
        blocks = read_deck(deck_text)
        log = MessageLog()
        ec_parsed = parse_engine_deck(blocks, log)

        # 3. Assert all round-trip attributes match
        assert ec_parsed.run_name == ec_orig.run_name
        assert pytest.approx(ec_parsed.t_end) == ec_orig.t_end
        assert pytest.approx(ec_parsed.dt_scale) == ec_orig.dt_scale
        assert pytest.approx(ec_parsed.dt_min) == ec_orig.dt_min
        assert pytest.approx(ec_parsed.th_dt) == ec_orig.th_dt
        assert pytest.approx(ec_parsed.anim_dt) == ec_orig.anim_dt
        assert ec_parsed.print_cycles == ec_orig.print_cycles

        assert ec_parsed.implicit is True
        assert pytest.approx(ec_parsed.impl_dt) == ec_orig.impl_dt
        assert ec_parsed.impl_line is True
        assert ec_parsed.impl_line_ilintf == ec_orig.impl_line_ilintf

        assert ec_parsed.impl_solv is True
        assert ec_parsed.impl_solv_isolv == ec_orig.impl_solv_isolv
        assert ec_parsed.impl_solv_iprec == ec_orig.impl_solv_iprec
        assert ec_parsed.impl_solv_l_lim == ec_orig.impl_solv_l_lim
        assert ec_parsed.impl_solv_itol == ec_orig.impl_solv_itol
        assert pytest.approx(ec_parsed.impl_solv_l_tol) == ec_orig.impl_solv_l_tol

        assert ec_parsed.impl_sbcs is True
        assert ec_parsed.impl_sbcs_msg_lvl == ec_orig.impl_sbcs_msg_lvl
        assert ec_parsed.impl_sbcs_b_order == ec_orig.impl_sbcs_b_order
        assert ec_parsed.impl_sbcs_b_mcore == ec_orig.impl_sbcs_b_mcore

        assert ec_parsed.impl_mumps is True
        assert ec_parsed.impl_mumps_m_msg == ec_orig.impl_mumps_m_msg
        assert ec_parsed.impl_mumps_m_order == ec_orig.impl_mumps_m_order
        assert ec_parsed.impl_mumps_m_ocore == ec_orig.impl_mumps_m_ocore

        assert ec_parsed.impl_ncycl_stop == ec_orig.impl_ncycl_stop
        assert ec_parsed.impl_rref == ec_orig.impl_rref
        assert pytest.approx(ec_parsed.impl_rref_rf_min) == ec_orig.impl_rref_rf_min
        assert pytest.approx(ec_parsed.impl_rref_rf_max) == ec_orig.impl_rref_rf_max

        assert ec_parsed.impl_diver is True
        assert pytest.approx(ec_parsed.impl_tol_div) == ec_orig.impl_tol_div
        assert ec_parsed.impl_ndiver == ec_orig.impl_ndiver

        assert ec_parsed.impl_gstif is True
        assert ec_parsed.impl_gstif_ikg == ec_orig.impl_gstif_ikg
        assert ec_parsed.impl_pstif is True
        assert ec_parsed.impl_pstif_ikpres == ec_orig.impl_pstif_ikpres

        assert ec_parsed.impl_shpproj_ikproj == ec_orig.impl_shpproj_ikproj
        assert ec_parsed.impl_sprin_isprn == ec_orig.impl_sprin_isprn
        assert ec_parsed.impl_monvo_impmv == ec_orig.impl_monvo_impmv

        assert ec_parsed.impl_contr is True
        assert pytest.approx(ec_parsed.impl_contr_dt_stop[0]) == ec_orig.impl_contr_dt_stop[0]
        assert pytest.approx(ec_parsed.impl_contr_dt_stop[1]) == ec_orig.impl_contr_dt_stop[1]
        assert pytest.approx(ec_parsed.impl_contr_kz_tol) == ec_orig.impl_contr_kz_tol
        assert pytest.approx(ec_parsed.impl_contr_sk_int) == ec_orig.impl_contr_sk_int

        assert ec_parsed.impl_print is True
        assert ec_parsed.impl_print_line == ec_orig.impl_print_line
        assert ec_parsed.impl_print_nonl == ec_orig.impl_print_nonl
        assert ec_parsed.impl_check == ec_orig.impl_check

        assert ec_parsed.impl_fatig_steinberg is True
        assert ec_parsed.impl_fatig_zhao_baker is True
        assert ec_parsed.impl_fatig_mean_method == ec_orig.impl_fatig_mean_method
        assert pytest.approx(ec_parsed.impl_fatig_mean_ult) == ec_orig.impl_fatig_mean_ult

        assert ec_parsed.impl_fatig_en is True
        assert pytest.approx(ec_parsed.impl_fatig_en_e) == ec_orig.impl_fatig_en_e
        assert pytest.approx(ec_parsed.impl_fatig_en_sigf) == ec_orig.impl_fatig_en_sigf

        assert ec_parsed.impl_fatig_notch is True
        assert ec_parsed.impl_fatig_notch_method == ec_orig.impl_fatig_notch_method
        assert pytest.approx(ec_parsed.impl_fatig_notch_kt) == ec_orig.impl_fatig_notch_kt
