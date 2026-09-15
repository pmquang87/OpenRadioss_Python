"""
Milestone M484 — /FAIL/BIQUAD unit tests.

Direct unit test coverage for ``pyradioss.failure.biquad``:
- Parabola fitting (``_parabola``)
- Preset material coefficients (``M_flag`` 1..7, 99, and default mild steel)
- Slope control modes (``S_flag`` 1 vs 2, plane-strain minimum)
- Triaxiality-dependent failure strain (``eps_f``, continuity, bounds)
- 3D solid damage step (``solid_step``, triaxiality, failure detection)
- Plane-stress shell damage step (``shell_step``, triaxiality, failure detection)
- Starter keyword parsing (/FAIL/BIQUAD in free and fixed format)
"""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from pyradioss.failure import biquad
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck


# ============================================================================
# 1. Parabola Fitting (_parabola)
# ============================================================================
class TestParabolaFitting:
    """Tests for the 3-point quadratic polynomial solver _parabola(x0, y0, x1, y1, x2, y2)."""

    def test_canonical_parabola(self):
        """y = x^2 through (-1, 1), (0, 0), (1, 1)."""
        a, b, c = biquad._parabola(-1.0, 1.0, 0.0, 0.0, 1.0, 1.0)
        assert a == pytest.approx(1.0)
        assert b == pytest.approx(0.0)
        assert c == pytest.approx(0.0)

    def test_offset_parabola(self):
        """y = 2*x^2 - 3*x + 5 through (0, 5), (1, 4), (2, 7)."""
        a, b, c = biquad._parabola(0.0, 5.0, 1.0, 4.0, 2.0, 7.0)
        assert a == pytest.approx(2.0)
        assert b == pytest.approx(-3.0)
        assert c == pytest.approx(5.0)

    def test_linear_degenerate(self):
        """Collinear points y = 2*x + 3 through (0, 3), (1, 5), (2, 7) -> a=0."""
        a, b, c = biquad._parabola(0.0, 3.0, 1.0, 5.0, 2.0, 7.0)
        assert a == pytest.approx(0.0, abs=1e-12)
        assert b == pytest.approx(2.0)
        assert c == pytest.approx(3.0)

    def test_constant_function(self):
        """Constant function y = 4 through (-1/3, 4), (0, 4), (1/3, 4)."""
        a, b, c = biquad._parabola(-1.0 / 3.0, 4.0, 0.0, 4.0, 1.0 / 3.0, 4.0)
        assert a == pytest.approx(0.0, abs=1e-12)
        assert b == pytest.approx(0.0, abs=1e-12)
        assert c == pytest.approx(4.0)


# ============================================================================
# 2. Material Presets (M_flag) in fit()
# ============================================================================
class TestPresetsMFlag:
    """Tests for biquad_coefficients.F material presets (M_flag 1..7, 99)."""

    def test_mflag_1_mild_steel(self):
        """M_flag=1: Mild Steel (c1=3.5*c3, c2=1.6*c3, c4=0.6*c3, c5=1.5*c3)."""
        c3 = 0.6
        params = {"c3": c3, "m_flag": 1, "s_flag": 1}
        biquad.fit(params)
        assert params["c1"] == pytest.approx(3.5 * c3)
        assert params["c2"] == pytest.approx(1.6 * c3)
        assert params["c4"] == pytest.approx(0.6 * c3)
        assert params["c5"] == pytest.approx(1.5 * c3)

    def test_mflag_2_dp600(self):
        """M_flag=2: DP600 (c1=4.3*c3, c2=1.4*c3, c4=0.6*c3, c5=1.6*c3)."""
        c3 = 0.5
        params = {"c3": c3, "m_flag": 2, "s_flag": 1}
        biquad.fit(params)
        assert params["c1"] == pytest.approx(4.3 * c3)
        assert params["c2"] == pytest.approx(1.4 * c3)
        assert params["c4"] == pytest.approx(0.6 * c3)
        assert params["c5"] == pytest.approx(1.6 * c3)

    def test_mflag_3_boron(self):
        """M_flag=3: Boron (c1=5.2*c3, c2=3.1*c3, c4=0.8*c3, c5=3.5*c3)."""
        c3 = 0.12
        params = {"c3": c3, "m_flag": 3, "s_flag": 1}
        biquad.fit(params)
        assert params["c1"] == pytest.approx(5.2 * c3)
        assert params["c2"] == pytest.approx(3.1 * c3)
        assert params["c4"] == pytest.approx(0.8 * c3)
        assert params["c5"] == pytest.approx(3.5 * c3)

    def test_mflag_4_aa5182(self):
        """M_flag=4: Aluminium AA5182 (c1=5.0*c3, c2=1.0*c3, c4=0.4*c3, c5=0.8*c3)."""
        c3 = 0.3
        params = {"c3": c3, "m_flag": 4, "s_flag": 1}
        biquad.fit(params)
        assert params["c1"] == pytest.approx(5.0 * c3)
        assert params["c2"] == pytest.approx(1.0 * c3)
        assert params["c4"] == pytest.approx(0.4 * c3)
        assert params["c5"] == pytest.approx(0.8 * c3)

    def test_mflag_5_aa6082_t6(self):
        """M_flag=5: Aluminium AA6082-T6 (c1=7.8*c3, c2=3.5*c3, c4=0.6*c3, c5=2.8*c3)."""
        c3 = 0.17
        params = {"c3": c3, "m_flag": 5, "s_flag": 1}
        biquad.fit(params)
        assert params["c1"] == pytest.approx(7.8 * c3)
        assert params["c2"] == pytest.approx(3.5 * c3)
        assert params["c4"] == pytest.approx(0.6 * c3)
        assert params["c5"] == pytest.approx(2.8 * c3)

    def test_mflag_6_pa6gf30(self):
        """M_flag=6: Plastic PA6GF30 (c1=3.6*c3, c2=0.6*c3, c4=0.5*c3, c5=0.6*c3)."""
        c3 = 0.1
        params = {"c3": c3, "m_flag": 6, "s_flag": 1}
        biquad.fit(params)
        assert params["c1"] == pytest.approx(3.6 * c3)
        assert params["c2"] == pytest.approx(0.6 * c3)
        assert params["c4"] == pytest.approx(0.5 * c3)
        assert params["c5"] == pytest.approx(0.6 * c3)

    def test_mflag_7_pp_t40(self):
        """M_flag=7: Plastic PP T40 (c1=10.0*c3, c2=2.7*c3, c4=0.6*c3, c5=0.7*c3)."""
        c3 = 0.11
        params = {"c3": c3, "m_flag": 7, "s_flag": 1}
        biquad.fit(params)
        assert params["c1"] == pytest.approx(10.0 * c3)
        assert params["c2"] == pytest.approx(2.7 * c3)
        assert params["c4"] == pytest.approx(0.6 * c3)
        assert params["c5"] == pytest.approx(0.7 * c3)

    def test_mflag_99_user_scaling(self):
        """M_flag=99: User scaling factors e1..e4."""
        c3 = 0.5
        params = {
            "c3": c3, "m_flag": 99, "s_flag": 1,
            "e1": 2.5, "e2": 1.2, "e3": 0.7, "e4": 1.8
        }
        biquad.fit(params)
        assert params["c1"] == pytest.approx(2.5 * c3)
        assert params["c2"] == pytest.approx(1.2 * c3)
        assert params["c4"] == pytest.approx(0.7 * c3)
        assert params["c5"] == pytest.approx(1.8 * c3)

    def test_all_zero_except_c3_defaults_to_mild_steel(self):
        """When c1, c2, c4, c5 are 0, fit() defaults to mild steel preset."""
        c3 = 0.4
        params = {"c3": c3, "c1": 0.0, "c2": 0.0, "c4": 0.0, "c5": 0.0, "s_flag": 1}
        biquad.fit(params)
        assert params["c1"] == pytest.approx(3.5 * c3)
        assert params["c2"] == pytest.approx(1.6 * c3)
        assert params["c4"] == pytest.approx(0.6 * c3)
        assert params["c5"] == pytest.approx(1.5 * c3)

    def test_explicit_coefficients_preserved(self):
        """Explicitly given nonzero c1..c5 without M_flag must be preserved."""
        params = {
            "c1": 0.8, "c2": 0.5, "c3": 0.3, "c4": 0.2, "c5": 0.4,
            "m_flag": 0, "s_flag": 1
        }
        biquad.fit(params)
        assert params["c1"] == 0.8
        assert params["c2"] == 0.5
        assert params["c3"] == 0.3
        assert params["c4"] == 0.2
        assert params["c5"] == 0.4


# ============================================================================
# 3. Slope Control (S_flag) in fit()
# ============================================================================
class TestSFlagBranches:
    """Tests for S_flag branches (S_flag=1 single parabola vs S_flag=2 split parabolas)."""

    def test_sflag_1_single_high_parabola(self):
        """S_flag=1 fits a single high parabola through (1/3, c3), (2/3, c4), (1, c5)."""
        params = {
            "c1": 1.0, "c2": 0.6, "c3": 0.4, "c4": 0.25, "c5": 0.5,
            "m_flag": 0, "s_flag": 1
        }
        biquad.fit(params)
        assert "phigh" in params
        assert "phigh_1" not in params
        assert "phigh_2" not in params

    def test_sflag_2_split_parabolas(self):
        """S_flag=2 creates two sub-parabolas phigh_1 and phigh_2 meeting at 1/sqrt(3)."""
        params = {
            "c1": 1.0, "c2": 0.6, "c3": 0.4, "c4": 0.25, "c5": 0.5,
            "m_flag": 0, "s_flag": 2
        }
        biquad.fit(params)
        assert "phigh_1" in params
        assert "phigh_2" in params
        assert "phigh" not in params

    def test_sflag_3_fallback_to_2(self):
        """S_flag=3 with non-positive or excessive inst_start falls back to S_flag=2."""
        params = {
            "c1": 1.0, "c2": 0.6, "c3": 0.4, "c4": 0.25, "c5": 0.5,
            "m_flag": 0, "s_flag": 3, "inst_start": 0.0
        }
        biquad.fit(params)
        assert "phigh_1" in params
        assert "phigh_2" in params


# ============================================================================
# 4. Failure Strain (eps_f)
# ============================================================================
class TestEpsFCalculation:
    """Tests for eps_f(fail, triax) interpolation and continuity."""

    @pytest.fixture
    def fail_s1(self):
        """Failure object with S_flag=1."""
        params = {
            "c1": 0.9, "c2": 0.5, "c3": 0.3, "c4": 0.2, "c5": 0.4,
            "m_flag": 0, "s_flag": 1
        }
        biquad.fit(params)
        return SimpleNamespace(params=params)

    @pytest.fixture
    def fail_s2(self):
        """Failure object with S_flag=2."""
        params = {
            "c1": 0.9, "c2": 0.5, "c3": 0.3, "c4": 0.2, "c5": 0.4,
            "m_flag": 0, "s_flag": 2
        }
        biquad.fit(params)
        return SimpleNamespace(params=params)

    def test_calibration_points_s1(self, fail_s1):
        """eps_f must reproduce c1..c5 at the five canonical triaxiality points."""
        triax = np.array([-1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        ef = biquad.eps_f(fail_s1, triax)
        expected = np.array([0.9, 0.5, 0.3, 0.2, 0.4])
        np.testing.assert_allclose(ef, expected, rtol=1e-12)

    def test_c0_continuity_at_transition(self, fail_s1, fail_s2):
        """eps_f must be C0 continuous across the transition triaxiality eta=1/3."""
        eta_left = np.array([1.0 / 3.0 - 1e-10])
        eta_right = np.array([1.0 / 3.0 + 1e-10])
        for fail_obj in (fail_s1, fail_s2):
            ef_l = biquad.eps_f(fail_obj, eta_left)[0]
            ef_r = biquad.eps_f(fail_obj, eta_right)[0]
            assert ef_l == pytest.approx(ef_r, abs=1e-8)
            assert ef_l == pytest.approx(0.3, abs=1e-8)

    def test_s2_continuity_and_zero_slope_at_plane_strain(self, fail_s2):
        """Under S_flag=2, phigh_1 and phigh_2 meet at eta=1/sqrt(3) with zero slope."""
        s1x = 1.0 / np.sqrt(3.0)
        h = 1e-7
        ef_l = biquad.eps_f(fail_s2, np.array([s1x - h]))[0]
        ef_r = biquad.eps_f(fail_s2, np.array([s1x + h]))[0]
        ef_c = biquad.eps_f(fail_s2, np.array([s1x]))[0]

        # Value continuity
        assert ef_l == pytest.approx(ef_c, abs=1e-8)
        assert ef_r == pytest.approx(ef_c, abs=1e-8)

        # Zero slope (finite difference derivative)
        deriv = (ef_r - ef_l) / (2 * h)
        assert deriv == pytest.approx(0.0, abs=1e-6)

    def test_floor_bound_on_extrapolation(self, fail_s1):
        """Extreme triaxialities where the parabola dips negative must be clamped to _FLOOR."""
        triax = np.array([-10.0, 10.0])
        ef = biquad.eps_f(fail_s1, triax)
        assert np.all(ef >= biquad._FLOOR)

    def test_vectorized_vs_scalar(self, fail_s1, fail_s2):
        """Vectorized evaluation must match element-by-element evaluation."""
        triax = np.linspace(-0.5, 1.2, 50)
        for fail_obj in (fail_s1, fail_s2):
            vec_res = biquad.eps_f(fail_obj, triax)
            for i, t in enumerate(triax):
                scalar_res = biquad.eps_f(fail_obj, np.array([t]))[0]
                assert vec_res[i] == pytest.approx(scalar_res, abs=1e-12)


# ============================================================================
# 5. 3D Solid Damage Step (solid_step)
# ============================================================================
class TestSolidStep:
    """Tests for solid_step(fail, sig, d_epsp, deps, dt, dama)."""

    @pytest.fixture
    def fail_obj(self):
        params = {
            "c1": 0.8, "c2": 0.4, "c3": 0.2, "c4": 0.15, "c5": 0.3,
            "m_flag": 0, "s_flag": 1
        }
        biquad.fit(params)
        return SimpleNamespace(params=params)

    def test_pure_shear_damage(self, fail_obj):
        """Pure shear stress (sigma_xy > 0, normals 0) -> triax = 0 -> eps_f = c2."""
        sig = np.array([[0.0, 0.0, 0.0, 100.0, 0.0, 0.0]])
        d_epsp = np.array([0.1])
        dama = np.zeros(1)

        broken = biquad.solid_step(fail_obj, sig, d_epsp, deps=None, dt=1e-4, dama=dama)

        # Expected damage increment = 0.1 / c2 = 0.1 / 0.4 = 0.25
        assert not broken[0]
        assert dama[0] == pytest.approx(0.25)

    def test_uniaxial_tension_damage(self, fail_obj):
        """Uniaxial tension (sigma_xx > 0, others 0) -> triax = 1/3 -> eps_f = c3."""
        sig = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        d_epsp = np.array([0.05])
        dama = np.zeros(1)

        broken = biquad.solid_step(fail_obj, sig, d_epsp, deps=None, dt=1e-4, dama=dama)

        # Expected damage increment = 0.05 / c3 = 0.05 / 0.2 = 0.25
        assert not broken[0]
        assert dama[0] == pytest.approx(0.25)

    def test_failure_trigger_at_damage_one(self, fail_obj):
        """When accumulated damage reaches or exceeds 1.0, element is broken."""
        sig = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        d_epsp = np.array([0.25])  # 0.25 / 0.2 = 1.25 > 1.0
        dama = np.zeros(1)

        broken = biquad.solid_step(fail_obj, sig, d_epsp, deps=None, dt=1e-4, dama=dama)

        assert broken[0]
        assert dama[0] == pytest.approx(1.25)

    def test_zero_plastic_strain_no_damage(self, fail_obj):
        """Zero plastic strain increment does not increase damage."""
        sig = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        d_epsp = np.array([0.0])
        dama = np.array([0.5])

        broken = biquad.solid_step(fail_obj, sig, d_epsp, deps=None, dt=1e-4, dama=dama)

        assert not broken[0]
        assert dama[0] == pytest.approx(0.5)

    def test_solid_step_signature_accepts_tstar(self, fail_obj):
        """solid_step signature accepts tstar for compatibility across failure criteria."""
        sig = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        d_epsp = np.array([0.05])
        dama = np.zeros(1)
        tstar = np.array([0.5])
        broken = biquad.solid_step(fail_obj, sig, d_epsp, deps=None, dt=1e-4, dama=dama, tstar=tstar)
        assert not broken[0]
        assert dama[0] == pytest.approx(0.25)


# ============================================================================
# 6. Plane-Stress Shell Damage Step (shell_step)
# ============================================================================
class TestShellStep:
    """Tests for shell_step(fail, sig, d_epsp, deps, dt, dama)."""

    @pytest.fixture
    def fail_obj(self):
        params = {
            "c1": 0.8, "c2": 0.4, "c3": 0.2, "c4": 0.15, "c5": 0.3,
            "m_flag": 0, "s_flag": 1
        }
        biquad.fit(params)
        return SimpleNamespace(params=params)

    def test_shell_pure_shear(self, fail_obj):
        """Plane-stress pure shear (sigma_12 > 0) -> triax = 0 -> eps_f = c2."""
        sig = np.array([[0.0, 0.0, 50.0]])
        d_epsp = np.array([0.08])
        dama = np.zeros(1)

        broken = biquad.shell_step(fail_obj, sig, d_epsp, deps=None, dt=1e-4, dama=dama)

        assert not broken[0]
        # 0.08 / 0.4 = 0.20
        assert dama[0] == pytest.approx(0.20)

    def test_shell_uniaxial_tension(self, fail_obj):
        """Plane-stress uniaxial tension (sigma_11 > 0, others 0) -> triax = 1/3 -> eps_f = c3."""
        sig = np.array([[150.0, 0.0, 0.0]])
        d_epsp = np.array([0.1])
        dama = np.zeros(1)

        broken = biquad.shell_step(fail_obj, sig, d_epsp, deps=None, dt=1e-4, dama=dama)

        assert not broken[0]
        # 0.1 / 0.2 = 0.50
        assert dama[0] == pytest.approx(0.50)

    def test_shell_equibiaxial_tension(self, fail_obj):
        """Plane-stress equibiaxial tension (sigma_11 = sigma_22 > 0) -> triax = 2/3 -> eps_f = c4."""
        sig = np.array([[100.0, 100.0, 0.0]])
        d_epsp = np.array([0.075])
        dama = np.zeros(1)

        broken = biquad.shell_step(fail_obj, sig, d_epsp, deps=None, dt=1e-4, dama=dama)

        assert not broken[0]
        # 0.075 / 0.15 = 0.50
        assert dama[0] == pytest.approx(0.50)

    def test_shell_step_signature_accepts_tstar_and_eps_tot(self, fail_obj):
        """shell_step signature accepts tstar and eps_tot for compatibility."""
        sig = np.array([[100.0, 100.0, 0.0]])
        d_epsp = np.array([0.075])
        dama = np.zeros(1)
        tstar = np.array([0.5])
        eps_tot = np.array([[0.01, 0.01, 0.0]])
        broken = biquad.shell_step(
            fail_obj, sig, d_epsp, deps=None, dt=1e-4, dama=dama,
            tstar=tstar, eps_tot=eps_tot
        )
        assert not broken[0]
        assert dama[0] == pytest.approx(0.50)


# ============================================================================
# 7. Starter Keyword Integration (/FAIL/BIQUAD)
# ============================================================================
class TestStarterIntegration:
    """Tests for parsing /FAIL/BIQUAD in Starter decks."""

    def test_free_format_biquad(self, tmp_path: Path):
        """Parse free-format /FAIL/BIQUAD deck and verify parameters and parabolas."""
        deck = """# RADIOSS STARTER DECK
/BEGIN
Test Free Format Biquad
/MAT/LAW1/1
steel
7.8e-6
210000.0 0.3
/FAIL/BIQUAD/1
0.8, 0.4, 0.2, 0.15, 0.3
/END
"""
        p = tmp_path / "TEST_0000.rad"
        p.write_text(deck, encoding="utf-8")
        model = Model()
        log = MessageLog()
        blocks = read_deck(str(p))
        parse_starter_deck(blocks, model, log)

        assert len(log.errors) == 0
        assert 1 in model.fails_biquad
        fb = model.fails_biquad[1]
        assert fb.c1 == pytest.approx(0.8)
        assert fb.c2 == pytest.approx(0.4)
        assert fb.c3 == pytest.approx(0.2)
        assert fb.c4 == pytest.approx(0.15)
        assert fb.c5 == pytest.approx(0.3)
        # Verify parabolas were fitted in raw_fails
        assert len(model.raw_fails) == 1
        fm = model.raw_fails[0][1]
        assert "plow" in fm.params
        assert "phigh" in fm.params or "phigh_1" in fm.params

    def test_fixed_format_biquad_preset(self, tmp_path: Path):
        """Parse fixed-format /FAIL/BIQUAD deck with M_flag preset."""
        # Card 1: C1..C5 in 5x20 fields; here only C3=0.5 is given, M_flag=2 on card 2
        c1 = f"{'':20s}{'':20s}{0.5:>20.4f}{'':20s}{'':20s}"
        # Card 2 layout FAIL_BIQUAD_2: [20, 10, 10, 20] -> p_thickfail(20), m_flag(10), s_flag(10), inst_start(20)
        c2 = f"{'':20s}{2:>10d}{1:>10d}{'':20s}"
        mat_rho = f"{7.8e-6:>20.6e}"
        mat_e_nu = f"{210000.0:>20.4f}{0.3:>20.4f}"
        deck = f"""# RADIOSS STARTER DECK
/BEGIN
Test Fixed Format Biquad Preset
2022 0
/MAT/LAW1/1
steel
{mat_rho}
{mat_e_nu}
/FAIL/BIQUAD/1
DP600 Preset
{c1}
{c2}
/END
"""
        p = tmp_path / "TEST_0000.rad"
        p.write_text(deck, encoding="utf-8")
        model = Model()
        log = MessageLog()
        blocks = read_deck(str(p))
        parse_starter_deck(blocks, model, log)

        assert len(log.errors) == 0
        assert 1 in model.fails_biquad
        fb = model.fails_biquad[1]
        # DP600 preset: c1=4.3*c3, c2=1.4*c3, c4=0.6*c3, c5=1.6*c3
        assert fb.c1 == pytest.approx(4.3 * 0.5)
        assert fb.c2 == pytest.approx(1.4 * 0.5)
        assert fb.c3 == pytest.approx(0.5)
        assert fb.c4 == pytest.approx(0.6 * 0.5)
        assert fb.c5 == pytest.approx(1.6 * 0.5)
        assert fb.m_flag == 2
        assert fb.s_flag == 1
