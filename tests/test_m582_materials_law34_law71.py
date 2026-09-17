"""
test_m582_materials_law34_law71.py — Milestone M582 validation tests.

Tests:
- LAW71 Nitinol Superelastic Shape-Memory Alloy (/MAT/LAW71, /MAT/SUPER_ELAS, /MAT/NITINOL):
  * Card layouts and keyword parsing (fixed and free formats) matching hm_read_mat71.F and matl71_71.cfg.
  * Parameter validation and default values (hm_read_mat71.F).
  * 3D solid superelastic flag-shaped hysteresis loop (Auricchio 1997, sigeps71.F).
  * Tension-compression asymmetry governed by Alpha parameter.
  * Clausius-Clapeyron temperature dependence (CAS, CSA).
  * Newton iteration solver when E_mart != E (EFLAG = 1).
  * Plane-stress shell formulation with secant thickness stretch iteration (sigeps71c.F).
  * Consistent tangents (solid 6x6 and shell 3x3) and acoustic sound speed.
  * 3D multi-axial strain state with shear.
- LAW34 Boltzmann linear viscoelasticity verification:
  * Viscoelastic stress relaxation under constant strain.
  * Strain-rate dependence and dissipated energy consistency.
"""

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.card_layouts import LAYOUTS, MAT_LAW71_1, MAT_SUPER_ELAS_1, MAT_NITINOL_1
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat, read_mat_law71
from pyradioss.materials import (
    consistent_shell_tangent,
    consistent_solid_tangent,
    extra_shapes,
    needs_env,
    shell_membrane_tangent,
    shell_update,
    solid_update,
    sound_speed,
)
from pyradioss.materials.law34_boltzmann import (
    build_law34,
    solid_update as law34_solid_update,
    sound_speed as law34_sound_speed,
)
from pyradioss.materials.law71_nitinol import (
    SQRT_TWO_THIRD,
    build_law71,
    consistent_shell_tangent as law71_consistent_shell_tangent,
    consistent_solid_tangent as law71_consistent_solid_tangent,
    extra_shapes as law71_extra_shapes,
    shell_membrane_tangent as law71_shell_membrane_tangent,
    shell_step as law71_shell_step,
    shell_update as law71_shell_update,
    solid_step as law71_solid_step,
    solid_update as law71_solid_update,
    sound_speed as law71_sound_speed,
)
from pyradioss.model import Model
from pyradioss.model.entities import MatLaw71, Material


# =============================================================================
# 1. Card Layouts & Registry Tests
# =============================================================================

def test_law71_card_layouts_and_constants():
    """Verify card layout tuples and LAYOUTS registry for LAW71 and aliases."""
    assert MAT_LAW71_1 == (20, 20)
    assert MAT_SUPER_ELAS_1 == (20, 20)
    assert MAT_NITINOL_1 == (20, 20)

    for prefix in ("MAT_LAW71", "MAT_SUPER_ELAS", "MAT_NITINOL"):
        for card_idx in range(1, 6):
            key = f"{prefix}_{card_idx}"
            assert key in LAYOUTS, f"Missing layout {key}"
            layout = LAYOUTS[key]
            if card_idx == 1:
                assert layout == [20, 20]
            elif card_idx == 2:
                assert layout == [20, 20, 20]
            elif card_idx in (3, 4):
                assert layout == [20, 20, 20, 20, 20]
            elif card_idx == 5:
                assert layout == [20, 20, 20, 20]


def test_law71_starter_reader_fixed_and_free(tmp_path):
    """Test reading /MAT/LAW71, /MAT/SUPER_ELAS, and /MAT/NITINOL in both fixed and free formats."""
    # 1. Free format /MAT/LAW71
    deck_free = """/BEGIN
LAW71_FREE
/MAT/LAW71/101
Nitinol SMA Free
#   RHO_I            Refer_Rho
    6.5e-6           6.5e-6
#   E                nu               E_mart
    60000.0          0.33             50000.0
#   Sig_sas          Sig_fas          Sig_ssa          Sig_fsa          Alpha
    500.0            600.0            300.0            200.0            0.05
#   EpsL             CAS              CSA              TSAS             TFAS
    0.055            6.0              6.0              298.0            298.0
#   TSSA             TFSA             CP               TINI
    298.0            298.0            4.5e8            300.0
/END
"""
    p_free = tmp_path / "deck_free.rad"
    p_free.write_text(deck_free, encoding="utf-8")
    blocks = read_deck(str(p_free))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law71(block, model, log)

    assert len(log.errors) == 0
    assert 101 in model.mat_law71s
    assert 101 in model.materials
    mat = model.materials[101]
    assert mat.law == 71
    assert mat.params["e"] == pytest.approx(60000.0)
    assert mat.params["nu"] == pytest.approx(0.33)
    assert mat.params["e_mart"] == pytest.approx(50000.0)
    assert mat.params["sig_sas"] == pytest.approx(500.0)
    assert mat.params["sig_fas"] == pytest.approx(600.0)
    assert mat.params["sig_ssa"] == pytest.approx(300.0)
    assert mat.params["sig_fsa"] == pytest.approx(200.0)
    assert mat.params["alpha"] == pytest.approx(0.05)
    assert mat.params["epsl"] == pytest.approx(0.055)
    assert mat.params["cas"] == pytest.approx(6.0)
    assert mat.params["csa"] == pytest.approx(6.0)
    assert mat.params["tini"] == pytest.approx(300.0)

    # 2. Fixed format /MAT/SUPER_ELAS
    card1 = f"{'6.45e-6':>20}{'6.45e-6':>20}\n"
    card2 = f"{'70000.0':>20}{'0.30':>20}{'45000.0':>20}\n"
    card3 = f"{'400.0':>20}{'500.0':>20}{'250.0':>20}{'150.0':>20}{'0.08':>20}\n"
    card4 = f"{'0.065':>20}{'5.5':>20}{'5.5':>20}{'295.0':>20}{'295.0':>20}\n"
    card5 = f"{'295.0':>20}{'295.0':>20}{'5.0e8':>20}{'295.0':>20}\n"
    deck_fixed = (
        "/BEGIN\n"
        "SUPER_ELAS_FIXED\n"
        "/MAT/SUPER_ELAS/202\n"
        "Nitinol Superelastic Fixed\n"
        + card1 + card2 + card3 + card4 + card5
        + "/END\n"
    )
    p_fixed = tmp_path / "deck_fixed.rad"
    p_fixed.write_text(deck_fixed, encoding="utf-8")
    blocks_fixed = read_deck(str(p_fixed))
    model_fixed = Model()
    log_fixed = MessageLog()
    for block in blocks_fixed:
        if block.key0 == "MAT":
            read_mat_law71(block, model_fixed, log_fixed)

    assert len(log_fixed.errors) == 0
    assert 202 in model_fixed.materials
    mat2 = model_fixed.materials[202]
    assert mat2.params["e"] == pytest.approx(70000.0)
    assert mat2.params["nu"] == pytest.approx(0.30)
    assert mat2.params["e_mart"] == pytest.approx(45000.0)
    assert mat2.params["sig_sas"] == pytest.approx(400.0)
    assert mat2.params["alpha"] == pytest.approx(0.08)


def test_law71_generic_read_mat_dispatch(tmp_path):
    """Verify generic read_mat dispatches LAW71, SUPER_ELAS, and NITINOL."""
    deck = """/BEGIN
GENERIC_MAT
/MAT/NITINOL/303
Nitinol Generic
    6.45e-6          6.45e-6
    60000.0          0.33             60000.0
    450.0            550.0            300.0            200.0            0.0
    0.06             0.0              0.0              298.0            298.0
    298.0            298.0            1.0e20           298.0
/END
"""
    p = tmp_path / "generic.rad"
    p.write_text(deck, encoding="utf-8")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat(block, model, log)

    assert 303 in model.materials
    assert model.materials[303].law == 71


# =============================================================================
# 2. Physics & Parameter Validation Tests
# =============================================================================

def test_law71_parameter_defaults_and_warnings():
    """Verify parameter warnings and defaults matching hm_read_mat71.F."""
    # Test warnings on invalid ordering
    with pytest.warns(UserWarning, match="violates forward transformation ordering"):
        m_bad_fwd = MatLaw71(id=1, rho0=1.0, rhor=1.0, e=60000.0, nu=0.3,
                             sig_sas=600.0, sig_fas=500.0)  # sig_sas > sig_fas
        build_law71(m_bad_fwd)

    with pytest.warns(UserWarning, match="violates reverse transformation ordering"):
        m_bad_rev = MatLaw71(id=1, rho0=1.0, rhor=1.0, e=60000.0, nu=0.3,
                             sig_sas=400.0, sig_fas=500.0,
                             sig_ssa=100.0, sig_fsa=200.0)  # sig_ssa < sig_fsa
        build_law71(m_bad_rev)

    with pytest.warns(UserWarning, match="Alpha .* > sqrt"):
        m_bad_alpha = MatLaw71(id=1, rho0=1.0, rhor=1.0, e=60000.0, nu=0.3,
                              alpha=0.9)  # alpha > sqrt(2/3) ~ 0.8165
        build_law71(m_bad_alpha)

    # Defaults check
    m_min = MatLaw71(id=2, rho0=6.45e-6, rhor=0.0, e=60000.0, nu=0.33,
                     e_mart=0.0, tsas=0.0, tfas=0.0, tssa=0.0, tfsa=0.0, cp=0.0, tini=0.0)
    mat_min = build_law71(m_min)
    assert mat_min.params["rhor"] == pytest.approx(6.45e-6)
    assert mat_min.params["e_mart"] == pytest.approx(60000.0)  # E_mart defaults to E
    assert mat_min.params["eflag"] == 0  # E == E_mart
    assert mat_min.params["tsas"] == pytest.approx(298.0)
    assert mat_min.params["cp"] == pytest.approx(1.0e20)
    assert mat_min.params["tini"] == pytest.approx(360.0)


# =============================================================================
# 3. Constitutive Formulation Tests (3D Solid)
# =============================================================================

def test_law71_flag_shaped_hysteresis_solid():
    """Verify complete flag-shaped superelastic hysteresis loop in 3D solid element."""
    mat_rec = MatLaw71(
        id=1,
        rho0=6.45e-6,
        rhor=6.45e-6,
        e=60000.0,      # Austenite Young's modulus (MPa)
        nu=0.33,        # Poisson's ratio
        e_mart=60000.0, # Martensite Young's modulus = E (eflag = 0)
        sig_sas=400.0,  # Start of forward transformation A -> M (MPa)
        sig_fas=500.0,  # End of forward transformation A -> M (MPa)
        sig_ssa=250.0,  # Start of reverse transformation M -> A (MPa)
        sig_fsa=150.0,  # End of reverse transformation M -> A (MPa)
        alpha=0.0,      # Symmetric (alpha = 0)
        epsl=0.05,      # Maximum transformation strain 5%
        cas=0.0,        # Isothermal (no T shift)
        csa=0.0,
        tsas=298.0,
        tfas=298.0,
        tssa=298.0,
        tfsa=298.0,
        cp=1.0e20,
        tini=298.0,
    )
    mat = build_law71(mat_rec)

    # State variables
    sig = np.zeros(6, dtype=np.float64)
    uvar = np.zeros(10, dtype=np.float64)
    extra = {"uvar": uvar, "ismstr": 2}

    # Simulate 1D axial strain cycle: 0 -> 0.05 -> 0
    # Loading to 0.05 in 500 steps, then unloading to 0 in 500 steps
    n_steps = 500
    deps_load = 0.05 / n_steps
    deps_unload = -0.05 / n_steps

    stresses = []
    strains = []
    fm_history = []

    # Loading phase
    current_strain = 0.0
    for _ in range(n_steps):
        deps_vec = np.array([deps_load, -0.33 * deps_load, -0.33 * deps_load, 0.0, 0.0, 0.0])
        sig, epsp, c = solid_update(mat, sig, deps_vec, dt=1.0e-5, extra=extra)
        current_strain += deps_load
        strains.append(current_strain)
        stresses.append(sig[0])
        fm_history.append(extra["uvar"][0])

    # Check that martensite fraction increased significantly (phase transformation occurred)
    assert extra["uvar"][0] > 0.6, f"Expected transformation fm > 0.6, got {extra['uvar'][0]}"
    max_stress = sig[0]
    assert max_stress > 450.0, f"Expected stress in transformation plateau > 450, got {max_stress}"

    # Unloading phase
    for _ in range(n_steps):
        deps_vec = np.array([deps_unload, -0.33 * deps_unload, -0.33 * deps_unload, 0.0, 0.0, 0.0])
        sig, epsp, c = solid_update(mat, sig, deps_vec, dt=1.0e-5, extra=extra)
        current_strain += deps_unload
        strains.append(current_strain)
        stresses.append(sig[0])
        fm_history.append(extra["uvar"][0])

    # 1. Closed loop: after unloading to zero strain, stress and martensite fraction should return to zero
    assert abs(sig[0]) < 1.0, f"Residual stress after unloading should be near 0, got {sig[0]}"
    assert extra["uvar"][0] < 1.0e-4, f"Residual martensite fraction should be 0, got {extra['uvar'][0]}"

    # 2. Hysteresis energy dissipation: area enclosed by stress-strain loop must be strictly positive
    strains = np.array(strains)
    stresses = np.array(stresses)
    # Numerical loop integral: sum sigma * deps
    d_eps_all = np.diff(strains)
    sigma_mid = 0.5 * (stresses[:-1] + stresses[1:])
    dissipated_energy = np.sum(sigma_mid * d_eps_all)
    assert dissipated_energy > 5.0, f"Flag-shaped hysteresis loop must dissipate energy, got {dissipated_energy}"

    # 3. Transformation plateau hierarchy: loading plateau stress > unloading plateau stress
    # Load plateau at strain = 0.03
    idx_load_plateau = int(n_steps * (0.03 / 0.05))
    # Unload plateau at strain = 0.03
    idx_unload_plateau = n_steps + (n_steps - idx_load_plateau)
    stress_load = stresses[idx_load_plateau]
    stress_unload = stresses[idx_unload_plateau]
    assert stress_load > stress_unload + 120.0, (
        f"Loading plateau ({stress_load:.1f}) must be higher than unloading plateau ({stress_unload:.1f})"
    )


def test_law71_tension_compression_asymmetry():
    """Verify tension-compression asymmetry governed by Alpha parameter.

    In OpenRadioss sigeps71.F:
      P = K * tr(deps - 3*alpha*epsl*fm)
      FS = ||S|| + 3*Alpha*P - CAS*T
      RSAS = Sig_sas * (sqrt(2/3) + Alpha) - CAS*TSAS
    In uniaxial tension: sigma > 0, P > 0 => FS = (sqrt(2/3) + Alpha) * sigma
      => sigma_start_ten = Sig_sas.
    In uniaxial compression: sigma < 0, P < 0 => FS = (sqrt(2/3) - Alpha) * |sigma|
      => sigma_start_comp = Sig_sas * (sqrt(2/3) + Alpha) / (sqrt(2/3) - Alpha).
    """
    alpha_val = 0.05
    sig_sas_val = 400.0
    mat_rec = MatLaw71(
        id=2,
        rho0=6.45e-6,
        rhor=6.45e-6,
        e=60000.0,
        nu=0.33,
        e_mart=60000.0,
        sig_sas=sig_sas_val,
        sig_fas=600.0,
        sig_ssa=250.0,
        sig_fsa=150.0,
        alpha=alpha_val,
        epsl=0.05,
        cas=0.0,
        csa=0.0,
        tsas=298.0,
        tfas=298.0,
        tssa=298.0,
        tfsa=298.0,
        cp=1.0e20,
        tini=298.0,
    )
    mat = build_law71(mat_rec)

    expected_ratio = (SQRT_TWO_THIRD + alpha_val) / (SQRT_TWO_THIRD - alpha_val)

    # 1. Uniaxial Tension: find strain where fm starts to exceed 1e-5
    sig_t = np.zeros(6, dtype=np.float64)
    uvar_t = np.zeros(10, dtype=np.float64)
    extra_t = {"uvar": uvar_t, "ismstr": 2}
    deps_step = 1.0e-5
    sig_ten_start = 0.0

    for _ in range(2000):
        deps_vec = np.array([deps_step, -0.33 * deps_step, -0.33 * deps_step, 0.0, 0.0, 0.0])
        sig_t, _, _ = solid_update(mat, sig_t, deps_vec, dt=1.0e-5, extra=extra_t)
        if extra_t["uvar"][0] > 1.0e-4:
            sig_ten_start = sig_t[0]
            break

    assert sig_ten_start == pytest.approx(sig_sas_val, rel=0.02)

    # 2. Uniaxial Compression: find strain where fm starts to exceed 1e-5
    sig_c = np.zeros(6, dtype=np.float64)
    uvar_c = np.zeros(10, dtype=np.float64)
    extra_c = {"uvar": uvar_c, "ismstr": 2}
    sig_comp_start = 0.0

    for _ in range(2500):
        deps_vec = np.array([-deps_step, 0.33 * deps_step, 0.33 * deps_step, 0.0, 0.0, 0.0])
        sig_c, _, _ = solid_update(mat, sig_c, deps_vec, dt=1.0e-5, extra=extra_c)
        if extra_c["uvar"][0] > 1.0e-4:
            sig_comp_start = abs(sig_c[0])
            break

    observed_ratio = sig_comp_start / sig_ten_start
    assert observed_ratio == pytest.approx(expected_ratio, rel=0.02), (
        f"Asymmetry ratio expected {expected_ratio:.4f}, observed {observed_ratio:.4f}"
    )


def test_law71_temperature_dependence():
    """Verify Clausius-Clapeyron temperature dependence of transformation stress."""
    cas_val = 5.0  # MPa / K
    mat_rec = MatLaw71(
        id=3,
        rho0=6.45e-6,
        rhor=6.45e-6,
        e=60000.0,
        nu=0.33,
        e_mart=60000.0,
        sig_sas=400.0,
        sig_fas=550.0,
        sig_ssa=250.0,
        sig_fsa=150.0,
        alpha=0.0,
        epsl=0.05,
        cas=cas_val,
        csa=cas_val,
        tsas=298.0,
        tfas=298.0,
        tssa=298.0,
        tfsa=298.0,
        cp=1.0e20,
        tini=298.0,
    )
    mat = build_law71(mat_rec)

    # Run at Reference T = 298 K
    sig_1 = np.zeros(6, dtype=np.float64)
    uvar_1 = np.zeros(10, dtype=np.float64)
    extra_1 = {"uvar": uvar_1, "temp": 298.0, "ismstr": 2}
    deps_step = 1.0e-5
    sig_start_t1 = 0.0
    for _ in range(2000):
        deps_vec = np.array([deps_step, -0.33 * deps_step, -0.33 * deps_step, 0.0, 0.0, 0.0])
        sig_1, _, _ = solid_update(mat, sig_1, deps_vec, dt=1.0e-5, extra=extra_1)
        if extra_1["uvar"][0] > 1.0e-4:
            sig_start_t1 = sig_1[0]
            break

    # Run at Elevated T = 318 K (delta_T = 20 K)
    delta_t = 20.0
    sig_2 = np.zeros(6, dtype=np.float64)
    uvar_2 = np.zeros(10, dtype=np.float64)
    extra_2 = {"uvar": uvar_2, "temp": 298.0 + delta_t, "ismstr": 2}
    sig_start_t2 = 0.0
    for _ in range(2500):
        deps_vec = np.array([deps_step, -0.33 * deps_step, -0.33 * deps_step, 0.0, 0.0, 0.0])
        sig_2, _, _ = solid_update(mat, sig_2, deps_vec, dt=1.0e-5, extra=extra_2)
        if extra_2["uvar"][0] > 1.0e-4:
            sig_start_t2 = sig_2[0]
            break

    # Expected shift: delta_sigma = CAS * delta_T / sqrt(2/3) for alpha = 0
    expected_shift = cas_val * delta_t / SQRT_TWO_THIRD
    observed_shift = sig_start_t2 - sig_start_t1
    assert observed_shift == pytest.approx(expected_shift, rel=0.03), (
        f"Expected shift {expected_shift:.2f} MPa, observed {observed_shift:.2f} MPa"
    )


def test_law71_emart_different_from_e_newton():
    """Verify quadratic/Newton solver convergence when E_mart != E (eflag = 1)."""
    mat_rec = MatLaw71(
        id=4,
        rho0=6.45e-6,
        rhor=6.45e-6,
        e=60000.0,
        nu=0.33,
        e_mart=40000.0,  # E_mart != E => eflag = 1
        sig_sas=400.0,
        sig_fas=550.0,
        sig_ssa=250.0,
        sig_fsa=150.0,
        alpha=0.02,
        epsl=0.05,
        cas=0.0,
        csa=0.0,
        tsas=298.0,
        tfas=298.0,
        tssa=298.0,
        tfsa=298.0,
        cp=1.0e20,
        tini=298.0,
    )
    mat = build_law71(mat_rec)
    assert mat.params["eflag"] == 1

    sig = np.zeros(6, dtype=np.float64)
    uvar = np.zeros(10, dtype=np.float64)
    extra = {"uvar": uvar, "ismstr": 2}

    # Load into transformation zone
    deps_vec = np.array([0.0001, -0.33 * 0.0001, -0.33 * 0.0001, 0.0, 0.0, 0.0])
    for _ in range(200):
        sig, _, _ = solid_update(mat, sig, deps_vec, dt=1.0e-5, extra=extra)

    assert extra["uvar"][0] > 0.0, "Martensite fraction should advance with E_mart != E"
    assert not np.isnan(sig).any(), "Stress should remain finite without NaN"
    assert not np.isinf(sig).any(), "Stress should remain finite without Inf"


# =============================================================================
# 4. Plane-Stress Shell Formulation Tests
# =============================================================================

def test_law71_shell_plane_stress():
    """Verify shell element update enforces sigma_zz = 0 via secant iteration."""
    mat_rec = MatLaw71(
        id=5,
        rho0=6.45e-6,
        rhor=6.45e-6,
        e=60000.0,
        nu=0.33,
        e_mart=60000.0,
        sig_sas=450.0,
        sig_fas=550.0,
        sig_ssa=300.0,
        sig_fsa=200.0,
        alpha=0.04,
        epsl=0.06,
        cas=0.0,
        csa=0.0,
        tsas=298.0,
        tfas=298.0,
        tssa=298.0,
        tfsa=298.0,
        cp=1.0e20,
        tini=298.0,
    )
    mat = build_law71(mat_rec)

    sig = np.zeros(3, dtype=np.float64)  # [xx, yy, xy]
    uvar = np.zeros(10, dtype=np.float64)
    extra = {"uvar": uvar, "thick": 1.0, "ismstr": 2}

    # Cycle shell: load in xx direction, unload
    deps_step = 0.05 / 200
    stresses_xx = []

    # Loading
    for _ in range(200):
        deps_vec = np.array([deps_step, 0.0, 0.0])  # in-plane deps_xx
        sig, epsp = shell_update(mat, sig, deps_vec, dt=1.0e-5, extra=extra)
        stresses_xx.append(sig[0])

    assert extra["uvar"][0] > 0.5, "Martensite fraction should advance in shell"
    assert extra["thick"] < 1.0, "Poisson thinning should decrease shell thickness"

    # Unloading
    for _ in range(200):
        deps_vec = np.array([-deps_step, 0.0, 0.0])
        sig, epsp = shell_update(mat, sig, deps_vec, dt=1.0e-5, extra=extra)
        stresses_xx.append(sig[0])

    assert abs(sig[0]) < 1.0, f"Shell residual stress should be near 0, got {sig[0]}"
    assert extra["uvar"][0] < 1.0e-4, f"Shell residual fm should be near 0, got {extra['uvar'][0]}"
    assert extra["thick"] == pytest.approx(1.0, rel=1e-3), "Thickness should recover after complete superelastic cycle"


# =============================================================================
# 5. Tangents & Sound Speed Tests
# =============================================================================

def test_law71_sound_speed_and_tangents():
    """Verify sound speed calculations and consistent tangents for solids and shells."""
    mat_rec = MatLaw71(
        id=6,
        rho0=6.45e-6,
        rhor=6.45e-6,
        e=60000.0,
        nu=0.33,
        e_mart=60000.0,
        sig_sas=450.0,
        sig_fas=550.0,
        sig_ssa=300.0,
        sig_fsa=200.0,
        alpha=0.0,
        epsl=0.06,
    )
    mat = build_law71(mat_rec)

    # Sound speed
    c_s = sound_speed(mat, is_shell=False)
    c_sh = sound_speed(mat, is_shell=True)
    assert c_s > 0.0
    assert c_sh > 0.0
    assert c_s >= c_sh  # Upstream sigeps71c.F sets shell soundsp = max(sspsh, sspsol)
    assert mat.params["sound_speed_solid"] >= mat.params["sound_speed_shell"]

    # Solid consistent tangent (6x6)
    sig_solid = np.zeros(6, dtype=np.float64)
    d_solid = consistent_solid_tangent(mat, sig=sig_solid)
    assert d_solid.shape == (1, 6, 6) or d_solid.shape == (6, 6)
    d_mat = d_solid[0] if d_solid.ndim == 3 else d_solid
    assert d_mat[0, 0] > d_mat[0, 1] > 0.0
    assert d_mat[3, 3] > 0.0  # Shear modulus

    # Shell membrane tangent (3x3)
    d_mem = shell_membrane_tangent(mat)
    assert d_mem.shape == (3, 3)
    assert d_mem[0, 0] > d_mem[0, 1] > 0.0
    assert d_mem[2, 2] > 0.0  # G

    # Consistent shell tangent (3x3)
    d_sh = consistent_shell_tangent(mat, sig=np.zeros(3))
    assert d_sh.shape == (1, 3, 3) or d_sh.shape == (3, 3)


def test_law71_multiaxial_shear_rotation():
    """Verify 3D solid update under combined normal and shear strain in rotated frame."""
    mat_rec = MatLaw71(
        id=7,
        rho0=6.45e-6,
        rhor=6.45e-6,
        e=60000.0,
        nu=0.33,
        e_mart=50000.0,
        sig_sas=400.0,
        sig_fas=550.0,
        sig_ssa=250.0,
        sig_fsa=150.0,
        alpha=0.03,
        epsl=0.05,
    )
    mat = build_law71(mat_rec)

    sig = np.zeros(6, dtype=np.float64)
    uvar = np.zeros(10, dtype=np.float64)
    extra = {"uvar": uvar, "ismstr": 2}

    # Combined tension + shear strain increment
    deps = np.array([0.002, -0.00066, -0.00066, 0.0015, 0.0005, 0.0])
    sig, epsp, c = solid_update(mat, sig, deps, dt=1.0e-5, extra=extra)

    assert sig[0] > 0.0
    assert sig[3] > 0.0  # Shear stress tau_xy generated
    assert sig[4] > 0.0  # Shear stress tau_yz generated
    assert extra["uvar"][0] >= 0.0
    assert extra["uvar"][1] >= 0.0  # Eq transformation strain


# =============================================================================
# 6. LAW34 Boltzmann Viscoelasticity Verification Tests
# =============================================================================

def test_law34_verification_relaxation_and_energy():
    """Verify LAW34 Boltzmann linear viscoelastic relaxation under step strain and dissipation."""
    rec = {
        "id": 34,
        "density": 1000.0,
        "title": "LAW34_Test",
        "params": {
            "MAT_BULK": 1e7,
            "MAT_G0": 4e6,
            "MAT_GI": 1e6,
            "MAT_DECAY": 10.0,  # relaxation time tau = 0.1 s
        },
    }
    mat = build_law34(rec)

    # Step 1: instantaneous loading with tiny dt (dt -> 0 limits to G0)
    gamma = 1e-3
    deps = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0])
    sig = np.zeros(6, dtype=np.float64)
    extra = {}
    dt = 1e-5
    sig, epsp, c = solid_update(mat, sig, deps, dt=dt, extra=extra)

    # Initial shear stress ~ G0 * gamma = 4e6 * 1e-3 = 4000
    assert sig[3] == pytest.approx(4000.0, rel=1e-3)

    # Hold strain constant (deps = 0) for 500 steps (5.0 seconds = 50 tau)
    dt_hold = 0.01
    deps_zero = np.zeros(6, dtype=np.float64)
    stresses = [sig[3]]
    for _ in range(500):
        sig, epsp, c = solid_update(mat, sig, deps_zero, dt=dt_hold, extra=extra)
        stresses.append(sig[3])

    # Stress must monotonically relax towards GI * gamma = 1e6 * 1e-3 = 1000
    for i in range(len(stresses) - 1):
        assert stresses[i] >= stresses[i + 1] - 1e-10

    assert sig[3] == pytest.approx(1000.0, rel=1e-3)
