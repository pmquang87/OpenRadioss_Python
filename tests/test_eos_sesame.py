"""Tests for /EOS/SESAME equation of state.

Upstream Fortran reference:
  common_source/eos/sesame.F
  common_source/eos/mintp_re.F
  common_source/eos/mintp_rt.F
  common_source/eos/mintp1_rt.F
  common_source/eos/minter1d_rat.F
  engine/source/materials/mat/mat026/mindex.F
  starter/source/materials/mat/mat026/mrdse2.F
  starter/source/materials/eos/sesame_tools.F
"""

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.materials.eos import (
    mindex_1d,
    minter1d_rat,
    mintp1_rt,
    mintp_re,
    read_sesame_file,
    coefficients,
    update,
    pressure,
    sound_speed,
    initial_state,
)
from pyradioss.model.entities import EquationOfState


def test_mindex_1d():
    """Verify 1-based index binary search matching mindex.F."""
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert mindex_1d(arr, 5.0) == 1
    assert mindex_1d(arr, 15.0) == 1
    assert mindex_1d(arr, 25.0) == 2
    assert mindex_1d(arr, 35.0) == 3
    assert mindex_1d(arr, 45.0) == 4
    assert mindex_1d(arr, 100.0) == 4


def test_mintp1_rt_2d_interpolation():
    """Verify 2D rational interpolation Z(x, y) and partial derivatives dZ/dx, dZ/dy."""
    # 4 density points and 4 temperature points
    r_tab = np.array([2000.0, 2500.0, 2700.0, 3000.0])
    t_tab = np.array([300.0, 600.0, 900.0, 1200.0])

    # Construct P(rho, T) = 5.0e7 * (rho / 2700)^2 + 1.0e5 * T
    p_mat = np.zeros((len(r_tab), len(t_tab)), dtype=float)
    for i, r in enumerate(r_tab):
        for j, t in enumerate(t_tab):
            p_mat[i, j] = 5.0e7 * (r / 2700.0) ** 2 + 1.0e5 * t

    # Evaluate at intermediate point
    r_eval = 2600.0
    t_eval = 750.0

    p_val, dpdr, dpdt = mintp1_rt(r_tab, t_tab, p_mat, r_eval, t_eval)

    p_expected = 5.0e7 * (r_eval / 2700.0) ** 2 + 1.0e5 * t_eval
    # Rational interpolation should match smooth quadratic/linear surface closely
    assert p_val == pytest.approx(p_expected, rel=0.05)
    assert dpdr > 0.0
    assert dpdt > 0.0


def test_mintp_re_inverse_interpolation():
    """Verify inverse rational interpolation finding T(rho, e) given internal energy e."""
    r_tab = np.array([2000.0, 2400.0, 2700.0, 3100.0])
    t_tab = np.array([300.0, 500.0, 800.0, 1200.0])

    # Specific internal energy: e(rho, T) = 900.0 * T
    e_mat = np.zeros((len(r_tab), len(t_tab)), dtype=float)
    for i, r in enumerate(r_tab):
        for j, t in enumerate(t_tab):
            e_mat[i, j] = 900.0 * t

    r_eval = 2550.0
    t_target = 650.0
    e_target = 900.0 * t_target

    t_calc, dtde = mintp_re(r_tab, t_tab, e_mat, r_eval, e_target)
    assert t_calc == pytest.approx(t_target, rel=0.02)
    # dt/de = 1 / 900 = ~0.00111
    assert dtde == pytest.approx(1.0 / 900.0, rel=0.05)


def test_sesame_tabulated_eos_evaluation():
    """Verify full SESAME EOS update and acoustic sound speed."""
    rho0 = 2700.0
    r_tab = np.array([2000.0, 2400.0, 2700.0, 3000.0, 3300.0])
    t_tab = np.array([300.0, 500.0, 800.0, 1200.0, 1600.0])

    # Construct P and E tables
    p_mat = np.zeros((len(r_tab), len(t_tab)), dtype=float)
    e_mat = np.zeros((len(r_tab), len(t_tab)), dtype=float)

    # Bulk modulus ~ 7.0e10, specific heat ~ 900 J/(kg.K)
    for i, r in enumerate(r_tab):
        eta = r / rho0
        for j, t in enumerate(t_tab):
            p_mat[i, j] = 7.0e10 * (eta - 1.0) + 2.0e6 * (t - 300.0)
            e_mat[i, j] = 900.0 * (t - 300.0) + 1.0e4

    eos = EquationOfState(
        kind="SESAME",
        rho0=rho0,
        params={
            "rho0_card": rho0,
            "rho_table": r_tab,
            "theta_table": t_tab,
            "p_table": p_mat,
            "e_table": e_mat,
            "e0": rho0 * 1.0e4,  # E0 in J/m^3
            "pmin": -1.0e11,
            "psh": 0.0,
        },
    )

    # 1. Initial state
    e0_init, p0_init = initial_state(eos)
    assert e0_init == pytest.approx(rho0 * 1.0e4, rel=1e-6)
    assert abs(p0_init) < 1.0e7  # Near 0 at eta=1, T=300

    # 2. Pressure evaluation in compression (mu = 0.05)
    mu_val = 0.05
    E_val = e0_init + 5.0e7
    p_calc = pressure(eos, mu_val, E_val)
    assert p_calc > 0.0

    # 3. Sound speed evaluation
    c_bulk = sound_speed(eos, mu_val, E_val)
    # sqrt(K/rho) ~ sqrt(7e10 / 2700) ~ 5091 m/s
    assert 4000.0 < c_bulk < 6500.0

    # 4. Implicit update
    mu_arr = np.array([0.05])
    dv_arr = np.array([-0.048])
    e_old = np.array([e0_init])
    p_old = np.array([p0_init])
    de_oth = np.array([0.0])

    p_new, e_new, c2 = update(eos, mu_arr, dv_arr, e_old, p_old, de_oth)
    assert p_new[0] > 0.0
    assert e_new[0] > e_old[0]
    assert c2[0] > 0.0


def test_sesame_read_ascii_file(tmp_path: Path):
    """Verify parsing standard SESAME ASCII format 301 file."""
    # Create synthetic SESAME format 301 file
    # Format:
    # line 1: title
    # line 2: NR, NT
    # line 3: B1, B2
    # next: RR (in Mg/m^3 = g/cm^3)
    # next: TT (in K)
    # next: PP (in GPa)
    # next: EE (in MJ/kg)
    nr, nt = 4, 3
    r_vals = [2.0, 2.5, 2.7, 3.0]  # Mg/m^3 -> 2000, 2500, 2700, 3000 kg/m^3
    t_vals = [300.0, 600.0, 1000.0]

    # Generate P in GPa and E in MJ/kg
    p_flat = []
    e_flat = []
    for t in t_vals:
        for r in r_vals:
            p_flat.append(70.0 * (r / 2.7 - 1.0) + 0.001 * t)
            e_flat.append(0.0009 * (t - 300.0) + 0.01)

    file_content = "SESAME 301 SYNTHETIC TEST\n"
    file_content += f"{float(nr)}  {float(nt)}\n"
    file_content += "13.0  27.0\n"  # B1, B2
    file_content += "  ".join(f"{r:.4e}" for r in r_vals) + "\n"
    file_content += "  ".join(f"{t:.4e}" for t in t_vals) + "\n"
    file_content += "  ".join(f"{p:.4e}" for p in p_flat) + "\n"
    file_content += "  ".join(f"{e:.4e}" for e in e_flat) + "\n"

    table_file = tmp_path / "sesame_synthetic_301.dat"
    table_file.write_text(file_content, encoding="utf-8")

    tab = read_sesame_file(str(table_file))
    assert tab["nr"] == nr
    assert tab["nt"] == nt
    assert tab["rho_table"][0] == pytest.approx(2000.0)
    assert tab["rho_table"][2] == pytest.approx(2700.0)
    assert tab["p_table"].shape == (nr, nt)
    assert tab["e_table"].shape == (nr, nt)

    # Use filename in EquationOfState
    eos = EquationOfState(
        kind="SESAME",
        rho0=2700.0,
        params={"filename": str(table_file), "e0": 2.7e7, "rho0_card": 2700.0},
    )

    p_val = pressure(eos, 0.0, 2.7e7)
    assert np.isfinite(p_val)
    c_val = sound_speed(eos, 0.0, 2.7e7)
    assert c_val > 0.0


def test_sesame_fallback_without_table():
    """Verify safe analytic fallback when SESAME EOS has no tabular data."""
    eos = EquationOfState(
        kind="SESAME",
        rho0=1000.0,
        params={"e0": 1.0e5, "rho0_card": 1000.0, "pmin": 0.0, "psh": 0.0},
    )

    e0, p0 = initial_state(eos)
    assert e0 == 1.0e5
    assert p0 == 0.0

    p_calc = pressure(eos, 0.05, 1.0e5)
    assert p_calc > 0.0

    c_calc = sound_speed(eos, 0.05, 1.0e5)
    assert c_calc > 0.0

    p_new, e_new, c2 = update(
        eos,
        np.array([0.02]),
        np.array([-0.019]),
        np.array([1.0e5]),
        np.array([0.0]),
        np.array([0.0]),
    )
    assert p_new[0] > 0.0
    assert e_new[0] > 1.0e5
    assert c2[0] > 0.0
