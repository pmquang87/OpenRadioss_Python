"""
Milestone M615: /MAT/LAW46 (/MAT/LES_FLUID) Viscous Fluid / Foam Model Unit Tests.
==================================================================================

Upstream Fortran References:
- Starter reader: ``starter/source/materials/mat/mat046/hm_read_mat46.F``
- Engine handler: ``engine/source/materials/mat/mat046/m46law.F``
- Constitutive kernel: ``engine/source/materials/mat/mat046/sigeps46.F``
- Card layout: ``hm_cfg_files/config/CFG/radioss110/MAT/matl46_les_fluid.cfg``
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law46_les_fluid import (
    Law46Params,
    build_law46,
    consistent_solid_tangent,
    shell_update,
    solid_tangent,
    solid_update,
    sound_speed,
)
from pyradioss.model.entities import Material, MaterialLaw46
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_mat_law46, check_model


def _parse_deck(tmp_path: Path, deck_text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "LAW46_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(p))
    parse_starter_deck(blocks, model, log)
    return model, log


# ============================================================================
# 1. Parameter Handling and Defaults (hm_read_mat46.F)
# ============================================================================

def test_law46_params_defaults():
    """Verify Fortran defaults in hm_read_mat46.F:118-138."""
    # Default istf=1, smag=0 -> defaults to smag=0.1, smag2=0.01
    p1 = Law46Params(rho0=1000.0, c=1500.0, nu=1.0e-3, istf=1, smag=0.0)
    assert p1.c1 == 1000.0 * (1500.0 ** 2)
    assert pytest.approx(p1.smag2) == 0.01
    assert p1.ca == 0.0

    # istf=0 (no SGS): smag2=0
    p0 = Law46Params(rho0=1000.0, c=1500.0, nu=1.0e-3, istf=0, smag=0.2)
    assert p0.smag2 == 0.0

    # istf=2 with cps=0 -> cps=smag -> ca = (cps/smag)^2 = 1.0
    p2 = Law46Params(rho0=1000.0, c=1500.0, nu=1.0e-3, istf=2, smag=0.2, cps=0.0)
    assert pytest.approx(p2.smag2) == 0.04
    assert pytest.approx(p2.ca) == 1.0

    # istf=2 with explicit cps
    p2b = Law46Params(rho0=1000.0, c=1500.0, nu=1.0e-3, istf=2, smag=0.2, cps=0.1)
    assert pytest.approx(p2b.ca) == (0.1 / 0.2) ** 2


# ============================================================================
# 2. Starter Parsing: /MAT/LAW46 and /MAT/LES_FLUID (Fixed and Free)
# ============================================================================

def test_mat_law46_starter_parsing(tmp_path: Path):
    """Test reading /MAT/LAW46 and /MAT/LES_FLUID decks."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW46_TEST
/MAT/LAW46/401
Foam Fluid Fixed
#              RHO_I               RHO_0
              1000.0              1000.0
#                  C                  NU
              1500.0              0.0015
#               Istf                Smag                 Cps
                   2                 0.2                 0.1
/MAT/LES_FLUID/402
Foam Fluid Free
1200.0
1400.0 0.002
1 0.15 0.0
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    assert 401 in model.mat_law46s
    m1 = model.mat_law46s[401]
    assert m1.id == 401
    assert m1.title == "Foam Fluid Fixed"
    assert pytest.approx(m1.rho0) == 1000.0
    assert pytest.approx(m1.c) == 1500.0
    assert pytest.approx(m1.nu) == 0.0015
    assert m1.istf == 2
    assert pytest.approx(m1.smag) == 0.2
    assert pytest.approx(m1.cps) == 0.1

    assert 402 in model.mat_law46s
    m2 = model.mat_law46s[402]
    assert pytest.approx(m2.rho0) == 1200.0
    assert pytest.approx(m2.c) == 1400.0
    assert pytest.approx(m2.nu) == 0.002
    assert m2.istf == 1
    assert pytest.approx(m2.smag) == 0.15


# ============================================================================
# 3. Model Checking: Validations and Diagnostics
# ============================================================================

def test_law46_parameter_checks():
    """Verify check_mat_law46 error reporting."""
    log = MessageLog()
    # Bad density
    bad_rho = Material(id=1, law=46, rho0=-1.0, params={"MAT_C": 1000.0, "MAT_NU": 0.01})
    check_mat_law46(mat=bad_rho, log=log)
    assert any("initial density RHO must be > 0" in e for e in log.errors)

    # Bad sound speed
    log.errors.clear()
    bad_c = Material(id=2, law=46, rho0=1000.0, params={"MAT_C": -50.0, "MAT_NU": 0.01})
    check_mat_law46(mat=bad_c, log=log)
    assert any("speed of sound C must be >= 0" in e for e in log.errors)

    # Bad viscosity
    log.errors.clear()
    bad_nu = Material(id=3, law=46, rho0=1000.0, params={"MAT_C": 1500.0, "MAT_NU": -0.01})
    check_mat_law46(mat=bad_nu, log=log)
    assert any("viscosity NU must be >= 0" in e for e in log.errors)

    # Bad subgrid flag
    log.errors.clear()
    bad_istf = Material(id=4, law=46, rho0=1000.0, params={"MAT_C": 1500.0, "MAT_NU": 0.01, "Istf": 9})
    check_mat_law46(mat=bad_istf, log=log)
    assert any("invalid subgrid scale model" in e for e in log.errors)


# ============================================================================
# 4. Constitutive Update (sigeps46.F) — Physics Parity
# ============================================================================

def test_law46_pure_shear_stress():
    """Verify shear stress tau = 2 * mu_eff * e_dot (Navier-Stokes Newtonian limit)."""
    p = Law46Params(rho0=1000.0, c=1500.0, nu=0.005, istf=0)  # No SGS, pure molecular viscosity
    sig = np.zeros((1, 6))
    dt = 1.0e-3
    gamma_xy_dot = 100.0  # s^-1
    deps = np.zeros((1, 6))
    deps[0, 3] = gamma_xy_dot * dt

    sig_out, _, _ = solid_update(p, sig, deps, dt=dt)

    # mu_eff = rho * nu = 1000 * 0.005 = 5.0 Pa.s
    # sig_xy = mu_eff * gamma_xy_dot = 5.0 * 100.0 = 500.0 Pa
    expected_tau = 5.0 * 100.0
    assert pytest.approx(sig_out[0, 3], rel=1e-5) == expected_tau
    # Hydrostatic stress should be zero when rho == rho0
    assert pytest.approx(sig_out[0, 0]) == 0.0
    assert pytest.approx(sig_out[0, 1]) == 0.0
    assert pytest.approx(sig_out[0, 2]) == 0.0


def test_law46_hydrostatic_pressure():
    """Verify hydrostatic pressure P = C1 * (rho/rho0 - 1) (sigeps46.F:142-150)."""
    rho0 = 1000.0
    c = 1500.0
    c1 = rho0 * (c ** 2)  # 2.25 GPa
    p = Law46Params(rho0=rho0, c=c, nu=0.0, istf=0)

    sig = np.zeros((1, 6))
    dt = 1.0e-4
    deps = np.zeros((1, 6))

    # 1% compression: current_rho = 1.01 * rho0
    extra = {"rho": np.array([1010.0])}
    sig_out, _, c_sound = solid_update(p, sig, deps, dt=dt, extra=extra)

    # sigma_n = C1 * (1 - rho/rho0) = 2.25e9 * (1 - 1.01) = -2.25e7 Pa (compression)
    expected_sigma = c1 * (1.0 - 1.01)
    assert pytest.approx(sig_out[0, 0], rel=1e-5) == expected_sigma
    assert pytest.approx(sig_out[0, 1], rel=1e-5) == expected_sigma
    assert pytest.approx(sig_out[0, 2], rel=1e-5) == expected_sigma
    assert pytest.approx(sig_out[0, 3]) == 0.0

    # Sound speed: sqrt(C1 / rho) = sqrt(2.25e9 / 1010) = 1492.55 m/s
    assert pytest.approx(c_sound[0], rel=1e-4) == math.sqrt(c1 / 1010.0)


def test_law46_smagorinsky_sgs_scaling():
    """Verify Smagorinsky eddy viscosity nu_sgs = Smag^2 * |S| * Delta (sigeps46.F:190)."""
    rho0 = 1.2
    c = 340.0
    nu_mol = 1.5e-5
    smag = 0.15
    p = Law46Params(rho0=rho0, c=c, nu=nu_mol, istf=1, smag=smag)

    sig = np.zeros((1, 6))
    dt = 1.0e-3
    # Pure shear strain rate
    gamma_xy_dot = 200.0
    deps = np.zeros((1, 6))
    deps[0, 3] = gamma_xy_dot * dt

    vol = 8.0e-6  # m^3 (2cm cube)
    # Delta = Vol^(2/3) = (8e-6)^(2/3) = 4e-4 m^2
    delta = vol ** (2.0 / 3.0)

    extra = {"rho": np.array([rho0]), "vol": np.array([vol])}
    sig_out, _, _ = solid_update(p, sig, deps, dt=dt, extra=extra)

    # SS calculation (sigeps46.F:161):
    # Dxy = 0.5 * gamma_xy_dot = 100.0
    # SS = sqrt(2 * Dxy^2) = sqrt(2 * 10000) = 141.421356
    ss = math.sqrt(2.0 * (100.0 ** 2))

    # nu_sgs = smag^2 * SS * delta = 0.0225 * 141.421356 * 4e-4 = 0.00127279 Pa.s / rho
    nu_sgs = (smag ** 2) * ss * delta
    nu_eff = nu_mol + nu_sgs
    mu_eff = rho0 * nu_eff
    expected_tau = mu_eff * gamma_xy_dot

    assert pytest.approx(sig_out[0, 3], rel=1e-4) == expected_tau


def test_law46_acoustic_damping():
    """Verify acoustic pressure damping under volumetric expansion/compression (sigeps46.F:220-224)."""
    p = Law46Params(rho0=1000.0, c=1500.0, nu=0.01, istf=2, smag=0.2, cps=0.2)
    sig = np.zeros((1, 6))
    dt = 1.0e-4

    # Volumetric compression rate
    deps = np.full((1, 6), 0.0)
    d_diag = -10.0 * dt  # Dxx = Dyy = Dzz = -10 s^-1
    deps[0, 0] = d_diag
    deps[0, 1] = d_diag
    deps[0, 2] = d_diag

    extra = {"rho": np.array([1000.0]), "vol": np.array([1.0e-3])}
    sig_out, _, _ = solid_update(p, sig, deps, dt=dt, extra=extra)

    # In pure spherical strain rate: Dxx = Dyy = Dzz = dav = -10
    # Deviatoric D' = 0
    # Viscous pressure = mu_2 * dav = 3 * rho * ca * nu_1 * dav
    # Here ca = 1.0, dav = -10.0, producing compressive viscous pressure
    assert sig_out[0, 0] < 0.0
    assert sig_out[0, 0] == sig_out[0, 1] == sig_out[0, 2]


def test_law46_positive_dissipation():
    """Verify second law of thermodynamics (positive viscous work dissipation sigma_v : D >= 0)."""
    p = Law46Params(rho0=1000.0, c=1500.0, nu=0.05, istf=1, smag=0.1)
    np.random.seed(42)

    for _ in range(10):
        deps = np.random.randn(1, 6) * 1.0e-3
        dt = 1.0e-4
        d_rate = deps / dt
        # Make trace zero to check pure deviatoric dissipation
        tr = (d_rate[0, 0] + d_rate[0, 1] + d_rate[0, 2]) / 3.0
        deps[0, 0] -= tr * dt
        deps[0, 1] -= tr * dt
        deps[0, 2] -= tr * dt

        sig = np.zeros((1, 6))
        sig_out, _, _ = solid_update(p, sig, deps, dt=dt, extra={"rho": np.array([1000.0])})

        # Stress power = sig : D
        d_rate = deps / dt
        power = (
            sig_out[0, 0] * d_rate[0, 0]
            + sig_out[0, 1] * d_rate[0, 1]
            + sig_out[0, 2] * d_rate[0, 2]
            + sig_out[0, 3] * d_rate[0, 3]
            + sig_out[0, 4] * d_rate[0, 4]
            + sig_out[0, 5] * d_rate[0, 5]
        )
        assert power >= 0.0, f"Negative dissipation detected: {power}"


# ============================================================================
# 5. Tangent Matrix
# ============================================================================

def test_law46_tangent():
    """Verify continuum tangent matrix structure."""
    p = Law46Params(rho0=1000.0, c=1500.0, nu=0.01)
    k_bulk = p.c1
    dt = 1.0e-3
    c_mat = solid_tangent(p, dt=dt)
    assert c_mat.shape == (6, 6)
    # Check bulk modulus on diagonal
    assert c_mat[0, 0] > k_bulk
    # Symmetry
    assert np.allclose(c_mat, c_mat.T)
    # Tangent builder alias
    assert np.allclose(consistent_solid_tangent(p, dt=dt), c_mat)


# ============================================================================
# 6. Full Starter Model Check with Brick Elements
# ============================================================================

def test_full_model_check_brick_law46(tmp_path: Path):
    """Verify check_model passes with solid brick elements referencing /MAT/LAW46."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW46_SOLID_CHECK
/TITLE
LAW46 BRICK TEST
/MAT/LAW46/1
Foam_Fluid
#              RHO_I
              1000.0
#                  C                  NU
              1500.0               0.005
#               Istf                Smag                 Cps
                   1                 0.1                 0.0
/PROP/SOLID/1
Solid_Prop
         0         0         0         0         0         0         0
/PART/1
Solid_Part
         1         1         0
/NODE
         1       0.0       0.0       0.0
         2       1.0       0.0       0.0
         3       1.0       1.0       0.0
         4       0.0       1.0       0.0
         5       0.0       0.0       1.0
         6       1.0       0.0       1.0
         7       1.0       1.0       1.0
         8       0.0       1.0       1.0
/BRICK/1
         1         1         2         3         4         5         6         7         8
/END
"""
    p = tmp_path / "LAW46_0000.rad"
    p.write_text(deck, encoding="ascii")
    from pyradioss.starter.starter import run_starter
    log = MessageLog()
    model = run_starter(str(p), log=log)
    assert not log.errors, f"Starter errors: {log.errors}"
    assert model.bricks is not None
    assert model.bricks.n == 1
    assert model.bricks.ids[0] == 1
    assert model.materials[1].law == 46
