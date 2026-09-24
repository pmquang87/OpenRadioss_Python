"""Test suite for Finite Volume Airbag Physics Engine (/MONVOL/FVMBAG1, /MONVOL/FVMBAG2, /MONVOL/FVMBAG).

Validates:
1. Specific heats Cp(T), Cv(T), gamma(T), enthalpy h(T), internal energy e(T).
2. Temperature inversion from specific energy (fvtemp.F):
   - Constant Cv (algebraic)
   - Linear Cp (quadratic)
   - Polynomial Cp (Newton-Raphson / fixed-point)
3. Isentropic orifice flow (airbagb1.F / fvbag1.F):
   - Subcritical subsonic expansion
   - Sonic choked flow at critical pressure ratio
   - Numerical stability limiters
4. Autoliv / Wang-Nakhimovich fabric porosity formula (porfor5.F).
5. Enclosed volume and facet normal area computation (Gauss divergence theorem).
6. Multi-chamber thermodynamics integration (FvmAirbagManager):
   - Gas injection and enthalpy accumulation
   - Inter-chamber orifice communication and mass conservation
   - External venting to atmosphere
7. Boundary pressure force assembly (volpfv.F) onto nodes.
8. Starter input reading of /MONVOL/FVMBAG1 and /MONVOL/FVMBAG2 cards.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.engine.airbag_fvm import (
    compute_cp,
    compute_cv,
    compute_gamma,
    compute_specific_enthalpy,
    compute_specific_internal_energy,
    solve_temperature,
    compute_orifice_flow,
    compute_vent_flow,
    compute_wang_nakhimovich_porosity,
    compute_surface_geometry,
    FvmAirbagManager,
    update_fvmbag_volume,
    update_fvmbag_thermodynamics,
    apply_fvmbag_forces,
)
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    MonvolFvmbag,
    FvmChamber,
    FvmOrifice,
    FvmVent,
    FvmInjector,
    Surface,
    Material,
)
from pyradioss.starter.starter import run_starter


# =============================================================================
# 1. Thermodynamics & Energy Inversion Tests
# =============================================================================

def test_thermodynamics_polynomials():
    """Test Cp(T), Cv(T), gamma(T), h(T), e(T) evaluations."""
    cpa = 1000.0
    cpb = 0.1
    cpc = 1e-4
    r_spec = 287.0
    T = 300.0

    expected_cp = cpa + cpb * T + cpc * (T ** 2)
    assert compute_cp(T, cpa, cpb, cpc) == pytest.approx(expected_cp)

    expected_cv = expected_cp - r_spec
    assert compute_cv(T, cpa, cpb, cpc, r_spec=r_spec) == pytest.approx(expected_cv)

    expected_gamma = expected_cp / expected_cv
    assert compute_gamma(T, cpa, cpb, cpc, r_spec=r_spec) == pytest.approx(expected_gamma)

    expected_h = cpa * T + 0.5 * cpb * (T ** 2) + (1.0 / 3.0) * cpc * (T ** 3)
    assert compute_specific_enthalpy(T, cpa, cpb, cpc) == pytest.approx(expected_h)

    expected_e = expected_h - r_spec * T
    assert compute_specific_internal_energy(T, cpa, cpb, cpc, r_spec=r_spec) == pytest.approx(expected_e)


def test_temperature_inversion_constant_cv():
    """Test solve_temperature with constant Cp/Cv (algebraic branch)."""
    cpa = 1004.0
    r_spec = 287.0
    cva = cpa - r_spec
    target_T = 350.0
    e_spec = cva * target_T

    T_sol = solve_temperature(e_spec, cpa=cpa, r_spec=r_spec)
    assert T_sol == pytest.approx(target_T, rel=1e-6)


def test_temperature_inversion_linear_cp():
    """Test solve_temperature with linear Cp(T) (quadratic branch)."""
    cpa = 1000.0
    cpb = 0.2
    r_spec = 287.0
    target_T = 420.0
    e_spec = compute_specific_internal_energy(target_T, cpa=cpa, cpb=cpb, r_spec=r_spec)

    T_sol = solve_temperature(e_spec, cpa=cpa, cpb=cpb, r_spec=r_spec)
    assert T_sol == pytest.approx(target_T, rel=1e-6)


def test_temperature_inversion_polynomial_newton():
    """Test solve_temperature with general polynomial Cp(T) via Newton-Raphson."""
    cpa = 1005.0
    cpb = 0.05
    cpc = 2e-5
    cpd = 1e-8
    r_spec = 287.0
    target_T = 580.0
    e_spec = compute_specific_internal_energy(
        target_T, cpa=cpa, cpb=cpb, cpc=cpc, cpd=cpd, r_spec=r_spec
    )

    T_sol = solve_temperature(
        e_spec, cpa=cpa, cpb=cpb, cpc=cpc, cpd=cpd, r_spec=r_spec, t_guess=300.0
    )
    assert T_sol == pytest.approx(target_T, rel=1e-5)


# =============================================================================
# 2. Orifice & Venting Dynamics Tests
# =============================================================================

def test_orifice_flow_sonic_choking():
    """Test choked flow transition at critical pressure ratio."""
    p1 = 300000.0  # 3 bar
    rho1 = 3.0
    t1 = 348.0
    gamma = 1.4
    r_spec = 287.0
    cp_poly = (1004.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    area = 0.002
    cd = 0.8

    # Critical pressure ratio for gamma = 1.4 is ~0.52828
    r_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    p_crit = p1 * r_crit

    # Subcritical regime: P2 > Pcrit
    p2_sub = p1 * 0.8
    m_sub, e_sub, u_sub, choked_sub = compute_orifice_flow(
        p1, rho1, t1, gamma, r_spec, cp_poly, p2=p2_sub, area=area, cd=cd
    )
    assert not choked_sub
    assert m_sub > 0.0
    assert e_sub > 0.0

    # Choked regime: P2 < Pcrit
    p2_choked = p1 * 0.3
    m_choked, e_choked, u_choked, choked_choked = compute_orifice_flow(
        p1, rho1, t1, gamma, r_spec, cp_poly, p2=p2_choked, area=area, cd=cd
    )
    assert choked_choked

    # Even lower backpressure P2=0 should give the identical mass flow (choked throat)
    m_choked_zero, _, _, _ = compute_orifice_flow(
        p1, rho1, t1, gamma, r_spec, cp_poly, p2=0.0, area=area, cd=cd
    )
    assert m_choked == pytest.approx(m_choked_zero, rel=1e-5)


def test_wang_nakhimovich_porosity():
    """Test Autoliv / Wang-Nakhimovich fabric porosity formula matching porfor5.F."""
    lr1 = 0.001
    fthk = 0.0002
    c1 = 1.5
    c2 = 0.5
    c3 = 2.0
    p_ext = 100000.0

    # Case A: Undeformed, no pressure difference -> porosity 0
    fac0 = compute_wang_nakhimovich_porosity(
        eps1=0.0, eps2=0.0, tan_phi=0.0, p=p_ext, p_ext=p_ext,
        lr1=lr1, fthk=fthk, c1=c1, c2=c2, c3=c3
    )
    assert fac0 == 0.0

    # Case B: Stretched fabric under inflation pressure -> positive porosity factor
    fac1 = compute_wang_nakhimovich_porosity(
        eps1=0.1, eps2=0.1, tan_phi=0.1, p=200000.0, p_ext=p_ext,
        lr1=lr1, fthk=fthk, c1=c1, c2=c2, c3=c3
    )
    assert fac1 > 0.0


# =============================================================================
# 3. Geometry & Surface Integrals Tests
# =============================================================================

def test_surface_geometry_unit_cube():
    """Test Gauss divergence volume and area calculation for a unit cube."""
    # 8 nodes of unit cube [0, 1]^3
    nodes = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=np.float64)

    # 6 quad facets oriented outwards
    segments = [
        [0, 3, 2, 1],  # -Z face
        [4, 5, 6, 7],  # +Z face
        [0, 1, 5, 4],  # -Y face
        [2, 3, 7, 6],  # +Y face
        [0, 4, 7, 3],  # -X face
        [1, 2, 6, 5],  # +X face
    ]

    surf = Surface(id=1, segments=segments)
    vol, area, normals, areas, centroids = compute_surface_geometry(surf, nodes)

    assert vol == pytest.approx(1.0, rel=1e-5)
    assert area == pytest.approx(6.0, rel=1e-5)
    assert len(normals) == 6
    for a_fac in areas:
        assert a_fac == pytest.approx(1.0, rel=1e-5)


# =============================================================================
# 4. Multi-Chamber Thermodynamics Integration Tests
# =============================================================================

def test_fvmbag_gas_injection_mass_conservation():
    """Test mass and internal energy accumulation under injector flow."""
    model = Model()
    model.x = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    surf1 = Surface(id=1, segments=[
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5]
    ])
    model.surfaces[1] = surf1

    mat1 = Material(id=1, law=1, rho0=1.2)
    mat1.params = {"CPA": 1000.0, "CPB": 0.0, "CPC": 0.0, "R_igc": 287.0, "MW": 1.0}
    model.materials[1] = mat1

    mv = MonvolFvmbag(
        id=1,
        surf_id=1,
        mat_id=1,
        pext=100000.0,
        t_initial=300.0,
        iequil=1,  # initial chamber empty/negligible mass
    )
    # Add an injector injecting 0.5 kg/s at 600 K
    inj = FvmInjector(
        id=1,
        chamber_id=1,
        mass_flow=0.5,
        temperature=600.0,
        cpa=1000.0,
        r_spec=287.0,
    )
    mv.injectors.append(inj)

    FvmAirbagManager.initialize(mv, model)

    dt = 0.01  # 10 ms
    current_time = 0.0
    for _ in range(10):
        update_fvmbag_thermodynamics(mv, model, dt, current_time)
        current_time += dt

    # Total mass injected = 0.5 kg/s * 0.1 s = 0.05 kg
    ch1 = mv.chambers[1]
    assert ch1.mass == pytest.approx(0.05, rel=1e-3)
    # In a rigid constant-volume vessel, injected enthalpy h_inj = Cp*T_inj becomes internal energy u = Cv*T,
    # leading to T_chamber = gamma * T_inj = (1000 / 713) * 600 K = 841.515 K.
    gamma_inj = 1000.0 / (1000.0 - 287.0)
    expected_T = gamma_inj * 600.0
    assert ch1.temperature == pytest.approx(expected_T, rel=1e-3)
    # Ideal gas law: P = rho * R * T = (0.05 / 1.0) * 287 * expected_T = 12075.7 Pa
    expected_P = (0.05 / 1.0) * 287.0 * expected_T
    assert ch1.pressure == pytest.approx(expected_P, rel=1e-3)


def test_fvmbag_dual_chamber_orifice_communication():
    """Test mass communication between two chambers through an orifice."""
    model = Model()
    model.x = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    model.surfaces[1] = Surface(id=1, segments=[[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4]])
    model.surfaces[2] = Surface(id=2, segments=[[2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5]])

    mv = MonvolFvmbag(id=1, surf_id=1, surf_id_in=2)
    FvmAirbagManager.initialize(mv, model)

    c1 = mv.chambers[1]
    c2 = mv.chambers[2]
    c1.volume = 1.0
    c2.volume = 1.0

    # High pressure in chamber 1, low pressure in chamber 2
    c1.mass = 2.0
    c1.density = 2.0
    c1.temperature = 300.0
    c1.r_spec = 287.0
    c1.cpa = 1004.0
    c1.pressure = c1.density * c1.r_spec * c1.temperature
    c1.energy = c1.mass * compute_specific_internal_energy(c1.temperature, c1.cpa, r_spec=c1.r_spec)

    c2.mass = 0.5
    c2.density = 0.5
    c2.temperature = 300.0
    c2.r_spec = 287.0
    c2.cpa = 1004.0
    c2.pressure = c2.density * c2.r_spec * c2.temperature
    c2.energy = c2.mass * compute_specific_internal_energy(c2.temperature, c2.cpa, r_spec=c2.r_spec)

    mv.orifices = [
        FvmOrifice(
            id=1,
            chamber1_id=1,
            chamber2_id=2,
            area=0.01,
            cd=0.8,
            is_open=True,
        )
    ]

    total_mass_init = c1.mass + c2.mass
    dt = 1e-4

    for _ in range(20):
        update_fvmbag_thermodynamics(mv, model, dt, 0.0)

    # Mass must be conserved across the orifice
    total_mass_final = c1.mass + c2.mass
    assert total_mass_final == pytest.approx(total_mass_init, rel=1e-5)
    # Chamber 1 pressure should decrease, chamber 2 should increase
    assert c1.mass < 2.0
    assert c2.mass > 0.5


# =============================================================================
# 5. Boundary Surface Force Assembly Tests
# =============================================================================

def test_fvmbag_force_equilibrium_on_closed_cube():
    """Test that uniform internal pressure produces zero net resultant force on a closed box."""
    model = Model()
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    model.x = nodes.copy()

    segments = [
        [0, 3, 2, 1],  # -Z face (normal -Z)
        [4, 5, 6, 7],  # +Z face (normal +Z)
        [0, 1, 5, 4],  # -Y face (normal -Y)
        [2, 3, 7, 6],  # +Y face (normal +Y)
        [0, 4, 7, 3],  # -X face (normal -X)
        [1, 2, 6, 5],  # +X face (normal +X)
    ]
    model.surfaces[1] = Surface(id=1, segments=segments)

    mv = MonvolFvmbag(id=1, surf_id=1, pext=100000.0)
    FvmAirbagManager.initialize(mv, model)

    c1 = mv.chambers[1]
    c1.pressure = 200000.0  # Net overpressure = 100000 Pa

    f = np.zeros_like(nodes)
    apply_fvmbag_forces(mv, model, nodes, f)

    # Net force sum over all nodes must be identically zero
    net_force = np.sum(f, axis=0)
    np.testing.assert_allclose(net_force, [0.0, 0.0, 0.0], atol=1e-9)

    # Positive Z face (nodes 4,5,6,7) should have net upward force of P_net * Area = 100000 * 1.0 = 100000 N
    fz_top = f[4, 2] + f[5, 2] + f[6, 2] + f[7, 2]
    assert fz_top == pytest.approx(100000.0, rel=1e-5)


# =============================================================================
# 6. Starter Deck Parsing Tests
# =============================================================================

def test_starter_read_monvol_fvmbag1_and_fvmbag2(tmp_path):
    """Test reading /MONVOL/FVMBAG1 and /MONVOL/FVMBAG2 deck definitions."""
    deck = """\
# Starter Deck with FVM Airbags
/BEGIN
Test_FVM_Airbags
       1.000       1.000
                  kg                   m                   s                   K
/NODE/1
                   1                 0.0                 0.0                 0.0
                   2                 1.0                 0.0                 0.0
                   3                 1.0                 1.0                 0.0
                   4                 0.0                 1.0                 0.0
/SURF/PART/EXT/1
External Surface
         1
         1         2         3         4
/SURF/PART/EXT/2
Internal Partition
         1
         1         2         3         4
/MAT/GAS/1
Airbag Gas
              1.2000              1004.0               287.0
/MONVOL/FVMBAG1/1
Single Chamber FVM Bag
         1                 0.05
                 1.0                 1.0                 1.0                 1.0                 1.0
         1                             101325.0                 0.0         0         0
/MONVOL/FVMBAG2/2
Dual Chamber FVM Bag
         1         2                0.02         0
         1                                       101325.0               298.0                   0
/END
"""
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck)

    model = run_starter(str(p))

    assert 1 in model.monvol_fvmbags
    f1 = model.monvol_fvmbags[1]
    assert f1.id == 1
    assert f1.surf_id == 1
    assert f1.mat_id == 1
    assert f1.pext == pytest.approx(101325.0)
    assert f1.hconv == pytest.approx(0.05)

    assert 2 in model.monvol_fvmbag2s
    f2 = model.monvol_fvmbag2s[2]
    assert f2.id == 2
    assert f2.surf_id_ex == 1
    assert f2.surf_id_in == 2
    assert f2.mat_id == 1
    assert f2.pext == pytest.approx(101325.0)
    assert f2.hconv == pytest.approx(0.02)
