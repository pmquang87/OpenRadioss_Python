"""Unit tests for OpenRadioss LAW14 (Composite Solid and Multilayer Shell).

References:
  - engine/source/materials/mat/mat014/sigeps14c.F (shell / multilayer composite formulation)
  - engine/source/materials/mat/mat014/m14law.F (solid 3D formulation)
  - engine/source/materials/mat/mat014/m14ama.F, m14gtf.F, m14ftg.F
  - starter/source/materials/mat/mat014/hm_read_mat14.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import law14_compso


@pytest.fixture
def standard_law14_mat():
    """Fixture returning standard orthotropic composite properties."""
    return law14_compso.build_law14(
        E11=140000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.3,
        nu23=0.35,
        nu31=0.02,
        G12=5000.0,
        G23=3000.0,
        G31=4500.0,
        rho0=1.6e-9,
        sigt1=1500.0,
        sigt2=50.0,
        delta=0.05,
        sigyt1=1200.0,
        sigyc1=1000.0,
        sigyt2=40.0,
        sigyc2=150.0,
        sigyt12=70.0,
        sigyc12=70.0,
        cb=200.0,
        cn=0.5,
        fmax=2000.0,
        wplaref=1.0,
    )


def test_law14_shell_elastic_response(standard_law14_mat):
    """Verify in-plane plane-stress elastic response before damage or yield."""
    mat = standard_law14_mat
    p = mat.params

    e11 = p["E11"]
    e22 = p["E22"]
    nu12 = p["nu12"]
    g12 = p["G12"]

    nu21 = nu12 * e22 / e11
    detc = 1.0 - nu12 * nu21
    Q11 = e11 / detc
    Q22 = e22 / detc
    Q12 = nu12 * e22 / detc

    # 1. Uniaxial longitudinal strain de11
    deps1 = np.array([1.0e-3, 0.0, 0.0])
    sig1, _, _ = law14_compso.shell_update(mat, np.zeros(3), deps1)
    assert sig1[0] == pytest.approx(Q11 * 1.0e-3, rel=1e-5)
    assert sig1[1] == pytest.approx(Q12 * 1.0e-3, rel=1e-5)
    assert sig1[2] == pytest.approx(0.0, abs=1e-10)

    # 2. Uniaxial transverse strain de22
    deps2 = np.array([0.0, 1.0e-3, 0.0])
    sig2, _, _ = law14_compso.shell_update(mat, np.zeros(3), deps2)
    assert sig2[0] == pytest.approx(Q12 * 1.0e-3, rel=1e-5)
    assert sig2[1] == pytest.approx(Q22 * 1.0e-3, rel=1e-5)
    assert sig2[2] == pytest.approx(0.0, abs=1e-10)

    # 3. Pure in-plane shear dgamma12
    deps12 = np.array([0.0, 0.0, 1.0e-3])
    sig12, _, _ = law14_compso.shell_update(mat, np.zeros(3), deps12)
    assert sig12[0] == pytest.approx(0.0, abs=1e-10)
    assert sig12[1] == pytest.approx(0.0, abs=1e-10)
    assert sig12[2] == pytest.approx(g12 * 1.0e-3, rel=1e-5)


def test_law14_shell_layer_orientation(standard_law14_mat):
    """Verify layer orientation transformation for 0, 45, and 90 degrees."""
    mat = standard_law14_mat
    p = mat.params

    e11 = p["E11"]
    e22 = p["E22"]
    nu12 = p["nu12"]
    g12 = p["G12"]

    nu21 = nu12 * e22 / e11
    detc = 1.0 - nu12 * nu21
    Q11 = e11 / detc
    Q22 = e22 / detc
    Q12 = nu12 * e22 / detc

    deps = np.array([1.0e-3, 0.0, 0.0])

    # 0 degrees: fiber aligned with shell x
    sig_0, _, _ = law14_compso.shell_update(mat, np.zeros(3), deps, extra={"angle": 0.0})
    assert sig_0[0] == pytest.approx(Q11 * 1.0e-3, rel=1e-5)
    assert sig_0[1] == pytest.approx(Q12 * 1.0e-3, rel=1e-5)

    # 90 degrees: fiber aligned with shell y, matrix along shell x
    sig_90, _, _ = law14_compso.shell_update(mat, np.zeros(3), deps, extra={"angle": 90.0})
    assert sig_90[0] == pytest.approx(Q22 * 1.0e-3, rel=1e-5)
    assert sig_90[1] == pytest.approx(Q12 * 1.0e-3, rel=1e-5)

    # 45 degrees: analytical transformed Q_xxxx = 0.25*(Q11 + Q22 + 2*Q12 + 4*G12)
    Q_45_xxxx = 0.25 * (Q11 + Q22 + 2.0 * Q12 + 4.0 * g12)
    sig_45, _, _ = law14_compso.shell_update(mat, np.zeros(3), deps, extra={"angle": 45.0})
    assert sig_45[0] == pytest.approx(Q_45_xxxx * 1.0e-3, rel=1e-5)


def test_law14_multilayer_shell_cross_ply(standard_law14_mat):
    """Verify multi-layer composite laminate shell integration for [0/90] laminate."""
    mat = standard_law14_mat
    p = mat.params

    nu21 = p["nu12"] * p["E22"] / p["E11"]
    detc = 1.0 - p["nu12"] * nu21
    Q11 = p["E11"] / detc
    Q22 = p["E22"] / detc

    # Two-layer cross-ply: layer 1 at 0 deg (t=0.5), layer 2 at 90 deg (t=0.5)
    extra_multi = {
        "layers": [
            {"thick": 0.5, "angle": 0.0},
            {"thick": 0.5, "angle": 90.0},
        ]
    }

    deps = np.array([1.0e-3, 0.0, 0.0])
    sig_avg, _, _ = law14_compso.shell_update(mat, np.zeros(3), deps, extra=extra_multi)

    # Expected average stress: 0.5 * Q11 + 0.5 * Q22
    expected_s_avg = 0.5 * (Q11 + Q22) * 1.0e-3
    assert sig_avg[0] == pytest.approx(expected_s_avg, rel=1e-5)

    # Resultant forces N = s_avg * total_thickness (1.0)
    assert extra_multi["N"][0] == pytest.approx(expected_s_avg * 1.0, rel=1e-5)

    # Resultant moment M: layer 1 at z = -0.25, layer 2 at z = +0.25
    expected_M = (-0.25 * Q11 * 0.5 + 0.25 * Q22 * 0.5) * 1.0e-3
    assert extra_multi["M"][0] == pytest.approx(expected_M, rel=1e-5)


def test_law14_shell_directional_damage():
    """Verify directional tensile cracking in fiber direction 1 and matrix direction 2."""
    mat = law14_compso.build_law14(
        E11=100000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.25,
        nu23=0.3,
        nu31=0.02,
        G12=4000.0,
        G23=3000.0,
        G31=3000.0,
        rho0=1.6e-9,
        sigt1=100.0,  # Low fiber tensile limit
        sigt2=30.0,   # Low matrix tensile limit
        delta=0.05,
        sigyt1=500.0, # High yield to isolate damage
        sigyc1=500.0,
        sigyt2=500.0,
        sigyc2=500.0,
        sigyt12=500.0,
        sigyc12=500.0,
    )

    # 1. Tension exceeding sigt1
    extra1 = {}
    deps1 = np.array([2.0e-3, 0.0, 0.0])  # trial stress ~ 205 MPa > 100 MPa
    sig1, _, _ = law14_compso.shell_update(mat, np.zeros(3), deps1, extra=extra1)
    assert extra1["dam14"][0, 0] == pytest.approx(0.05, abs=1e-6)
    assert sig1[0] <= 100.0

    # 2. Tension exceeding sigt2
    extra2 = {}
    deps2 = np.array([0.0, 5.0e-3, 0.0])  # trial stress ~ 51 MPa > 30 MPa
    sig2, _, _ = law14_compso.shell_update(mat, np.zeros(3), deps2, extra=extra2)
    assert extra2["dam14"][0, 1] == pytest.approx(0.05, abs=1e-6)
    assert sig2[1] <= 30.0


def test_law14_shell_crack_closure():
    """Verify unilateral crack closure when reversing from tension to compression."""
    mat = law14_compso.build_law14(
        E11=100000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.25,
        nu23=0.3,
        nu31=0.02,
        G12=4000.0,
        G23=3000.0,
        G31=3000.0,
        rho0=1.6e-9,
        sigt1=80.0,
        sigt2=30.0,
        delta=0.05,
        sigyt1=500.0,
        sigyc1=500.0,
        sigyt2=500.0,
        sigyc2=500.0,
        sigyt12=500.0,
        sigyc12=500.0,
    )

    extra = {}
    # Step 1: Open crack in direction 1
    deps_open = np.array([1.5e-3, 0.0, 0.0])
    sig_open, _, _ = law14_compso.shell_update(mat, np.zeros(3), deps_open, extra=extra)
    assert extra["dam14"][0, 0] > 0.0
    assert extra["epc14"][0, 0] > 0.0

    # Step 2: Reverse into compression -> crack is open, tensile stress relieved to 0
    deps_rev = np.array([-1.0e-3, 0.0, 0.0])
    sig_rev, _, _ = law14_compso.shell_update(mat, sig_open, deps_rev, extra=extra)
    assert sig_rev[0] == pytest.approx(0.0, abs=1e-10)


def test_law14_shell_tsai_wu_plasticity():
    """Verify Tsai-Wu anisotropic yield surface activation and plastic work accumulation."""
    mat = law14_compso.build_law14(
        E11=100000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.25,
        nu23=0.3,
        nu31=0.02,
        G12=4000.0,
        G23=3000.0,
        G31=3000.0,
        rho0=1.6e-9,
        sigt1=1000.0,  # High tensile damage threshold to isolate plasticity
        sigt2=1000.0,
        sigyt1=80.0,   # Yield limits
        sigyc1=100.0,
        sigyt2=30.0,
        sigyc2=80.0,
        sigyt12=40.0,
        sigyc12=40.0,
        cb=150.0,
        cn=0.5,
        fmax=500.0,
        wplaref=1.0,
    )

    extra = {}
    deps = np.array([1.5e-3, 0.0, 0.0])  # trial stress ~ 150 MPa > 80 MPa
    sig_plas, ep_plas, _ = law14_compso.shell_update(mat, np.zeros(3), deps, extra=extra)

    # Plastic work accumulated and stress relaxed relative to trial
    assert extra["wpla14"][0] > 0.0
    assert sig_plas[0] < 150.0


def test_law14_shell_tangents(standard_law14_mat):
    """Verify consistent_shell_tangent and shell_membrane_tangent."""
    mat = standard_law14_mat

    # Membrane tangent
    C_mem = law14_compso.shell_membrane_tangent(mat)
    assert C_mem.shape == (3, 3)
    assert np.allclose(C_mem, C_mem.T)
    assert C_mem[0, 0] > C_mem[1, 1] > C_mem[2, 2] > 0.0

    # Consistent numerical tangent
    C_cons = law14_compso.consistent_shell_tangent(mat, np.zeros(3))
    assert C_cons.shape == (3, 3)
    assert np.allclose(C_cons, C_mem, rtol=1e-4)


def test_law14_tangent_dispatcher(standard_law14_mat):
    """Verify tangent(group) dispatcher for shells and solids."""
    mat = standard_law14_mat

    class ShellGroup:
        def __init__(self, m):
            self.mat = m
            self.elem_type = "shell"

    class SolidGroup:
        def __init__(self, m):
            self.mat = m
            self.elem_type = "solid"

    sh_grp = ShellGroup(mat)
    T_sh = law14_compso.tangent(sh_grp)
    assert T_sh is not None
    assert T_sh.shape == (3, 3)

    so_grp = SolidGroup(mat)
    T_so = law14_compso.tangent(so_grp)
    assert T_so is not None
    assert T_so.shape == (6, 6)


def test_law14_5_component_shell(standard_law14_mat):
    """Verify 5-component stress/strain state with transverse shear."""
    mat = standard_law14_mat

    deps5 = np.array([1.0e-3, 0.5e-3, 0.2e-3, 0.1e-3, 0.15e-3])
    sig5, _, _ = law14_compso.shell_update(mat, np.zeros(5), deps5)

    assert len(sig5) == 5
    assert sig5[3] == pytest.approx(mat.params["G23"] * 0.1e-3, rel=1e-5)
    assert sig5[4] == pytest.approx(mat.params["G31"] * 0.15e-3, rel=1e-5)


def test_law14_solid_update_regression(standard_law14_mat):
    """Verify that existing 3D solid element update functionality is fully preserved."""
    mat = standard_law14_mat

    sig6 = np.zeros(6)
    deps6 = np.array([1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_out, ep_out, ssp = law14_compso.solid_update(mat, sig6, deps6)

    assert sig_out.shape == (6,)
    assert sig_out[0] == pytest.approx(mat.params["D11"] * 1.0e-3, rel=1e-4)
    assert ssp > 0.0
