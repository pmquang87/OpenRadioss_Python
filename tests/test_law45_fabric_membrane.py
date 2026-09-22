"""
Tests for LAW45 — Orthotropic fabric/membrane material model with rate-dependent Zhao plasticity (/MAT/LAW45).

Verifies:
- In-plane tension stiffness in 2 orthogonal fiber directions (warp/weft)
- Zero or near-zero compressive stiffness (fiber buckling)
- Optional shear modulus (G_ab / G_12)
- Rate-dependent elasto-plastic Zhao formulation matching OpenRadioss Fortran:
  * engine/source/materials/mat/mat045/sigeps45.F (solids)
  * engine/source/materials/mat/mat045/sigeps45c.F (shells)
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import numpy as np
import pytest

from pyradioss.materials import (
    law45_orth_fabric,
    solid_update as mat_solid_update,
    shell_update as mat_shell_update,
    sound_speed as mat_sound_speed,
    solid_tangent as mat_solid_tangent,
    shell_membrane_tangent as mat_shell_membrane_tangent,
)
from pyradioss.model.entities import Material


def test_law45_params_defaults_and_derived():
    p = law45_orth_fabric.Law45Params(
        e=210000.0,
        nu=0.3,
        ca=250.0,
        cb=500.0,
        cn=0.4,
        epsm=0.2,
        sigm=800.0,
        cc=30.0,
        cd=10.0,
        cm=0.8,
        eps0=0.001,
        ce=0.5,
        ck=0.2,
        cutfre=500.0,
    )
    assert p.young == 210000.0
    assert abs(p.shear - 210000.0 / (2.0 * 1.3)) < 1e-6
    assert abs(p.bulk - 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))) < 1e-6
    assert abs(p.a1 - 210000.0 / (1.0 - 0.09)) < 1e-6
    assert abs(p.a2 - 0.3 * p.a1) < 1e-6
    # Fabric defaults equal isotropic defaults
    assert p.ea == p.e
    assert p.eb == p.e
    assert p.nuba == p.nu
    assert p.gab == p.shear
    assert p.rcomp == 0.0


def test_law45_orthotropic_fabric_stiffness_and_fiber_buckling():
    """Test orthotropic directional moduli (warp/weft) and zero compressive stiffness (fiber buckling)."""
    p = law45_orth_fabric.Law45Params(
        ea=50000.0,      # Warp stiffness
        eb=20000.0,      # Weft stiffness
        nu=0.3,          # nu_12
        gab=5000.0,      # In-plane shear modulus
        rcomp=0.0,       # Zero compressive stiffness (buckling)
        ca=1e6,          # High yield stress for pure elastic investigation
    )
    # Check derived orthotropic coefficients
    nu_ba = 0.3 * (20000.0 / 50000.0)  # 0.12
    denom = 1.0 - 0.3 * nu_ba           # 0.964
    assert abs(p.c11 - 50000.0 / denom) < 1e-3
    assert abs(p.c22 - 20000.0 / denom) < 1e-3
    assert abs(p.c12 - 0.3 * 20000.0 / denom) < 1e-3

    # Shell update in pure warp tension: deps_xx = 1e-3, deps_yy = 0
    sig = np.zeros(3)
    deps_tension = np.array([1e-3, 0.0, 0.0])
    s_ten, _, _ = law45_orth_fabric.shell_update(p, sig, deps_tension, dt=1e-4)
    assert s_ten[0] > 0.0
    assert abs(s_ten[0] - p.c11 * 1e-3) < 1e-3

    # Shell update in pure warp compression (fiber buckling): deps_xx = -1e-3
    deps_compression = np.array([-1e-3, 0.0, 0.0])
    s_comp, _, _ = law45_orth_fabric.shell_update(p, sig, deps_compression, dt=1e-4)
    # Due to fiber buckling (rcomp = 0.0), compressive stress must be zero!
    assert s_comp[0] == 0.0
    assert s_comp[1] == 0.0

    # Test with non-zero rcomp (e.g. small residual compressive resistance: 0.05)
    p_comp = law45_orth_fabric.Law45Params(
        ea=50000.0,
        eb=20000.0,
        nu=0.3,
        gab=5000.0,
        rcomp=0.05,
        ca=1e6,
    )
    s_comp_res, _, _ = law45_orth_fabric.shell_update(p_comp, sig, deps_compression, dt=1e-4)
    expected_sxx = 0.05 * (-p.c11 * 1e-3)
    assert abs(s_comp_res[0] - expected_sxx) < 1e-3


def test_law45_resolve_from_material_and_dict():
    mat = Material(
        id=45,
        law=45,
        rho0=2.7e-3,
        title="FabricZhao",
        params={
            "MAT_E": 70000.0,
            "MAT_NU": 0.33,
            "MAT_A": 150.0,
            "MAT_B": 300.0,
            "MAT_N": 0.5,
            "MAT_EPS": 0.25,
            "MAT_SIG": 600.0,
            "MAT_C": 20.0,
            "MAT_D": 5.0,
            "MAT_M": 1.2,
            "MAT_EPS0": 0.01,
            "MAT_E_VISC": 2.0,
            "MAT_K_VISC": 0.15,
            "CUTFRE": 1000.0,
            # Fabric specific keys
            "EA": 72000.0,
            "EB": 68000.0,
            "NUBA": 0.31,
            "GAB": 27000.0,
            "RCOMP": 0.02,
        },
    )
    p = law45_orth_fabric.resolve(mat)
    assert p.rho0 == 2.7e-3
    assert p.e == 70000.0
    assert p.nu == 0.33
    assert p.ca == 150.0
    assert p.cb == 300.0
    assert p.cn == 0.5
    assert p.epsm == 0.25
    assert p.sigm == 600.0
    assert p.cc == 20.0
    assert p.cd == 5.0
    assert p.cm == 1.2
    assert p.eps0 == 0.01
    assert p.ce == 2.0
    assert p.ck == 0.15
    assert p.cutfre == 1000.0
    assert p.ea == 72000.0
    assert p.eb == 68000.0
    assert p.nuba == 0.31
    assert p.gab == 27000.0
    assert p.rcomp == 0.02


def test_law45_eval_yield_formula():
    p = law45_orth_fabric.Law45Params(
        ca=200.0,
        cb=400.0,
        cn=0.5,
        epsm=0.3,
        sigm=1000.0,
        cc=25.0,
        cd=10.0,
        cm=1.5,
        eps0=0.01,
        ce=1.0,
        ck=0.2,
    )
    # 1. Zero plastic strain and below eps0:
    sigy, h = p.eval_yield(epsp=0.0, epsdot=0.001)
    assert sigy == 200.0 + 1.0 * (0.001 ** 0.2)
    assert h == 0.0

    # 2. Plastic strain > 0 and epsdot > eps0:
    epsp = 0.1
    epsdot = 10.0
    ch1 = 200.0 + 400.0 * (0.1 ** 0.5)
    ch2 = (25.0 - 10.0 * (0.1 ** 1.5)) * math.log(10.0 / 0.01)
    ch3 = 1.0 * (10.0 ** 0.2)
    expected_sigy = min(1000.0 + ch3, ch1 + ch2 + ch3)

    # Hardening modulus H = qh1 + qh2 (sigeps45.F line 289)
    # cn = 0.5 < 1 -> qh1 = cb * cn * epsp^(1 - cn)
    qh1 = 400.0 * 0.5 * (0.1 ** (1.0 - 0.5))
    # cm = 1.5 >= 1 -> qh2 = cd * cm * epsp^(cm - 1) * log(epsdot / eps0)
    qh2 = 10.0 * 1.5 * (0.1 ** (1.5 - 1.0)) * math.log(10.0 / 0.01)
    expected_h = qh1 + qh2

    sigy_calc, h_calc = p.eval_yield(epsp=epsp, epsdot=epsdot)
    assert abs(sigy_calc - expected_sigy) < 1e-6
    assert abs(h_calc - expected_h) < 1e-6

    # 3. Solid degradation when epsp > epsm (sigeps45.F line 273):
    sigy_degraded, _ = p.eval_yield(epsp=0.35, epsdot=1.0, is_shell=False)
    assert sigy_degraded == 0.0
    # In shells, sigy remains positive until fail:
    sigy_shell, _ = p.eval_yield(epsp=0.35, epsdot=1.0, is_shell=True)
    assert sigy_shell > 0.0


def test_law45_solid_update_elastic():
    p = law45_orth_fabric.Law45Params(
        e=200000.0,
        nu=0.25,
        ca=500.0,
        cb=0.0,
        cn=1.0,
    )
    sig = np.zeros(6)
    # Small strain: pure uniaxial tension deps_xx = 1e-4
    deps = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_new, ep_new, c = law45_orth_fabric.solid_update(p, sig, deps, dt=1e-5)

    assert ep_new == 0.0
    assert s_new[0] > 0.0
    assert c > 0.0

    # For pure elastic increment with nu=0.25:
    # K = 200000 / (3 * 0.5) = 133333.33
    # G = 200000 / (2 * 1.25) = 80000
    # lam = K - 2G/3 = 133333.33 - 53333.33 = 80000
    # dsig_xx = (lam + 2G) * 1e-4 = (80000 + 160000) * 1e-4 = 24.0
    # dsig_yy = dsig_zz = lam * 1e-4 = 8.0
    assert abs(s_new[0] - 24.0) < 1e-3
    assert abs(s_new[1] - 8.0) < 1e-3
    assert abs(s_new[2] - 8.0) < 1e-3


def test_law45_solid_update_plastic_radial_return():
    p = law45_orth_fabric.Law45Params(
        e=200000.0,
        nu=0.25,
        ca=100.0,
        cb=200.0,
        cn=1.0,
    )
    sig = np.zeros(6)
    # Large strain to produce plastic flow: deps_xx = 0.01, others zero
    deps = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_new, ep_new, c = law45_orth_fabric.solid_update(p, sig, deps, dt=1e-3)

    assert ep_new > 0.0
    # Compute von Mises stress of the resulting stress state
    p_m = (s_new[0] + s_new[1] + s_new[2]) / 3.0
    dev = np.array([s_new[0] - p_m, s_new[1] - p_m, s_new[2] - p_m, s_new[3], s_new[4], s_new[5]])
    j2 = 0.5 * (dev[0]**2 + dev[1]**2 + dev[2]**2) + dev[3]**2 + dev[4]**2 + dev[5]**2
    vm = math.sqrt(3.0 * j2)

    # Von Mises stress should match the yield stress
    sigy_expected, _ = p.eval_yield(ep_new, epsdot=0.01 / 1e-3)
    assert abs(vm - sigy_expected) / sigy_expected < 0.05


def test_law45_solid_update_with_element_deletion():
    p = law45_orth_fabric.Law45Params(e=100000.0, nu=0.3, ca=200.0)
    sig = np.zeros(6)
    deps = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    extra = {"off": np.array([0.0])}
    s_new, ep_new, _ = law45_orth_fabric.solid_update(p, sig, deps, dt=1e-5, extra=extra)
    # When off is 0, stress must be completely zero
    assert np.allclose(s_new, 0.0)


def test_law45_shell_update_elastic_and_plastic():
    p = law45_orth_fabric.Law45Params(
        e=100000.0,
        nu=0.3,
        ca=150.0,
        cb=0.0,
        cn=1.0,
    )
    sig = np.zeros(3)
    # Small strain increment
    deps_el = np.array([5e-4, 0.0, 0.0])
    s_el, ep_el, c_el = law45_orth_fabric.shell_update(p, sig, deps_el, dt=1e-4)
    assert ep_el == 0.0
    assert s_el[0] > 0.0
    # In plane stress: dsig_xx = A1 * deps_xx = (E / (1 - nu^2)) * 5e-4
    expected_sxx = (100000.0 / (1.0 - 0.09)) * 5e-4
    assert abs(s_el[0] - expected_sxx) < 1e-2

    # Large strain increment for plastic flow
    deps_pl = np.array([0.01, 0.0, 0.0])
    s_pl, ep_pl, _ = law45_orth_fabric.shell_update(p, sig, deps_pl, dt=1e-4)
    assert ep_pl > 0.0
    # In plane stress uniaxial tension, von Mises stress is exactly s_xx
    vm_shell = math.sqrt(s_pl[0]**2 + s_pl[1]**2 - s_pl[0] * s_pl[1] + 3.0 * s_pl[2]**2)
    assert abs(vm_shell - 150.0) < 1e-1


def test_law45_tangents():
    p = law45_orth_fabric.Law45Params(e=200000.0, nu=0.3)
    c_sol = law45_orth_fabric.solid_tangent(p)
    assert c_sol.shape == (6, 6)
    assert np.allclose(c_sol, c_sol.T)

    c_sh = law45_orth_fabric.shell_tangent(p)
    assert c_sh.shape == (3, 3)
    assert np.allclose(c_sh, c_sh.T)

    # Shell membrane tangent alias
    c_mem = law45_orth_fabric.shell_membrane_tangent(p)
    assert np.array_equal(c_mem, c_sh)

    # Consistent tangents
    assert np.array_equal(law45_orth_fabric.consistent_solid_tangent(p), c_sol)
    assert np.array_equal(law45_orth_fabric.consistent_shell_tangent(p), c_sh)

    # General template tangent
    assert np.array_equal(law45_orth_fabric.tangent(p), c_sol)


def test_law45_element_group_interface():
    p = law45_orth_fabric.Law45Params(e=150000.0, nu=0.25)
    group = SimpleNamespace(mat=p)

    # Test element-group solid_update fallback
    s, ep, c = law45_orth_fabric.solid_update(group, x=None, u=None, ur=None, dt=1e-5, fint=None, mint=None)
    assert s.shape == (6,)
    assert ep.shape == (1,)
    assert c > 0.0

    # Test tangent(group)
    c_tan = law45_orth_fabric.tangent(group)
    assert c_tan.shape == (6, 6)
    assert c_tan[0, 0] > 0.0


def test_law45_dispatcher_registration():
    mat = Material(id=1, law=45, rho0=1.5e-3, title="Mat45", params={"E": 50000.0, "NU": 0.28, "MAT_A": 200.0})
    # Dispatch through top-level pyradioss.materials functions
    c_speed = mat_sound_speed(mat)
    assert c_speed > 0.0

    c_solid = mat_solid_tangent(mat)
    assert c_solid.shape == (6, 6)

    c_shell = mat_shell_membrane_tangent(mat)
    assert c_shell.shape == (3, 3)

    sig = np.zeros(6)
    deps = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_out, ep_out, _ = mat_solid_update(mat, sig, deps, dt=1e-5)
    assert s_out[0] > 0.0


def test_law45_to_uparam():
    """Verify to_uparam matches the 19-parameter array in sigeps45.F / sigeps45c.F."""
    p = law45_orth_fabric.Law45Params(
        e=200000.0,
        nu=0.3,
        g=76923.0,
        ca=250.0,
        cb=500.0,
        cn=0.4,
        epsm=0.2,
        sigm=800.0,
        cc=30.0,
        cd=10.0,
        cm=0.8,
        eps0=0.001,
        ce=0.5,
        ck=0.2,
        cutfre=500.0,
    )
    up = p.to_uparam()
    assert up.shape == (19,)
    assert up[0] == 200000.0   # UPARAM(1): YOUNG
    assert up[1] == 0.3        # UPARAM(2): ANU
    assert up[2] == 76923.0    # UPARAM(3): G
    assert up[3] == 250.0      # UPARAM(4): CA
    assert up[4] == 500.0      # UPARAM(5): CB
    assert up[5] == 0.4        # UPARAM(6): CN
    assert up[6] == 0.2        # UPARAM(7): EPSM
    assert up[7] == 800.0      # UPARAM(8): SIGM
    assert up[8] == 30.0       # UPARAM(9): CC
    assert up[9] == 10.0       # UPARAM(10): CD
    assert up[10] == 0.8       # UPARAM(11): CM
    assert up[11] == 0.001     # UPARAM(12): EPS0
    assert up[12] == 0.5       # UPARAM(13): CE
    assert up[13] == 0.2       # UPARAM(14): CK
    assert abs(up[14] - p.k) < 1e-6       # UPARAM(15): C1
    assert abs(up[15] - p.c14g3) < 1e-6   # UPARAM(16): C14G3
    assert abs(up[16] - p.a1) < 1e-6      # UPARAM(17): A1
    assert abs(up[17] - p.a2) < 1e-6      # UPARAM(18): A2
    assert up[18] == 500.0     # UPARAM(19): CUTFRE


def test_law45_fabric_membrane_alias():
    """Verify pyradioss.materials.law45_fabric_membrane re-exports match law45_orth_fabric."""
    from pyradioss.materials import law45_fabric_membrane
    assert law45_fabric_membrane.Law45Params is law45_orth_fabric.Law45Params
    assert law45_fabric_membrane.solid_update is law45_orth_fabric.solid_update
    assert law45_fabric_membrane.shell_update is law45_orth_fabric.shell_update
    assert law45_fabric_membrane.tangent is law45_orth_fabric.tangent


def test_law45_element_group_full_template_signature():
    """Verify solid_update(group, x, u, ur, dt, fint, mint) per template interface."""
    p = law45_orth_fabric.Law45Params(e=120000.0, nu=0.3)
    group = SimpleNamespace(mat=p)
    x = np.zeros((8, 3))
    u = np.zeros((8, 3))
    ur = np.zeros((8, 3))
    dt = 1e-5
    fint = np.zeros((8, 3))
    mint = np.zeros((8, 3))
    res = law45_orth_fabric.solid_update(group, x, u, ur, dt, fint, mint)
    assert len(res) == 3
    s, ep, c = res
    assert s.shape == (6,)
    assert c > 0.0


def test_law45_shell_thickness_evolution():
    """Verify thickness strain dezz and thickness change per sigeps45c.F lines 310-313."""
    p = law45_orth_fabric.Law45Params(e=100000.0, nu=0.3, ca=1e6)  # Elastic
    sig = np.zeros(3)
    deps = np.array([1e-3, 1e-3, 0.0])
    extra = {"thk": np.array([1.5]), "dezz": np.zeros(1)}
    s, ep, _ = law45_orth_fabric.shell_update(p, sig, deps, dt=1e-4, extra=extra)

    # Elastic: dezz = -(deps_xx + deps_yy) * (nu / (1 - nu))
    expected_dezz = -(1e-3 + 1e-3) * (0.3 / 0.7)
    assert abs(extra["dezz"][0] - expected_dezz) < 1e-8
    expected_thk = 1.5 + expected_dezz * 1.5
    assert abs(extra["thk"][0] - expected_thk) < 1e-8


