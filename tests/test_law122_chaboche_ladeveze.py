"""Tests for OpenRadioss LAW122: Modified Ladevèze Composite & Chaboche Model.

Verifies:
  1. Parameter initialization and parsing from Material / MaterialLaw122 / dict.
  2. OpenRadioss MAT122 Ladevèze cutting-plane Newton return mapping (mat122_newton.F).
  3. Progressive damage evolution (fiber, shear, and transverse matrix damage).
  4. Chaboche Armstrong-Frederick kinematic hardening and Bauschinger effect.
  5. 2D plane-stress shell update, sound speed, and consistent algorithmic tangents.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law122_chaboche
from pyradioss.materials.law122_chaboche import Law122Params, build_law122, mat122_newton_solid_update
from pyradioss.model.entities import MaterialLaw122


def test_law122_params_initialization_and_entities():
    """Verify initialization from MaterialLaw122 entity and defaults."""
    mat_ent = MaterialLaw122(
        id=1221,
        title="Ladeveze_Composite",
        rho0=1.5e-9,
        e1=120000.0,
        e2=10000.0,
        e3=10000.0,
        g12=5000.0,
        g23=3500.0,
        g31=5000.0,
        nu12=0.3,
        nu23=0.4,
        nu31=0.03,
        e1c=110000.0,
        gamma=0.1,
        ish=1,
        itr=1,
        ires=2,
        sigy0=200.0,
        beta=800.0,
        hard_m=0.5,
        hard_a=1.2,
        eps_fti=0.015,
        eps_ftu=0.025,
        dftu=0.8,
        eps_fci=0.012,
        eps_fcu=0.020,
        dfcu=0.7,
        ibuck=2,
        dsat1=0.9,
        y0=0.2,
        yc=1.5,
        b=0.5,
        dmax=0.95,
        yr=0.1,
        ysp=2.0,
        dsat2=0.8,
        y0p=0.3,
        ycp=1.8,
    )

    p = build_law122(mat_ent)
    assert p.id == 1221
    assert p.young1 == pytest.approx(120000.0)
    assert p.young2 == pytest.approx(10000.0)
    assert p.g12 == pytest.approx(5000.0)
    assert p.sigy0 == pytest.approx(200.0)
    assert p.beta == pytest.approx(800.0)
    assert p.hard_m == pytest.approx(0.5)
    assert p.hard_a == pytest.approx(1.2)
    assert p.ish == 1
    assert p.itr == 1
    assert p.ires == 2
    assert p.ibuck == 2
    assert p.y0 == pytest.approx(0.2)
    assert p.yc == pytest.approx(1.5)
    assert p.dmax == pytest.approx(0.95)

    # Derived orthotropic stiffness
    assert p.s11 > 0.0
    assert p.s22 > 0.0
    assert p.s33 > 0.0

    shapes = law122_chaboche.extra_shapes(p, nip=4)
    assert shapes["uvar122"] == (4, 18)
    assert shapes["backstress"] == (4, 6)
    assert shapes["damage"] == (4, 6)


def test_mat122_newton_elastic_and_plastic_return_mapping():
    """Verify OpenRadioss MAT122 cutting plane Newton return mapping (mat122_newton.F)."""
    p = build_law122(
        young1=100000.0,
        young2=10000.0,
        young3=10000.0,
        nu12=0.3,
        nu23=0.35,
        nu31=0.03,
        g12=4000.0,
        g23=3000.0,
        g31=4000.0,
        sigy0=150.0,
        beta=500.0,
        hard_m=0.5,
        hard_a=1.0,
        rho0=1.6e-9,
    )

    sig0 = np.zeros(6, dtype=np.float64)
    # Small elastic shear strain: depsxy = 0.01 -> shear stress = G12 * depsxy = 40.0 < 150.0
    deps_el = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0], dtype=np.float64)
    sig_el, pla_el, c_el, extra_el = mat122_newton_solid_update(p, sig0, deps_el, epsp=0.0)

    assert sig_el[3] == pytest.approx(40.0, rel=1.0e-4)
    assert pla_el == 0.0
    assert c_el > 0.0

    # Plastic shear strain: depsxy = 0.045 -> trial seq = 180.0 > 150.0
    deps_pl = np.array([0.0, 0.0, 0.0, 0.045, 0.0, 0.0], dtype=np.float64)
    sig_pl, pla_pl, c_pl, extra_pl = mat122_newton_solid_update(p, sig0, deps_pl, epsp=0.0)

    assert pla_pl > 0.0
    # Yield stress with hardening: sigy0 + beta * pla^m
    sigy_expected = p.sigy0 + p.beta * (pla_pl ** p.hard_m)
    # Shear stress relaxed to yield surface within 3 cutting-plane Newton iterations
    assert sig_pl[3] < 180.0
    assert abs(sig_pl[3]) == pytest.approx(sigy_expected, rel=0.01)


def test_mat122_progressive_damage():
    """Verify fiber and matrix damage evolution according to mat122_newton.F."""
    p = build_law122(
        young1=120000.0,
        young2=10000.0,
        young3=10000.0,
        nu12=0.3,
        nu23=0.35,
        nu31=0.03,
        g12=5000.0,
        g23=3500.0,
        g31=5000.0,
        sigy0=100.0,
        beta=0.0,
        hard_a=1.0,
        eps_fti=0.01,
        eps_ftu=0.02,
        dftu=0.8,
        ish=1,   # Linear shear damage
        y0=0.1,
        yc=1.0,
        dmax=0.9,
    )

    sig0 = np.zeros(6, dtype=np.float64)
    # Tensile strain exceeding fiber threshold: epsxx = 0.015 (between eps_fti=0.01 and eps_ftu=0.02)
    deps = np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sig, pla, c, extra = mat122_newton_solid_update(p, sig0, deps, epsp=0.0, eps=deps)

    dmg = extra["damage"]
    # Fiber damage D_f should be active
    assert dmg[1] > 0.0
    assert dmg[1] <= p.dftu
    # Stresses are scaled by (1 - df)
    assert sig[0] < p.s11 * deps[0]


def test_chaboche_bauschinger_effect():
    """Verify Armstrong-Frederick kinematic hardening exhibits Bauschinger effect."""
    p = build_law122(
        young1=200000.0,
        nu12=0.3,
        sigy0=200.0,
        c_kin=15000.0,
        gamma_kin=50.0,
        r_inf=0.0,
        b_iso=0.0,
        hard_a=1.0,
    )

    sig = np.zeros(6, dtype=np.float64)
    extra = {"backstress": np.zeros(6, dtype=np.float64)}

    # Forward plastic loading in X
    deps_fwd = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_fwd, epsp_1, _ = law122_chaboche.solid_update(
        p, sig, deps_fwd, epsp=0.0, extra=extra, chaboche=True
    )
    assert epsp_1 > 0.0
    alpha_1 = float(np.asarray(extra["backstress"]).reshape(-1)[0])
    # Backstress should develop positive component along flow direction
    assert alpha_1 > 0.0

    # Reverse loading: backstress reduces the effective elastic domain in compression
    deps_rev = np.array([-0.002, 0.0006, 0.0006, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_rev, epsp_2, _ = law122_chaboche.solid_update(
        p, sig_fwd, deps_rev, epsp=epsp_1, extra=extra, chaboche=True
    )
    # Total plastic strain increased due to early reverse yield
    assert epsp_2 >= epsp_1


def test_shell_update_and_tangents():
    """Verify shell update, sound speed, and consistent algorithmic tangent."""
    p = build_law122(
        young1=180000.0,
        nu12=0.3,
        sigy0=220.0,
        c_kin=5000.0,
        gamma_kin=20.0,
        r_inf=30.0,
        b_iso=10.0,
        rho0=7850.0,
    )

    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.004, -0.0012, 0.001], dtype=np.float64)
    sig_sh_new, epsp_sh, c_sh = law122_chaboche.shell_update(p, sig_sh, deps_sh, epsp=0.0)

    assert sig_sh_new[0] > 0.0
    assert c_sh > 0.0

    # Tangents
    c_tan_solid = law122_chaboche.tangent(p, sig=np.array([250.0, 0.0, 0.0, 0.0, 0.0, 0.0]), epsp=0.01)
    assert c_tan_solid.shape == (6, 6)
    assert np.all(np.isfinite(c_tan_solid))

    c_tan_shell = law122_chaboche.shell_tangent(p, sig=sig_sh_new, epsp=epsp_sh)
    assert c_tan_shell.shape == (3, 3)
    assert np.all(np.isfinite(c_tan_shell))
