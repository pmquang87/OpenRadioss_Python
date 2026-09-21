"""Unit tests for LAW87 (Barlat 2000 / Yld2000-2d anisotropic yield criterion).

Upstream OpenRadioss Fortran reference:
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat087\\sigeps87.F
  (engine/source/materials/mat/mat087/sigeps87c.F90)
  engine/source/materials/mat/mat087/mat87c_swift_voce.F90
  starter/source/materials/mat/mat087/hm_read_mat87.F90
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.materials.law87_barlat2000 import (
    Law87Params,
    barlat2000_equivalent_stress,
    barlat2000_gradient,
    barlat2000_yield_and_gradient,
    barlat2000_yield_surface,
    build_law87,
    consistent_shell_tangent,
    shell_tangent,
    shell_update,
    solid_update,
    tangent,
)
from pyradioss.model.entities import Material


def test_barlat2000_isotropic_mises_limit():
    """Verify that when alpha_1..8 = 1.0 and m = 2, Barlat Yld2000-2d reduces exactly to von Mises."""
    p = Law87Params(
        e=200000.0,
        nu=0.3,
        expa=2.0,
        al1=1.0, al2=1.0, al3=1.0, al4=1.0,
        al5=1.0, al6=1.0, al7=1.0, al8=1.0,
    )

    # Test several distinct stress states
    test_stresses = [
        np.array([250.0, 0.0, 0.0]),
        np.array([0.0, 300.0, 0.0]),
        np.array([0.0, 0.0, 150.0]),
        np.array([200.0, 100.0, 50.0]),
        np.array([-150.0, 80.0, -40.0]),
    ]

    for sig in test_stresses:
        sxx, syy, sxy = sig[0], sig[1], sig[2]
        # Plane-stress von Mises: sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)
        expected_vm = math.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)

        # 1. Equivalent stress
        seq = barlat2000_equivalent_stress(sig, p)
        assert pytest.approx(seq, rel=1e-6) == expected_vm

        # 2. Yield surface: phi = 2 * sigma_vm^2
        phi = barlat2000_yield_surface(sig, p)
        assert pytest.approx(phi, rel=1e-6) == 2.0 * (expected_vm ** 2)

        # 3. Yield condition relative to sigma_y
        sigma_y = expected_vm
        f_val = barlat2000_yield_surface(sig, p, sigma_y=sigma_y)
        assert pytest.approx(f_val, abs=1e-5) == 0.0


def test_barlat2000_anisotropic_yield_surface():
    """Verify 8 anisotropy coefficients (alpha_1..8) with BCC/FCC exponents (m=6, m=8)."""
    # Typical aluminium sheet anisotropy parameters (Barlat et al. 2003, m=8 for FCC)
    p_fcc = Law87Params(
        expa=8.0,
        al1=0.95,
        al2=1.05,
        al3=0.98,
        al4=1.02,
        al5=1.01,
        al6=0.99,
        al7=1.10,
        al8=1.08,
    )

    # Check linear projection coefficients L' and L''
    assert pytest.approx(p_fcc.lp11) == 2.0 * 0.95 / 3.0
    assert pytest.approx(p_fcc.lp12) == -0.95 / 3.0
    assert pytest.approx(p_fcc.lp21) == -1.05 / 3.0
    assert pytest.approx(p_fcc.lp22) == 2.0 * 1.05 / 3.0
    assert pytest.approx(p_fcc.lp66) == 1.10
    assert pytest.approx(p_fcc.lpp66) == 1.08

    sig = np.array([320.0, 150.0, 45.0])
    phi = barlat2000_yield_surface(sig, p_fcc)
    seq = barlat2000_equivalent_stress(sig, p_fcc)

    # phi must equal 2 * seq^m
    assert pytest.approx(phi, rel=1e-6) == 2.0 * (seq ** 8.0)


def test_barlat2000_analytical_gradient_vs_finite_difference():
    """Verify analytical gradient d(sigma_eq)/d(sigma) against central finite differences."""
    p = Law87Params(
        expa=6.0,  # BCC exponent
        al1=0.92,
        al2=1.08,
        al3=1.04,
        al4=0.96,
        al5=1.02,
        al6=0.98,
        al7=1.15,
        al8=1.05,
    )

    stresses = [
        np.array([280.0, 140.0, 35.0]),
        np.array([100.0, 220.0, -50.0]),
        np.array([-180.0, -90.0, 20.0]),
        np.array([0.0, 150.0, 80.0]),
    ]

    h = 1.0e-6
    for sig in stresses:
        grad_analytical = barlat2000_gradient(sig, p)

        # Numerical perturbation
        grad_num = np.zeros(3)
        for k in range(3):
            e_k = np.zeros(3)
            e_k[k] = h
            seq_p = barlat2000_equivalent_stress(sig + e_k, p)
            seq_m = barlat2000_equivalent_stress(sig - e_k, p)
            grad_num[k] = (seq_p - seq_m) / (2.0 * h)

        assert np.allclose(grad_analytical, grad_num, rtol=1e-4, atol=1e-5)

        # Euler's homogeneity theorem: sig : d_seq/d_sig = seq
        seq_val = barlat2000_equivalent_stress(sig, p)
        euler_val = np.dot(sig, grad_analytical)
        assert pytest.approx(euler_val, rel=1e-4) == seq_val


def test_barlat2000_yield_and_gradient_vectorized():
    """Verify batch/vectorized evaluation of yield surface and gradient."""
    p = Law87Params(expa=8.0)
    sig_batch = np.array([
        [200.0, 100.0, 0.0],
        [150.0, 150.0, 30.0],
        [0.0, 180.0, 50.0],
    ])

    phi_batch, grad_batch = barlat2000_yield_and_gradient(sig_batch, p)
    assert phi_batch.shape == (3,)
    assert grad_batch.shape == (3, 3)

    for i in range(3):
        phi_single, grad_single = barlat2000_yield_and_gradient(sig_batch[i], p)
        assert pytest.approx(phi_batch[i], rel=1e-6) == phi_single
        assert np.allclose(grad_batch[i], grad_single, rtol=1e-6)


def test_barlat2000_group_interface():
    """Verify element group calling convention compliance: solid_update and tangent."""
    p = Law87Params(e=210000.0, nu=0.3, expa=6.0)
    group = SimpleNamespace(
        nel=1,
        elements=[1],
        mat=p,
    )
    fint_in = np.zeros(12)

    # solid_update on shell element group returns fint placeholder
    res = solid_update(group, x=None, u=None, ur=None, dt=1.0e-5, fint=fint_in)
    assert res is fint_in

    # solid_update with direct material raises NotImplementedError (shell only)
    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_update(p, np.zeros(6), np.zeros(6))

    # tangent dispatch returns shell membrane tangent
    tang = tangent(group)
    assert tang is not None
    assert tang.shape in ((3, 3), (1, 3, 3))
