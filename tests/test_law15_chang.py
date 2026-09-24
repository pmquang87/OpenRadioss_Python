"""Unit tests for OpenRadioss LAW15 (Chang-Chang Orthotropic Composite Failure).

References:
  - engine/source/materials/mat/mat015/sigeps15c.F
  - engine/source/materials/mat/mat015/m15crak.F
  - starter/source/materials/mat/mat015/hm_read_mat15.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import law15_chang


def test_law15_elastic_response():
    """Verify linear orthotropic elastic response before yield or failure."""
    E1 = 150000.0
    E2 = 10000.0
    nu12 = 0.3
    G12 = 4500.0
    rho0 = 1.6e-9

    mat = law15_chang.build_law15(
        E1=E1,
        E2=E2,
        nu12=nu12,
        G12=G12,
        G23=3000.0,
        G31=4500.0,
        rho0=rho0,
        sigyt1=1e10,  # High yield to remain purely elastic
        sigyc1=1e10,
        sigyt2=1e10,
        sigyc2=1e10,
        sigt12=1e10,
        s1=1e10,      # High failure limits
        s2=1e10,
        c1=1e10,
        c2=1e10,
        s12=1e10,
        itype=0,
    )

    nu21 = nu12 * E2 / E1
    detc = 1.0 - nu12 * nu21
    C11 = E1 / detc
    C22 = E2 / detc
    C12 = nu21 * C11

    # 1. Uniaxial longitudinal strain
    deps1 = np.array([1.0e-3, 0.0, 0.0])
    sig1, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps1)
    assert sig1[0] == pytest.approx(C11 * 1.0e-3, rel=1e-5)
    assert sig1[1] == pytest.approx(C12 * 1.0e-3, rel=1e-5)
    assert sig1[2] == pytest.approx(0.0, abs=1e-10)

    # 2. Uniaxial transverse strain
    deps2 = np.array([0.0, 1.0e-3, 0.0])
    sig2, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps2)
    assert sig2[0] == pytest.approx(C12 * 1.0e-3, rel=1e-5)
    assert sig2[1] == pytest.approx(C22 * 1.0e-3, rel=1e-5)
    assert sig2[2] == pytest.approx(0.0, abs=1e-10)

    # 3. Pure in-plane shear strain
    deps3 = np.array([0.0, 0.0, 2.0e-3])
    sig3, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps3)
    assert sig3[0] == pytest.approx(0.0, abs=1e-10)
    assert sig3[1] == pytest.approx(0.0, abs=1e-10)
    assert sig3[2] == pytest.approx(G12 * 2.0e-3, rel=1e-5)


def test_law15_fiber_tensile_failure_trigger():
    """Verify Hashin-type tensile fiber failure: (s11/s1)^2 + beta*(s12/s12)^2 >= 1.0."""
    Xt = 1200.0  # Longitudinal tensile strength (s1)
    S12 = 80.0   # Shear strength
    E1 = 150000.0
    E2 = 10000.0
    nu12 = 0.3
    G12 = 4500.0

    mat = law15_chang.build_law15(
        E1=E1,
        E2=E2,
        nu12=nu12,
        G12=G12,
        rho0=1.6e-9,
        sigyt1=1e10, sigyc1=1e10, sigyt2=1e10, sigyc2=1e10, sigt12=1e10,
        s1=Xt,
        s2=1e10,
        c1=1e10,
        c2=1e10,
        s12=S12,
        beta_s=1.0,
        tmax=0.005,  # Non-zero relaxation time
        itype=0,     # No immediate deletion on itype=0
    )

    nu21 = nu12 * E2 / E1
    detc = 1.0 - nu12 * nu21
    C11 = E1 / detc

    # Case A: Below tensile fiber limit
    extra_a = {}
    strain_safe = (Xt * 0.8) / C11
    deps_safe = np.array([strain_safe, 0.0, 0.0])
    sig_a, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps_safe, extra=extra_a)
    damt_a = extra_a["damt15"]
    assert damt_a[0, 0] == 1.0  # Fiber intact
    assert damt_a[0, 1] == 1.0  # Matrix intact
    assert sig_a[0] == pytest.approx(Xt * 0.8, rel=1e-4)

    # Case B: Exceed tensile fiber limit
    extra_b = {}
    strain_fail = (Xt * 1.2) / C11
    deps_fail = np.array([strain_fail, 0.0, 0.0])
    sig_b, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps_fail, extra=extra_b)
    damt_b = extra_b["damt15"]
    # Fiber breakage triggered!
    assert damt_b[0, 0] < 1.0


def test_law15_fiber_compressive_failure_trigger():
    """Verify compressive fiber failure: (s11/c1)^2 >= 1.0 when s11 < 0."""
    Xc = 900.0  # Longitudinal compressive strength (c1)
    E1 = 150000.0
    E2 = 10000.0
    nu12 = 0.3

    mat = law15_chang.build_law15(
        E1=E1,
        E2=E2,
        nu12=nu12,
        G12=4500.0,
        rho0=1.6e-9,
        sigyt1=1e10, sigyc1=1e10, sigyt2=1e10, sigyc2=1e10, sigt12=1e10,
        s1=1e10,
        s2=1e10,
        c1=Xc,
        c2=1e10,
        s12=1e10,
        tmax=0.005,
        itype=0,
    )

    nu21 = nu12 * E2 / E1
    detc = 1.0 - nu12 * nu21
    C11 = E1 / detc

    # Compressive strain exceeding Xc
    extra = {}
    strain_comp = -(Xc * 1.15) / C11
    deps = np.array([strain_comp, 0.0, 0.0])
    sig, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    damt = extra["damt15"]
    # Compressive fiber failure triggered
    assert damt[0, 0] < 1.0


def test_law15_matrix_cracking_trigger():
    """Verify matrix cracking failure criteria in tension and compression."""
    Yt = 60.0    # Transverse strength (c2 / s2)
    S12 = 70.0
    E1 = 150000.0
    E2 = 10000.0
    nu12 = 0.3

    mat = law15_chang.build_law15(
        E1=E1,
        E2=E2,
        nu12=nu12,
        G12=4500.0,
        rho0=1.6e-9,
        sigyt1=1e10, sigyc1=1e10, sigyt2=1e10, sigyc2=1e10, sigt12=1e10,
        s1=1e10,  # High fiber strength so fiber never fails
        c1=1e10,
        c2=Yt,
        s12=S12,
        tmax=0.005,
        itype=0,
    )

    nu21 = nu12 * E2 / E1
    detc = 1.0 - nu12 * nu21
    C22 = E2 / detc

    # Transverse tensile strain triggering matrix cracking
    extra = {}
    deps = np.array([0.0, (Yt * 1.2) / C22, 0.0])
    sig, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps, extra=extra)

    damt = extra["damt15"]
    # Fiber remains intact
    assert damt[0, 0] == 1.0
    # Matrix cracking triggered
    assert damt[0, 1] < 1.0


def test_law15_element_deletion_on_fiber_failure():
    """Verify element deletion (off=0.0) when itype=2 upon fiber failure."""
    Xt = 1000.0
    E1 = 150000.0
    E2 = 10000.0
    nu12 = 0.3

    mat = law15_chang.build_law15(
        E1=E1,
        E2=E2,
        nu12=nu12,
        G12=4500.0,
        rho0=1.6e-9,
        sigyt1=1e10, sigyc1=1e10, sigyt2=1e10, sigyc2=1e10, sigt12=1e10,
        s1=Xt,
        c1=1e10,
        s2=1e10,
        c2=1e10,
        s12=1e10,
        itype=2,  # Deletion on fiber failure
    )

    nu21 = nu12 * E2 / E1
    detc = 1.0 - nu12 * nu21
    C11 = E1 / detc

    extra = {}
    # Strain exceeding Xt
    deps_fail = np.array([(Xt * 1.3) / C11, 0.0, 0.0])
    sig, _, _ = law15_chang.shell_update(mat, np.zeros(3), deps_fail, extra=extra)

    # Element status must be deleted (off=0.0)
    assert extra["off15"][0] == 0.0
    # Stresses must be completely zeroed
    np.testing.assert_allclose(sig, 0.0, atol=1e-12)

    # In subsequent step, element remains deleted and stress remains zero
    sig_next, _, _ = law15_chang.shell_update(mat, sig, deps_fail, extra=extra)
    assert extra["off15"][0] == 0.0
    np.testing.assert_allclose(sig_next, 0.0, atol=1e-12)


def test_law15_tangents_and_sound_speed():
    """Verify shell membrane tangent and acoustic sound speed."""
    E1 = 160000.0
    E2 = 12000.0
    nu12 = 0.28
    G12 = 5000.0
    rho0 = 1.5e-9

    mat = law15_chang.build_law15(
        E1=E1,
        E2=E2,
        nu12=nu12,
        G12=G12,
        G23=3500.0,
        G31=5000.0,
        rho0=rho0,
    )

    # Tangent stiffness
    t_mat = law15_chang.shell_membrane_tangent(mat)
    assert t_mat.shape == (3, 3)
    np.testing.assert_allclose(t_mat, t_mat.T, rtol=1e-12)
    assert np.all(np.diag(t_mat) > 0.0)

    # Sound speed
    c = law15_chang.sound_speed(mat)
    assert c > 0.0
    assert isinstance(c, float)
