"""Unit tests for QBAT shell pinching and IGA 3D solid elements (Work Stream 11).

Tests both:
1. Isogeometric Analysis 3D (pyradioss.elements.iga3d)
   - 1D Cox-de Boor recursive basis and derivatives (dersonebasisfun.F, dersbasisfuns.F)
   - 3D NURBS rational basis partition of unity and gradient sum-to-zero (ig3dfint.F)
   - Patch test: uniform strain / constant stress equilibrium
   - Jaumann rate stress rotation invariance (srota3.F)
   - Lumped mass conservation (ig3dmass3.F)
   - Full forces() step and critical time step
2. QBAT Shell Pinching (pyradioss.elements.shell_qbat)
   - Coordinate transformation and scaling (cbacoorpinch.F)
   - Pinching deformation and strain rates (cbadefpinch.F, cbastra3pinch.F)
   - Pinching internal force assembly (cbaforipinch.F)
   - Normal projection to global 3D coordinates (cbapinchproj.F)
   - Dynamic thickness update (cbapinchthk.F)
   - 3D solid P-wave critical time step (cndt3pinch.F, sigeps01gpinch.F)
   - Full forces() cycle with energy accounting and fpinch accumulation
"""

import numpy as np
import pytest

from pyradioss.elements import iga3d
from pyradioss.elements import shell_qbat


# ============================================================================
# IGA 3D NURBS Elements Tests
# ============================================================================

class TestIGA3DBasis:
    """Test Cox-de Boor B-spline and NURBS basis routines."""

    def test_ders_one_basis_fun_linear(self):
        """Test 1D basis function and derivative for p=1 (linear hat function).

        # Ported from engine/source/elements/ige3d/dersonebasisfun.F
        """
        knot = np.array([0.0, 0.0, 1.0, 1.0])
        p = 1

        # Basis function N_{0, 1} on [0, 1] is N(u) = 1 - u, dN/du = -1.0
        n0, dn0 = iga3d.ders_one_basis_fun(0, p, 0.25, knot)
        np.testing.assert_allclose(n0, 0.75, atol=1e-14)
        np.testing.assert_allclose(dn0, -1.0, atol=1e-14)

        n1, dn1 = iga3d.ders_one_basis_fun(0, p, 0.75, knot)
        np.testing.assert_allclose(n1, 0.25, atol=1e-14)
        np.testing.assert_allclose(dn1, -1.0, atol=1e-14)

    def test_ders_basis_funs_quadratic(self):
        """Test Algorithm A2.2 (ders_basis_funs) for p=2 (quadratic).

        # Ported from engine/source/elements/ige3d/dersbasisfuns.F
        """
        knot = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        p = 2
        span = 2  # Knot span [0.0, 0.5)

        for u in [0.1, 0.25, 0.4]:
            ders1, ders2 = iga3d.ders_basis_funs(span, p, u, knot)
            # Partition of unity: sum of basis functions = 1
            np.testing.assert_allclose(np.sum(ders1), 1.0, atol=1e-14)
            # Sum of derivatives = 0
            np.testing.assert_allclose(np.sum(ders2), 0.0, atol=1e-14)

    def test_nurbs_3d_partition_of_unity(self):
        """Test that 3D NURBS basis functions satisfy partition of unity

        sum R_a = 1 and sum grad R_a = 0 everywhere.
        # Ported from engine/source/elements/ige3d/ig3dfint.F lines 242-327
        """
        # Trilinear hex unit cube: 8 control points
        cpts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        weights = np.array([1.0, 1.2, 0.8, 1.1, 1.0, 0.9, 1.3, 1.0])  # Non-trivial rational weights

        span = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        for u in [0.2, 0.5, 0.8]:
            for v in [0.3, 0.6]:
                for w in [0.1, 0.9]:
                    xi_parent = np.array([2.0 * u - 1.0, 2.0 * v - 1.0, 2.0 * w - 1.0])
                    R, dR_dx, Jmat, detJ = iga3d.nurbs_3d_basis_and_derivs(
                        xi_parent, (1, 1, 1), span, weights, cpts
                    )
                    assert len(R) == 8
                    np.testing.assert_allclose(np.sum(R), 1.0, atol=1e-13)
                    np.testing.assert_allclose(np.sum(dR_dx, axis=0), [0.0, 0.0, 0.0], atol=1e-13)
                    assert detJ > 0.0


class TestIGA3DMechanics:
    """Test kinematics, stress rotation, mass and force integration for IGA 3D."""

    def test_jaumann_rate_stress_rotation(self):
        """Test Jaumann rate stress rotation maintains invariance under rigid rotation.

        # Ported from common_source/elements/solids/srota3.F
        """
        # Initial isotropic + deviatoric stress [xx, yy, zz, xy, yz, zx]
        sig = np.array([100.0, 50.0, 30.0, 10.0, 5.0, 0.0])
        # Pure spin around z-axis: omega_z = 0.1 rad/s
        omega = np.array([0.0, 0.0, 0.1])
        dt = 0.05
        spin = omega * dt

        sig_rot = iga3d.srota3_stress(sig, spin)
        # Trace should be preserved under pure rotation: tr(sigma) = sig_xx + sig_yy + sig_zz
        tr_old = sig[0] + sig[1] + sig[2]
        tr_new = sig_rot[0] + sig_rot[1] + sig_rot[2]
        np.testing.assert_allclose(tr_new, tr_old, atol=1e-12)

    def test_lumped_mass_conservation(self):
        """Test lumped mass conservation: sum M_a = rho * V_0.

        # Ported from starter/source/elements/ige3d/ig3dmass3.F
        """
        class DummyGroup:
            def __init__(self):
                self.n = 1
                self.conn = np.array([np.arange(8)])
                self.state = {}

        class DummyModel:
            def __init__(self):
                self.x0 = np.array([
                    [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0], [2.0, 3.0, 0.0],
                    [0.0, 0.0, 4.0], [2.0, 0.0, 4.0],
                    [0.0, 3.0, 4.0], [2.0, 3.0, 4.0],
                ])

        class DummyMat:
            rho0 = 7800.0
            E = 2.1e11
            nu = 0.3
            law = 1

        class DummyProp:
            params = {
                "degree_u": 1, "degree_v": 1, "degree_w": 1,
                "n_cpts_u": 2, "n_cpts_v": 2, "n_cpts_w": 2,
                "knot_u": [0.0, 0.0, 1.0, 1.0],
                "knot_v": [0.0, 0.0, 1.0, 1.0],
                "knot_w": [0.0, 0.0, 1.0, 1.0],
            }

        group = DummyGroup()
        group.state["slices"] = [(slice(0, 1), DummyMat(), DummyProp())]
        model = DummyModel()

        node_idx, mass_c, _ = iga3d.init_group(group, model, None)

        # Volume = 2 * 3 * 4 = 24 m^3
        expected_total_mass = 7800.0 * 24.0
        np.testing.assert_allclose(np.sum(mass_c), expected_total_mass, rtol=1e-10)

    def test_iga3d_patch_test_equilibrium(self):
        """Constant strain patch test: uniform tension produces uniform internal stress

        and node forces that sum to zero (exact equilibrium).
        # Ported from engine/source/elements/ige3d/ig3dfint.F
        """
        # Unit cube with 8 control points
        x = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        ], dtype=float)

        conn = np.array([np.arange(8)])

        class DummyMat:
            E = 2.0e11
            nu = 0.3
            rho0 = 7850.0
            law = 1

        class DummyProp:
            params = {
                "degree_u": 1, "degree_v": 1, "degree_w": 1,
                "n_cpts_u": 2, "n_cpts_v": 2, "n_cpts_w": 2,
            }

        class DummyModel:
            x0 = x.copy()

        class DummyGroup:
            def __init__(self):
                self.n = 1
                self.conn = conn
                self.state = {
                    "slices": [(slice(0, 1), DummyMat(), DummyProp())],
                }

        group = DummyGroup()
        iga3d.init_group(group, DummyModel(), None)

        # Affine velocity field: uniform elongation along x
        # v_x = 0.01 * x, v_y = -0.003 * y, v_z = -0.003 * z
        v = np.zeros_like(x)
        v[:, 0] = 0.01 * x[:, 0]
        v[:, 1] = -0.003 * x[:, 1]
        v[:, 2] = -0.003 * x[:, 2]

        dt = 0.001
        fint = np.zeros_like(x)

        dt_e = iga3d.forces(group, x, v, None, dt, fint, None)
        assert dt_e[0] > 0.0

        # Total internal force sum across all nodes must be exactly zero (Newton's 3rd law / equilibrium)
        np.testing.assert_allclose(np.sum(fint, axis=0), [0.0, 0.0, 0.0], atol=1e-4)

        # Internal energy must be strictly positive (work done by stretching)
        assert group.state["eint"][0] > 0.0


# ============================================================================
# QBAT Shell Pinching Tests
# ============================================================================

class TestQBATShellPinching:
    """Test through-thickness stretch DOFs for QBAT shell element."""

    def test_cbacoorpinch_kinematics(self):
        """Test global to local pinching velocity transformation and scaling.

        # Ported from engine/source/elements/shell/coqueba/cbacoorpinch.F lines 93-200
        """
        n = 1
        E = np.eye(3)[None, :, :]  # Local frame aligned with global
        # Flat element nodal frames: normal along z, t1 along x, t2 along y
        vqn = np.zeros((n, 4, 9))
        for j in range(4):
            vqn[:, j, 0:3] = [1.0, 0.0, 0.0]  # t1
            vqn[:, j, 3:6] = [0.0, 1.0, 0.0]  # t2
            vqn[:, j, 6:9] = [0.0, 0.0, 1.0]  # normal

        thick = np.array([0.02])
        dt = 1e-5
        lc = np.array([0.1])

        # Test pure normal pinching velocity: vz = 0.5 m/s at all 4 nodes
        vpe = np.zeros((n, 4, 3))
        vpe[:, :, 2] = 0.5

        vp_xyz, vp_t1, vp_t2, facp, ezzavg, avgthk = shell_qbat._cbacoorpinch(
            E, vqn, vpe, thick, dt, lc
        )

        # avgthk should be thick * (1 + 2 * 0.5 * dt)
        expected_thk = 0.02 * (1.0 + 2.0 * 0.5 * dt)
        np.testing.assert_allclose(avgthk[0], expected_thk, rtol=1e-6)

        # Tangents should be zero for pure normal pinching
        np.testing.assert_allclose(vp_t1, 0.0, atol=1e-12)
        np.testing.assert_allclose(vp_t2, 0.0, atol=1e-12)

        # Normal rates scaled by 2 / avgthk
        expected_rate = 0.5 * (2.0 / expected_thk)
        np.testing.assert_allclose(vp_xyz, expected_rate, rtol=1e-6)

    def test_cbadefpinch_and_stra3(self):
        """Test pinching transverse shear and through-thickness normal strain rate.

        # Ported from engine/source/elements/shell/coqueba/cbadefpinch.F and cbastra3pinch.F
        """
        n = 1
        # Unit square Jacobian
        tc = np.zeros((n, 2, 2))
        tc[:, 0, 0] = 1.0
        tc[:, 1, 1] = 1.0

        vqg_11 = np.ones(n)
        vqg_12 = np.ones(n)

        # Pure normal stretch rate: vp_xyz = 100.0 at all nodes
        vp_xyz = np.full((n, 4), 100.0)
        vp_t1 = np.zeros((n, 4))
        vp_t2 = np.zeros((n, 4))

        dt = 0.001
        for ng in range(4):
            vdefp, bcp, bp, dbetadxy = shell_qbat._cbadefpinch(
                tc, vqg_11, vqg_12, vp_xyz, vp_t1, vp_t2, ng
            )
            # Bilinear shape functions sum to 1 at any Gauss point, so sum(bp * 100.0) = 100.0
            np.testing.assert_allclose(vdefp[:, 2], 100.0, atol=1e-12)

            ep_xz, ep_yz, ezz = shell_qbat._cbastra3pinch(vdefp, dt)
            np.testing.assert_allclose(ezz, 100.0 * dt, atol=1e-12)

    def test_cbaforipinch_and_proj(self):
        """Test pinching generalized internal force assembly and global 3D projection.

        # Ported from engine/source/elements/shell/coqueba/cbaforipinch.F and cbapinchproj.F
        """
        n = 1
        cdet = np.array([1.0])
        thick = np.array([0.01])
        bcp = np.zeros((n, 4, 2))
        bp = np.array([0.25, 0.25, 0.25, 0.25])
        sig_zz = np.array([1.0e6])  # 1 MPa tension
        mom_p = np.zeros((n, 2))

        # Internal force: c2 * bp * sig_zz = thick * cdet * 0.25 * 1e6 = 0.01 * 1.0 * 0.25 * 1e6 = 2500 N
        vfpinch = shell_qbat._cbaforipinch(cdet, thick, bcp, bp, sig_zz, mom_p)
        np.testing.assert_allclose(vfpinch, 2500.0, rtol=1e-10)

        # Global projection along normal (z)
        E = np.eye(3)[None, :, :]
        vqn = np.zeros((n, 4, 9))
        for j in range(4):
            vqn[:, j, 6:9] = [0.0, 0.0, 1.0]  # normal along z

        fp = shell_qbat._cbapinchproj(E, vqn, vfpinch, thick)
        # Force along z: fp[:, :, 2] = vfpinch / thick = 2500 / 0.01 = 250,000 N
        np.testing.assert_allclose(fp[:, :, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(fp[:, :, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(fp[:, :, 2], 250000.0, rtol=1e-10)

    def test_cbapinchthk_thickness_update(self):
        """Test thickness dynamic update based on ezz at the 4 Gauss points.

        # Ported from engine/source/elements/shell/coqueba/cbapinchthk.F
        """
        thick = np.array([0.01, 0.02])
        # 10% stretch at all Gauss points
        ezzpg = np.full((2, 4), 0.10)
        thk_new = shell_qbat._cbapinchthk(thick, ezzpg)
        np.testing.assert_allclose(thk_new, thick * 1.10, rtol=1e-12)

    def test_cndt3pinch_sound_speed(self):
        """Test that pinched shell uses 3D solid P-wave speed for time step.

        # Ported from engine/source/elements/shell/coqueba/cndt3pinch.F & sigeps01gpinch.F
        """
        lc = np.array([0.05])
        amu = np.array([0.0])
        rho = np.array([7800.0])
        E_mod = 2.1e11
        nu = 0.3

        dt_pinch, ssp_pinch = shell_qbat._cndt3pinch(lc, amu, rho, E_mod, nu)

        # 3D solid P-wave modulus: PA1 = E*(1-nu)/((1+nu)(1-2nu))
        pa1 = E_mod * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        expected_ssp = np.sqrt(pa1 / rho)
        np.testing.assert_allclose(ssp_pinch, expected_ssp, rtol=1e-10)

        # Pinched solid P-wave speed > plane-stress shell sound speed sqrt(E/(rho*(1-nu^2)))
        ssp_shell = np.sqrt(E_mod / (rho * (1.0 - nu ** 2)))
        assert ssp_pinch[0] > ssp_shell[0]
        # Consequently, critical time step for pinched shell is strictly smaller (more conservative)
        assert dt_pinch[0] < lc[0] / ssp_shell[0]

    def test_full_qbat_forces_with_pinching(self):
        """Test full forces() execution with QBAT pinching enabled:

        checks through-thickness force accumulation (fpinch),
        dynamic thickness evolution, and internal energy accrual.
        """
        # 1-element flat unit quad [0, 1] x [0, 1]
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
        conn = np.array([[0, 1, 2, 3]])

        class DummyMat:
            E = 2.1e11
            nu = 0.3
            G = 2.1e11 / (2.0 * (1.0 + 0.3))
            rho0 = 7800.0
            law = 1
            fail = None
            params = {}

        class DummyProp:
            thick = 0.01
            ipinch = 1  # Enable pinching!
            nip = 1
            dn = 0.0

        class DummyModel:
            x0 = x.copy()

        class DummyGroup:
            def __init__(self):
                self.n = 1
                self.conn = conn
                self.state = {
                    "slices": [(slice(0, 1), DummyMat(), DummyProp())],
                }

        group = DummyGroup()
        shell_qbat.init_group(group, DummyModel(), None)

        assert group.state["ipinch"] is True
        # Verify initial sound speed was initialized to 3D solid speed
        pa1 = 2.1e11 * (1.0 - 0.3) / ((1.0 + 0.3) * (1.0 - 2.0 * 0.3))
        expected_ssp = np.sqrt(pa1 / 7800.0)
        np.testing.assert_allclose(group.state["ssp0"], expected_ssp, rtol=1e-6)

        # Apply velocity: zero in-plane velocities, positive through-thickness pinching rate
        v = np.zeros_like(x)
        vr = np.zeros_like(x)
        vpinch = np.zeros_like(x)
        vpinch[:, 2] = 10.0  # 10 m/s normal stretch velocity

        dt = 1e-6
        fint = np.zeros_like(x)
        mint = np.zeros_like(x)
        fpinch = np.zeros_like(x)

        dt_e = shell_qbat.forces(group, x, v, vr, dt, fint, mint, vpinch=vpinch, fpinch=fpinch)

        # 1. Critical time step is positive and governed by solid sound speed
        assert dt_e[0] > 0.0

        # 2. Pinching forces fpinch accumulated along normal (z)
        assert np.any(np.abs(fpinch[:, 2]) > 0.0)

        # 3. Internal energy eint increased due to through-thickness work
        assert group.state["eint"][0] > 0.0

        # 4. Thickness updated dynamically (stretch rate > 0 => thickness grew)
        assert group.state["thick"][0] > 0.01
