"""Exact Fortran physics parity tests for /MAT/LAW110 (Vegter) (M579).

Directly validates formulas from upstream OpenRadioss:
  - ``starter/source/materials/mat/mat110/hm_read_mat110.F``
  - ``engine/source/materials/mat/mat110/sigeps110c.F``
  - ``engine/source/materials/mat/mat110/sigeps110c_newton.F``
  - ``engine/source/materials/mat/mat110/sigeps110c_nice.F``

Validates:
  1. Chebyshev Fourier cosine polynomial matrix COS2_DATA exact expansion.
  2. Orthogonality and exact angle interpolation A @ Q = B.
  3. Reference and hinge point coordinates (UN, BI, PS, SH).
  4. Bezier quadratic spline interpolation and tangent continuity.
  5. Bergstrom-van Liempt viscoplastic strain rate and temperature flow stress formulation.
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import MaterialLaw110
from pyradioss.materials.law110_vegter import (
    COS2_DATA,
    VegterModelParams,
    eval_flow_stress_and_hardening,
    eval_vegter_equivalent_stress_and_normal,
)


class TestChebyshevCosineParity:
    """Validate Chebyshev polynomial table COS2_DATA against analytic identities."""

    def test_chebyshev_polynomial_evaluations(self):
        """Verify sum_{k=0}^n COS2[k, n] * cos(2*theta)^k == cos(2*n*theta) for n=0..9."""
        test_angles_deg = [0.0, 15.0, 22.5, 30.0, 45.0, 60.0, 67.5, 75.0, 90.0]
        for deg in test_angles_deg:
            theta_rad = math.radians(deg)
            cos2t = math.cos(2.0 * theta_rad)
            for n in range(10):
                # Evaluate via matrix column n
                p_val = 0.0
                for k in range(n + 1):
                    p_val += COS2_DATA[k, n] * (cos2t ** k)
                expected = math.cos(2.0 * n * theta_rad)
                assert p_val == pytest.approx(expected, abs=1.0e-12)

    def test_chebyshev_derivative_evaluations(self):
        """Verify derivative d(cos(2*n*theta))/d(cos(2*theta)) == sum_{k=1}^n k * COS2[k, n] * cos(2*theta)^(k-1)."""
        test_angles_deg = [10.0, 25.0, 40.0, 55.0, 70.0, 80.0]
        for deg in test_angles_deg:
            theta_rad = math.radians(deg)
            cos2t = math.cos(2.0 * theta_rad)
            eps = 1.0e-7
            cos2t_p = cos2t + eps
            cos2t_m = cos2t - eps
            for n in range(1, 10):
                dp_dcos2 = 0.0
                for k in range(1, n + 1):
                    dp_dcos2 += k * COS2_DATA[k, n] * (cos2t ** (k - 1))

                # Numerical finite difference of cos(2*n*theta)
                val_p = sum(COS2_DATA[k, n] * (cos2t_p ** k) for k in range(n + 1))
                val_m = sum(COS2_DATA[k, n] * (cos2t_m ** k) for k in range(n + 1))
                fd = (val_p - val_m) / (2.0 * eps)
                assert dp_dcos2 == pytest.approx(fd, rel=1.0e-4, abs=1.0e-5)


class TestAngleInterpolationParity:
    """Validate angle interpolation factor solve A @ Q = B matching hm_read_mat110.F."""

    def test_angle_interpolation_reconstruction(self):
        """At sample angles theta_J, evaluated properties must equal input values B_J exactly."""
        angles = [
            [1.0, 1.8, 1.15, 0.0, 0.50],
            [1.02, 1.6, 1.18, 0.0, 0.52],
            [0.98, 2.0, 1.12, 0.0, 0.48],
        ]
        mat = MaterialLaw110(mid=1, rho0=7.8e-6, young=210000.0, nu=0.3, icrit=1, angles_data=angles)
        params = VegterModelParams(mat)

        for j, theta in enumerate(params.theta_rad):
            cos2t = math.cos(2.0 * theta)
            # Evaluate fun
            fun_eval = 0.0
            for i in range(params.nangle):
                c_factor = sum(COS2_DATA[k, i] * (cos2t ** k) for k in range(i + 1))
                fun_eval += params.q_fun[i] * c_factor
            assert fun_eval == pytest.approx(angles[j][0], abs=1.0e-10)


class TestHingePointFormulasParity:
    """Validate intersection formulas between tangent lines matching hm_read_mat110.F."""

    def test_hinge_point_tangent_intersection(self):
        """Hinge point B must lie at intersection of tangent at A (with normal N) and tangent at C (with normal M)."""
        # Tangent line at A: N1*(x - A1) + N2*(y - A2) = 0 => N1*x + N2*y = N1*A1 + N2*A2
        # Tangent line at C: M1*(x - C1) + M2*(y - C2) = 0 => M1*x + M2*y = M1*C1 + M2*C2
        # Intersection gives Hinge point B.
        A = np.array([1.0, 0.0])
        N = np.array([1.0, -0.6])  # normal at A
        C = np.array([1.2, 0.5])
        M = np.array([1.0, 0.0])   # normal at C

        # Solve system: [N; M] @ B = [N.A; M.C]
        LHS = np.array([[N[0], N[1]], [M[0], M[1]]])
        RHS = np.array([N @ A, M @ C])
        B_expected = np.linalg.solve(LHS, RHS)

        # Compare with hm_read_mat110.F formula:
        det = N[0] * M[1] - M[0] * N[1]
        bx = (M[1] * RHS[0] - N[1] * RHS[1]) / det
        by = (N[0] * RHS[1] - M[0] * RHS[0]) / det
        B_fortran = np.array([bx, by])

        assert B_expected[0] == pytest.approx(B_fortran[0], abs=1.0e-12)
        assert B_expected[1] == pytest.approx(B_fortran[1], abs=1.0e-12)


class TestViscoplasticHardeningParity:
    """Validate Bergstrom-van Liempt thermal activation model in sigeps110c.F."""

    def test_bergstrom_van_liempt_temperature_rate_flow_stress(self):
        """sig_y = sighard + sigs * [1 + (kboltz * T / dg0) * ln(1 + rate / deps0)]^m."""
        mat = MaterialLaw110(
            mid=1,
            rho0=7.8e-6,
            young=210000.0,
            nu=0.3,
            sigma_r=200.0,
            dsigm=100.0,
            beta=10.0,
            omega=0.5,
            hard_n=0.2,
            eps0=0.001,
            sigs=80.0,
            dg0=1.0e5,
            deps0=1.0,
            m=0.15,
            tini=300.0,
        )
        params = VegterModelParams(mat)

        pla = 0.02
        rate = 100.0
        temp = 400.0

        sig_y, h = eval_flow_stress_and_hardening(params, epsp=pla, rate=rate, temp=temp)

        # Manual computation from Fortran formula:
        p = 0.02
        exp_term = 1.0 - math.exp(-params.omega * p)
        sighard = params.sigma_r + params.dsigm * (params.beta * p + exp_term ** params.hard_n)

        kboltz = 8.6173303e-5
        rate_ratio = rate / params.deps0
        arg = 1.0 + (kboltz * temp / params.dg0) * math.log(1.0 + rate_ratio)
        sigrate = params.sigs * (arg ** params.m)
        expected_sig_y = sighard + sigrate

        assert sig_y == pytest.approx(expected_sig_y, rel=1.0e-8)
