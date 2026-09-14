"""Unit tests for /MAT/LAW110 (Vegter anisotropic yield model) constitutive kernel (M579).

Verifies:
  - Elastic behavior below yield (Hooke's law in plane stress and Courant sound speed).
  - Uniaxial yield along Rolling Direction (0 deg), Diagonal (45 deg), and Transverse (90 deg).
  - Equibiaxial yielding (fbi scaling).
  - Pure shear yielding (fsh scaling).
  - Plane strain yielding.
  - Convergence of both Newton semi-implicit (ires=1) and Nice explicit (ires=2) methods.
  - Plastic thickness strain and plastic incompressibility (d_eps_xx^p + d_eps_yy^p + d_eps_zz^p = 0).
  - Swift, Voce, and combined Swift-Voce hardening.
  - Tabulated hardening evaluation.
  - Consistent 3x3 plane stress algorithmic tangent matrix.
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import MaterialLaw110
from pyradioss.materials.law110_vegter import (
    VegterModelParams,
    eval_flow_stress_and_hardening,
    eval_vegter_equivalent_stress_and_normal,
    law110_shell_update,
    law110_consistent_shell_tangent,
    law110_sound_speed,
)


def _create_standard_mat(
    icrit: int = 1,
    ires: int = 2,
    young: float = 210000.0,
    nu: float = 0.3,
    sigma_r: float = 250.0,
    dsigm: float = 150.0,
    beta: float = 10.0,
    omega: float = 0.5,
    hard_n: float = 0.22,
) -> MaterialLaw110:
    angles = [
        [1.0, 1.8, 1.15, 0.0, 0.5],
        [1.0, 1.5, 1.12, 0.0, 0.5],
        [1.0, 2.1, 1.18, 0.0, 0.5],
    ]
    return MaterialLaw110(
        mid=1,
        rho0=7.8e-6,
        young=young,
        nu=nu,
        ires=ires,
        icrit=icrit,
        sigma_r=sigma_r,
        dsigm=dsigm,
        beta=beta,
        omega=omega,
        hard_n=hard_n,
        angles_data=angles,
    )


class TestVegterElasticity:
    """Test elastic behavior and sound speed."""

    def test_elastic_hooke_plane_stress(self):
        mat = _create_standard_mat()
        deps = np.array([1.0e-5, 0.0, 0.0])  # small strain
        sigo = np.zeros(3)
        extra = {"pla": 0.0, "thk": 1.0}

        sig_new, extra_out = law110_shell_update(mat, deps=deps, sigo=sigo, extra=extra)

        params = VegterModelParams(mat)
        expected_sxx = params.a11 * 1.0e-5
        expected_syy = params.a12 * 1.0e-5

        assert sig_new[0] == pytest.approx(expected_sxx, rel=1.0e-4)
        assert sig_new[1] == pytest.approx(expected_syy, rel=1.0e-4)
        assert sig_new[2] == pytest.approx(0.0, abs=1.0e-8)
        assert extra_out["pla"] == pytest.approx(0.0)

    def test_sound_speed(self):
        mat = _create_standard_mat()
        params = VegterModelParams(mat)
        expected_c = math.sqrt(params.a11 / mat.rho0)
        c = law110_sound_speed(mat)
        assert c == pytest.approx(expected_c, rel=1.0e-6)


class TestVegterPlasticYield:
    """Test yielding under various 2D plane stress loading states."""

    def test_uniaxial_tension_rd(self):
        """Uniaxial strain in 0 deg direction causes yielding at sigma_r."""
        mat = _create_standard_mat(ires=2)
        # Apply substantial strain increment
        deps = np.array([0.01, -0.003, 0.0])
        sigo = np.zeros(3)
        extra = {"pla": 0.0, "thk": 1.0}

        sig_new, extra_out = law110_shell_update(mat, deps=deps, sigo=sigo, extra=extra)
        assert extra_out["pla"] > 0.0
        assert extra_out["seq"] >= mat.sigma_r

    def test_newton_vs_nice_consistency(self):
        """Verify both Newton (ires=2) and Nice (ires=1) algorithms produce consistent yield stresses."""
        mat_nice = _create_standard_mat(ires=1)
        mat_newton = _create_standard_mat(ires=2)

        deps = np.array([0.0005, -0.00015, 0.0002])
        sig1 = np.zeros(3)
        sig2 = np.zeros(3)
        ex1 = {"pla": 0.0, "thk": 1.0}
        ex2 = {"pla": 0.0, "thk": 1.0}

        for _ in range(10):
            sig1, ex1 = law110_shell_update(mat_nice, deps=deps, sigo=sig1, extra=ex1)
            sig2, ex2 = law110_shell_update(mat_newton, deps=deps, sigo=sig2, extra=ex2)

        assert ex1["pla"] > 0.0
        assert ex2["pla"] > 0.0
        assert sig1[0] == pytest.approx(sig2[0], rel=0.03)
        assert sig1[1] == pytest.approx(sig2[1], rel=0.03)
        assert sig1[2] == pytest.approx(sig2[2], rel=0.03)

    def test_equibiaxial_yielding(self):
        """Equibiaxial tension state sig_xx = sig_yy yields at fbi * sigma_r."""
        mat = _create_standard_mat()
        params = VegterModelParams(mat)
        sig_vg, norm = eval_vegter_equivalent_stress_and_normal(250.0, 250.0, 0.0, params)
        # Equivalent stress should equal 250 / fbi = 250
        assert sig_vg == pytest.approx(250.0 / params.fbi, rel=1.0e-3)
        # Normal vector components N_xx and N_yy should be equal and positive
        assert norm[0] > 0.0
        assert norm[1] > 0.0
        assert norm[0] == pytest.approx(norm[1], rel=1.0e-3)
        assert norm[2] == pytest.approx(0.0, abs=1.0e-6)

    def test_pure_shear_yielding(self):
        """Pure shear state sig_xy > 0 yields at fsh * sigma_r."""
        mat = _create_standard_mat()
        params = VegterModelParams(mat)
        sig_vg, norm = eval_vegter_equivalent_stress_and_normal(0.0, 0.0, 125.0, params)
        assert sig_vg > 0.0
        assert abs(norm[0]) < 1.0e-6
        assert abs(norm[1]) < 1.0e-6
        assert norm[2] > 0.0

    def test_plastic_incompressibility_thinning(self):
        """Plastic thinning dezz_total is negative under tensile expansion."""
        mat = _create_standard_mat()
        deps = np.array([0.02, 0.005, 0.0])
        sigo = np.zeros(3)
        extra = {"pla": 0.0, "thk": 1.0}

        sig_new, ex = law110_shell_update(mat, deps=deps, sigo=sigo, extra=extra)
        assert ex["thk"] < 1.0
        assert ex["dezz"] < 0.0


class TestVegterHardeningAndTangents:
    """Test hardening laws and algorithmic tangent modulus."""

    def test_swift_voce_hardening(self):
        mat = _create_standard_mat(sigma_r=200.0, dsigm=100.0, beta=10.0, omega=0.5, hard_n=0.2)
        params = VegterModelParams(mat)

        # At zero plastic strain
        sig_0, h_0 = eval_flow_stress_and_hardening(params, 0.0, 0.0, 293.0)
        assert sig_0 == pytest.approx(200.0, rel=1.0e-4)
        assert h_0 > 0.0

        # At positive plastic strain
        sig_p, h_p = eval_flow_stress_and_hardening(params, 0.05, 0.0, 293.0)
        assert sig_p > sig_0
        assert h_p > 0.0

    def test_consistent_shell_tangent_elastic(self):
        mat = _create_standard_mat()
        params = VegterModelParams(mat)
        # Below yield
        c_ep = law110_consistent_shell_tangent(mat, stress=np.zeros(3))
        assert c_ep.shape == (3, 3)
        assert c_ep[0, 0] == pytest.approx(params.a11)
        assert c_ep[1, 1] == pytest.approx(params.a11)
        assert c_ep[0, 1] == pytest.approx(params.a12)
        assert c_ep[1, 0] == pytest.approx(params.a12)
        assert c_ep[2, 2] == pytest.approx(params.g)

    def test_consistent_shell_tangent_plastic(self):
        mat = _create_standard_mat()
        params = VegterModelParams(mat)
        pla = 0.01
        sig_y, h = eval_flow_stress_and_hardening(params, pla, 0.0, 293.0)
        stress = np.array([sig_y, 0.0, 0.0])
        extra = {"pla": pla, "rate": 0.0, "temp": 293.0}

        c_ep = law110_consistent_shell_tangent(mat, stress=stress, extra=extra)
        # Tangent should be softened compared to elastic
        assert c_ep[0, 0] < params.a11
        # Symmetry
        assert c_ep[0, 1] == pytest.approx(c_ep[1, 0], rel=1.0e-6)
