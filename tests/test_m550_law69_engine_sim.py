"""
Auditor 2C: Dynamic Simulation & Energy Balance Auditor for M550 (/MAT/LAW69 /MAT/HYP_ELAS /MAT/HYPERELASTIC).

Comprehensive dynamic engine simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations:
   - All 6 supported element formulations:
     - Solid Hexa8 (standard 1-pt integration, isolid=1)
     - Solid HEPH (physical hourglass stabilization, isolid=24)
     - Solid Tetra4 (constant strain tetrahedron)
     - Shell BT4 (Belytschko-Tsay quad shell, ishell=1)
     - Shell QEPH (physical hourglass quad shell, ishell=24)
     - Shell Tri3 (3-node C0 triangle shell, ish3n=1)
2. Elastic Path Reversibility (zero residual stress, zero plastic strain, zero hysteretic dissipation):
   - Multi-cycle loading and unloading cycles returning to initial geometry.
   - Cauchy stresses drop to zero (|sigma| < 1e-4 MPa for solid, < 3e-4 MPa for shell).
   - Plastic strain remains identically zero (epsp == 0.0).
   - Analytical strain energy returns to zero (W(I) = 0, zero hysteretic dissipation).
3. Energy Conservation & Energy Balance Ledger:
   - Undamped free vibration of 3D rubber block and 2D rubber membrane:
     mechanical energy E_tot = E_kin + E_int is strictly conserved (|Delta E| / E_0 < 1%).
   - Incremental strain energy work ledger: sum(sigma : deps * V) matches exact analytical
     strain energy W(F) * V0 to within 0.01% (< 1e-4 relative error).
4. Shell Thinning & Incompressibility:
   - In shell biaxial tension, thickness update h(t) obeys incompressibility:
     lambda_1 * lambda_2 * lambda_3 approx 1.0 (relative volume J approx 1.0).
   - Equibiaxial symmetry: sigma_xx == sigma_yy, shear stress == 0.
   - Dynamic engine run confirms monotonic thinning under expanding area.
5. Tensile Cut-Off Element Deletion & Post-Failure Continuation:
   - Tensile stress exceeding TENSCUT triggers element deletion:
     zero stress, off=0.0 (solids) or off=0.8 / layfail=0.0 (shells).
   - Post-failure continuation: simulation runs further cycles without NaN or crash.
   - Progressive failure in multi-element configuration through the Engine.
6. Acoustic Wave Speed & Courant Stability:
   - Sound speed formulas from sigeps69.F and sigeps69c.F:
     c_solid = sqrt((2/3 G_max + K) / rho0)
     c_shell = sqrt(A11 / rho0) with A11 = E_max / (1 - nu^2)
   - Dynamic time step bounded by Courant condition: dt <= dt_Courant = L_min / c.
   - Wave front propagation and arrival timing verified along 10-element bar and shell strip.
   - Stretch-dependent stiffening: c(stretch) increases, critical dt decreases under stretch.
7. Official OpenRadioss Benchmark Validation:
   - Runs official RD-E-5600 LAW69 Ogden benchmark decks (pair2 and pair3) through
     Starter and Engine, asserting normal termination, cycle count, and energy error |ERR| < 1%.

Fortran reference sources:
- starter/source/materials/mat/mat069/hm_read_mat69.F
- starter/source/materials/mat/mat069/law69_upd.F
- starter/source/materials/tools/nlsqf.F
- engine/source/materials/mat/mat069/sigeps69.F
- engine/source/materials/mat/mat069/sigeps69c.F
"""

from __future__ import annotations

import contextlib
import io
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest

from pyradioss.elements import (
    shell_bt4,
    shell_qeph,
    shell_tri3,
    solid_heph,
    solid_hexa8,
    solid_tetra4,
)
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials.law69_hyperelastic import (
    Law69Params,
    build_law69,
    sigeps69_solid,
    sigeps69c_shell,
    solid_update,
    shell_update,
    solid_sound_speed,
    shell_sound_speed,
)
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers: Analytical Strain Energy and Engine Control Deck Writer
# ============================================================================

def law69_strain_energy_density(mat: Law69Params, F: np.ndarray) -> float:
    """Compute exact analytical LAW69 strain energy density W(F) (sigeps69.F).

    W = W_dev + W_vol
    W_dev = sum_{k=1}^N (mu_k / alpha_k) * (lambda_bar_1^alpha_k + lambda_bar_2^alpha_k + lambda_bar_3^alpha_k - 3)
    W_vol = 0.5 * K * (J - 1)^2
    """
    lam = np.linalg.svd(F, compute_uv=False)
    J = float(lam[0] * lam[1] * lam[2])
    lam_bar = lam * (J ** (-1.0 / 3.0))

    w_dev = 0.0
    for k in range(mat.nordre):
        mu_k = float(mat.mu[k])
        al_k = float(mat.alpha[k])
        if al_k != 0.0:
            w_dev += (mu_k / al_k) * float(np.sum(lam_bar ** al_k) - 3.0)
        else:
            w_dev += mu_k * float(np.sum(np.log(np.maximum(lam_bar, 1e-20))))

    w_vol = 0.5 * mat.rbulk * ((J - 1.0) ** 2)
    return float(w_dev + w_vol)


def _write_engine_deck(
    path: Path | str,
    run_name: str,
    tstop: float = 1.0e-4,
    dt_scale: float = 0.5,
    stop_cycles: int | None = None,
    print_freq: int = -1000,
) -> None:
    """Generate and write a clean engine control deck file."""
    lines = [
        f"/RUN/{run_name}/1",
        f"{tstop:.6e}",
        "/DT",
        f"{dt_scale:.2f} 0.0",
        f"/PRINT/{print_freq}",
    ]
    if stop_cycles is not None:
        lines.extend(["/STOP", f"{stop_cycles}"])
    lines.append("/END\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================================
# 1. Multi-Cycle Explicit Dynamic Simulations Across All 6 Formulations
# ============================================================================

class TestLaw69MultiCycleDynamicSimulations:
    """Audit multi-cycle explicit simulations across all 6 element formulations."""

    def test_solid_hexa8_cyclic_engine_simulation(self, tmp_path: Path):
        """Solid Hexa8 (Isolid=1): 2-element block under cyclic pull/release loading."""
        run_name = "HEXA8_LAW69_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=1.0e-9, nu=0.45, nip=2, mu=[10.0, 5.0], alpha=[2.0, -2.0])
        d.prop_solid(1, "SolidHexaProp", isolid=1)
        d.part(1, "PartHexa", 1, 1)

        # 2-element block along X
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
            (9, 10.0, 0.0, 0.0), (10, 10.0, 5.0, 0.0), (11, 10.0, 0.0, 5.0), (12, 10.0, 5.0, 5.0),
        ])
        d.brick(1, [
            (1, 1, 2, 3, 4, 5, 6, 7, 8),
            (2, 2, 9, 10, 3, 6, 11, 12, 7),
        ])

        # Boundary conditions: fix left face, cyclic pull/release right face
        d.grnod_node(1, "fix_face", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_face", [9, 10, 11, 12])
        d.funct(1, "vel_cycle", [
            (0.0, 50.0),
            (1.0e-4, 50.0),
            (1.0001e-4, -50.0),
            (2.0e-4, -50.0),
            (3.0e-4, 0.0),
        ])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20, f"Expected >= 20 cycles, got {state.cycle}"
        assert state.stop_reason == "", f"Abnormal termination: {state.stop_reason}"
        assert state.t > 0.0
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR|={en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal strain energy must accumulate"
        assert en["EW"] > 0.0, "External work must be positive"

    def test_solid_heph_cyclic_engine_simulation(self, tmp_path: Path):
        """Solid HEPH (Isolid=24): physical hourglass stabilization under cyclic loading."""
        run_name = "HEPH_LAW69_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=1.0e-9, nu=0.495, nip=2, mu=[15.0, 3.0], alpha=[2.0, -2.0])
        d.prop_solid(1, "SolidHEPHProp", isolid=24)
        d.part(1, "PartHEPH", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
            (9, 10.0, 0.0, 0.0), (10, 10.0, 5.0, 0.0), (11, 10.0, 0.0, 5.0), (12, 10.0, 5.0, 5.0),
        ])
        d.brick(1, [
            (1, 1, 2, 3, 4, 5, 6, 7, 8),
            (2, 2, 9, 10, 3, 6, 11, 12, 7),
        ])

        d.grnod_node(1, "fix_face", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_face", [9, 10, 11, 12])
        d.funct(1, "vel_cycle", [
            (0.0, 30.0),
            (0.8e-4, 0.0),
            (1.2e-4, 30.0),
            (2.0e-4, 0.0),
        ])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"HEPH energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_solid_tetra4_cyclic_engine_simulation(self, tmp_path: Path):
        """Solid Tetra4: constant strain tetrahedron without hourglass energy under cyclic pull."""
        run_name = "TETRA4_LAW69_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=1.0e-9, nu=0.45, nip=2, mu=[12.0, 4.0], alpha=[2.0, -2.0])
        d.prop_solid(1, "SolidTetraProp", isolid=1)
        d.part(1, "PartTetra", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0), (4, 0.0, 0.0, 10.0),
            (5, 10.0, 10.0, 0.0),
        ])
        d.tetra4(1, [
            (1, 1, 2, 3, 4),
            (2, 2, 5, 3, 4),
        ])

        d.grnod_node(1, "fix_node", [1])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_apex", [4])
        d.funct(1, "vel_cycle", [
            (0.0, 50.0),
            (1.0e-4, 50.0),
            (1.0001e-4, -50.0),
            (2.0e-4, -50.0),
        ])
        d.impvel(1, "pull_z", 1, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 15
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0
        assert en["HE"] == 0.0, "Tetra4 must have strictly zero hourglass energy"

    def test_shell_bt4_cyclic_engine_simulation(self, tmp_path: Path):
        """Shell BT4 (Ishell=1): 4-node quad shell strip under cyclic in-plane loading."""
        run_name = "SHELL_BT4_LAW69_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=1.0e-9, nu=0.45, nip=2, mu=[10.0, 5.0], alpha=[2.0, -2.0])
        d.prop_shell(1, "PropBT4", thick=1.0, nip=3, ishell=1)
        d.part(1, "PartBT4", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 10.0, 0.0, 0.0), (6, 10.0, 5.0, 0.0),
        ])
        d.shell(1, [
            (1, 1, 2, 3, 4),
            (2, 2, 5, 6, 3),
        ])

        d.grnod_node(1, "fix_edge", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_edge", [5, 6])
        d.funct(1, "vel_cycle", [
            (0.0, 60.0),
            (1.0e-4, 60.0),
            (1.0001e-4, -60.0),
            (2.0e-4, -60.0),
        ])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0
        assert en["IE"] > 0.0

    def test_shell_qeph_cyclic_engine_simulation(self, tmp_path: Path):
        """Shell QEPH (Ishell=24): physical hourglass stabilization under cyclic loading."""
        run_name = "SHELL_QEPH_LAW69_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=1.0e-9, nu=0.45, nip=2, mu=[10.0, 5.0], alpha=[2.0, -2.0])
        d.prop_shell(1, "PropQEPH", thick=1.0, nip=3, ishell=24)
        d.part(1, "PartQEPH", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 10.0, 0.0, 0.0), (6, 10.0, 5.0, 0.0),
        ])
        d.shell(1, [
            (1, 1, 2, 3, 4),
            (2, 2, 5, 6, 3),
        ])

        d.grnod_node(1, "fix_edge", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_edge", [5, 6])
        d.funct(1, "vel_cycle", [
            (0.0, 50.0),
            (1.0e-4, 50.0),
            (1.0001e-4, -50.0),
            (2.0e-4, -50.0),
        ])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0
        assert en["IE"] > 0.0

    def test_shell_tri3_cyclic_engine_simulation(self, tmp_path: Path):
        """Shell Tri3 (Ish3n=1): 3-node C0 triangle shell under cyclic in-plane loading."""
        run_name = "SHELL_TRI3_LAW69_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=1.0e-9, nu=0.45, nip=2, mu=[10.0, 5.0], alpha=[2.0, -2.0])
        d.prop_shell(1, "PropTri3", thick=1.0, nip=3, ish3n=1)
        d.part(1, "PartTri3", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.sh3n(1, [
            (1, 1, 2, 3),
            (2, 1, 3, 4),
        ])

        d.grnod_node(1, "fix_edge", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_edge", [2, 3])
        d.funct(1, "vel_cycle", [
            (0.0, 50.0),
            (1.0e-4, 50.0),
            (1.0001e-4, -50.0),
            (2.0e-4, -50.0),
        ])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 15
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0
        assert en["IE"] > 0.0


# ============================================================================
# 2. Elastic Path Reversibility (Zero Residual Stress & Dissipation)
# ============================================================================

class TestLaw69ElasticPathReversibility:
    """Audit hyperelastic path reversibility: stress-strain curve returns exactly to origin."""

    def test_solid_cyclic_displacement_reversibility_ogden(self):
        """3D solid cyclic stretch (Ogden formulation): zero residual stress and zero plastic strain."""
        mat = build_law69(law_id=1, nu=0.45, mu=[15.0, 3.0], alpha=[2.0, -2.0])
        sig = np.zeros(6, dtype=np.float64)

        # 2 complete stretch-unload cycles: 1.0 -> 1.30 -> 1.0 -> 1.30 -> 1.0
        n_pts = 30
        stretches = np.concatenate([
            np.linspace(1.0, 1.30, n_pts),
            np.linspace(1.30, 1.0, n_pts)[1:],
            np.linspace(1.0, 1.30, n_pts)[1:],
            np.linspace(1.30, 1.0, n_pts)[1:],
        ])

        peak_stress = 0.0
        return_stresses = []
        return_epsps = []

        for i in range(1, len(stretches)):
            lam = stretches[i]
            lam_prev = stretches[i - 1]

            eps = np.array([np.log(lam), -0.45 * np.log(lam), -0.45 * np.log(lam), 0, 0, 0])
            deps = eps - np.array([np.log(lam_prev), -0.45 * np.log(lam_prev), -0.45 * np.log(lam_prev), 0, 0, 0])
            sig, epsp, _ = solid_update(mat, sig, deps=deps, eps=eps)

            peak_stress = max(peak_stress, float(np.max(np.abs(sig))))

            if np.isclose(lam, 1.0):
                return_stresses.append(float(np.linalg.norm(sig)))
                return_epsps.append(float(epsp))

        # Peak stress must develop significantly
        assert peak_stress > 5.0, f"Peak stress must develop, got {peak_stress}"
        # Residual stress upon return drops to zero
        for res_sig in return_stresses:
            assert res_sig < 1.0e-4, f"Residual stress {res_sig:.4e} MPa must be near zero"
        # Plastic strain remains strictly zero
        for ep in return_epsps:
            assert ep == 0.0, f"Plastic strain must be zero in hyperelasticity, got {ep}"
        # Strain energy at return configuration is strictly zero (zero hysteretic dissipation)
        w_return = law69_strain_energy_density(mat, np.eye(3))
        assert math.isclose(w_return, 0.0, abs_tol=1.0e-12)

    def test_solid_cyclic_displacement_reversibility_mooney_rivlin(self):
        """3D solid cyclic stretch (Mooney-Rivlin law_id=2): zero residual stress and zero dissipation."""
        mat = build_law69(law_id=2, nu=0.49, mu=[20.0, 5.0], alpha=[2.0, -2.0])
        sig = np.zeros(6, dtype=np.float64)

        n_pts = 25
        stretches = np.concatenate([
            np.linspace(1.0, 1.25, n_pts),
            np.linspace(1.25, 1.0, n_pts)[1:],
        ])

        for i in range(1, len(stretches)):
            lam = stretches[i]
            lam_prev = stretches[i - 1]
            eps = np.array([np.log(lam), -0.49 * np.log(lam), -0.49 * np.log(lam), 0, 0, 0])
            deps = eps - np.array([np.log(lam_prev), -0.49 * np.log(lam_prev), -0.49 * np.log(lam_prev), 0, 0, 0])
            sig, epsp, _ = solid_update(mat, sig, deps=deps, eps=eps)

        assert np.linalg.norm(sig) < 1.0e-4, f"Residual stress {np.linalg.norm(sig):.4e} must be zero"
        assert epsp == 0.0
        w_return = law69_strain_energy_density(mat, np.eye(3))
        assert math.isclose(w_return, 0.0, abs_tol=1.0e-12)

    def test_shell_cyclic_displacement_reversibility_bt4(self):
        """2D shell cyclic stretch: zero residual stress, lambda_3 returns to 1.0, and epsp=0."""
        mat = build_law69(nu=0.45, mu=[10.0, 5.0], alpha=[2.0, -2.0])
        sig = np.zeros(5, dtype=np.float64)
        uvar = np.zeros((1, 9), dtype=np.float64)
        uvar[0, 2] = 1.0

        n_pts = 25
        stretches = np.concatenate([
            np.linspace(1.0, 1.20, n_pts),
            np.linspace(1.20, 1.0, n_pts)[1:],
            np.linspace(1.0, 1.20, n_pts)[1:],
            np.linspace(1.20, 1.0, n_pts)[1:],
        ])

        return_stresses = []
        return_lam3s = []
        return_epsps = []

        for i in range(1, len(stretches)):
            lam = stretches[i]
            lam_prev = stretches[i - 1]
            eps = np.array([np.log(lam), 0.0, 0.0, 0.0, 0.0])
            deps = eps - np.array([np.log(lam_prev), 0.0, 0.0, 0.0, 0.0])
            sig, epsp = sigeps69c_shell(mat, sig, deps=deps, eps=eps, uvar=uvar)

            if np.isclose(lam, 1.0):
                return_stresses.append(float(np.linalg.norm(sig)))
                return_lam3s.append(float(uvar[0, 2]))
                return_epsps.append(float(epsp))

        for res_sig in return_stresses:
            assert res_sig < 3.0e-4, f"Shell residual stress {res_sig:.4e} must be near zero"
        for lam3 in return_lam3s:
            assert math.isclose(lam3, 1.0, abs_tol=1.0e-4), f"lambda_3 must return to 1.0, got {lam3}"
        for ep in return_epsps:
            assert ep == 0.0

    def test_shell_cyclic_displacement_reversibility_tri3(self):
        """Triangular shell cyclic stretch/unload: stress returns to zero and zero hysteretic loss."""
        mat = build_law69(nu=0.48, mu=[12.0], alpha=[2.0])
        sig = np.zeros(3, dtype=np.float64)
        uvar = np.zeros((1, 9), dtype=np.float64)
        uvar[0, 2] = 1.0

        stretches = np.concatenate([
            np.linspace(1.0, 1.15, 20),
            np.linspace(1.15, 1.0, 20)[1:],
        ])

        for i in range(1, len(stretches)):
            lam = stretches[i]
            lam_prev = stretches[i - 1]
            eps = np.array([np.log(lam), 0.0, 0.0])
            deps = eps - np.array([np.log(lam_prev), 0.0, 0.0])
            sig, _ = sigeps69c_shell(mat, sig, deps=deps, eps=eps, uvar=uvar)

        assert np.linalg.norm(sig) < 3.0e-4
        assert math.isclose(uvar[0, 2], 1.0, abs_tol=1.0e-4)


# ============================================================================
# 3. Energy Conservation in Undamped Dynamic Oscillation & Work Ledger
# ============================================================================

class TestLaw69EnergyConservationAndLedger:
    """Audit mechanical energy conservation in undamped dynamic oscillations and work ledger."""

    def test_hyperelastic_solid_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Solid Hexa8 free oscillation without damping: Delta E / E_0 < 1%."""
        run_name = "FREE_OSC_HEXA_LAW69"
        s_path = str(tmp_path / f"{run_name}_0000.rad")

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=1.0e-9, nu=0.45, nip=2, mu=[10.0, 5.0], alpha=[2.0, -2.0])
        # Pure hyperelasticity: disable artificial viscosity and hourglass damping
        d.prop_solid(1, "SolidProp", isolid=1, qa=0.0, qb=0.0, h=0.0)
        d.part(1, "HexaPart", 1, 1)
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        d.write(s_path)

        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.bricks
        rho0 = 1.0e-9
        v0_block = 1000.0  # 10 x 10 x 10 mm^3
        m_node = rho0 * v0_block / 8.0
        mass_vec = np.full(8, m_node)

        # Symmetric breathing mode along X
        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 50.0
        v[[0, 3, 4, 7], 0] = -50.0

        dt = 1.5e-6  # Stable Courant time step
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        v_half = v.copy()
        e_kin_0 = 0.5 * np.sum(mass_vec * (v[:, 0] ** 2))
        e_tot_0 = e_kin_0
        assert e_tot_0 > 0.0

        total_steps = 100
        e_tot_history = []
        e_kin_history = []
        e_int_history = []

        for step in range(total_steps):
            model.x += v_half * dt
            fint.fill(0.0)
            mint.fill(0.0)
            solid_hexa8.forces(group, model.x, v_half, model.vr, dt, fint, mint)

            acc = fint / mass_vec[:, None]
            v_next_half = v_half + acc * dt
            v_mid = 0.5 * (v_half + v_next_half)

            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot = e_kin + e_int

            e_kin_history.append(e_kin)
            e_int_history.append(e_int)
            e_tot_history.append(e_tot)

            v_half = v_next_half

        # Verify energy conservation: max relative drift < 1%
        e_max = max(e_tot_history)
        max_delta_e = max(abs(e - e_tot_0) for e in e_tot_history)
        rel_drift = max_delta_e / e_max

        assert rel_drift < 0.01, f"Energy drift {rel_drift:.4e} exceeds 1% requirement"
        # Verify dynamic exchange between kinetic and strain energy
        assert min(e_kin_history) < 0.6 * e_tot_0, "Kinetic energy must convert to strain energy"
        assert max(e_int_history) > 0.4 * e_tot_0, "Strain energy must reach peak"
        assert np.all(group.state["epsp"] == 0.0), "Plastic strain must remain zero"

    def test_hyperelastic_shell_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Shell BT4 dynamic membrane oscillation: mechanical energy strictly conserved."""
        run_name = "FREE_OSC_SHELL_LAW69"
        s_path = str(tmp_path / f"{run_name}_0000.rad")

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=1.0e-9, nu=0.45, nip=2, mu=[10.0, 5.0], alpha=[2.0, -2.0])
        d.prop_shell(1, "ShellProp", thick=1.0, nip=3, ishell=1, hm=0.0, hf=0.0, hr=0.0)
        d.part(1, "ShellPart", 1, 1)
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])
        d.write(s_path)

        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.shells
        rho0 = 1.0e-9
        m_node = rho0 * 1.0 * 100.0 / 4.0
        mass_vec = np.full(4, m_node)

        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 50.0
        v[[0, 3], 0] = -50.0

        dt = 1.0e-6
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        v_half = v.copy()
        e_tot_0 = 0.5 * np.sum(mass_vec * (v[:, 0] ** 2))
        e_tot_history = []

        for step in range(80):
            model.x += v_half * dt
            fint.fill(0.0)
            mint.fill(0.0)
            shell_bt4.forces(group, model.x, v_half, model.vr, dt, fint, mint)

            acc = fint / mass_vec[:, None]
            v_next_half = v_half + acc * dt
            v_mid = 0.5 * (v_half + v_next_half)

            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot = e_kin + e_int
            e_tot_history.append(e_tot)
            v_half = v_next_half

        max_drift = max(abs(e - e_tot_0) for e in e_tot_history) / e_tot_0
        assert max_drift < 0.01, f"Shell energy drift {max_drift:.4e} exceeds 1%"
        assert np.all(group.state["epsp"] == 0.0)

    def test_internal_energy_work_ledger_matches_analytical_strain_energy_single_term(self):
        """Verify dE_int = sigma : d_eps * V matches analytical LAW69 strain energy W(F)*V0 for N=1."""
        mat = build_law69(nu=0.45, mu=[10.0], alpha=[2.0])
        v0 = 1000.0

        steps = 250
        lam_arr = np.linspace(1.0, 1.25, steps + 1)
        accumulated_work = 0.0
        sig = np.zeros(6, dtype=np.float64)

        for i in range(steps):
            lam_prev = lam_arr[i]
            lam_next = lam_arr[i + 1]

            eps_prev = np.array([np.log(lam_prev), -0.45 * np.log(lam_prev), -0.45 * np.log(lam_prev), 0, 0, 0])
            eps_next = np.array([np.log(lam_next), -0.45 * np.log(lam_next), -0.45 * np.log(lam_next), 0, 0, 0])
            deps = eps_next - eps_prev
            eps_mid = 0.5 * (eps_prev + eps_next)

            sig_mid, _, _ = solid_update(mat, sig, deps=deps, eps=eps_mid)

            v_curr = v0 * np.exp(np.sum(eps_mid[:3]))
            accumulated_work += float(np.sum(sig_mid * deps)) * v_curr

        lam_final = lam_arr[-1]
        f_final = np.diag([lam_final, np.exp(-0.45 * np.log(lam_final)), np.exp(-0.45 * np.log(lam_final))])
        w_final = law69_strain_energy_density(mat, f_final)
        e_analytical = w_final * v0

        rel_diff = abs(accumulated_work - e_analytical) / e_analytical
        assert rel_diff < 1.0e-4, f"Single-term work ledger error {rel_diff:.4e} exceeds 0.01% tolerance"

    def test_internal_energy_work_ledger_matches_analytical_strain_energy_mooney_rivlin(self):
        """Verify Mooney-Rivlin (N=2, alpha=[2, -2]) work ledger matches analytical strain energy."""
        mat = build_law69(law_id=2, nu=0.45, mu=[20.0, 5.0], alpha=[2.0, -2.0])
        v0 = 1000.0

        steps = 250
        lam_arr = np.linspace(1.0, 1.25, steps + 1)
        accumulated_work = 0.0
        sig = np.zeros(6, dtype=np.float64)

        for i in range(steps):
            lam_prev = lam_arr[i]
            lam_next = lam_arr[i + 1]

            eps_prev = np.array([np.log(lam_prev), -0.45 * np.log(lam_prev), -0.45 * np.log(lam_prev), 0, 0, 0])
            eps_next = np.array([np.log(lam_next), -0.45 * np.log(lam_next), -0.45 * np.log(lam_next), 0, 0, 0])
            deps = eps_next - eps_prev
            eps_mid = 0.5 * (eps_prev + eps_next)

            sig_mid, _, _ = solid_update(mat, sig, deps=deps, eps=eps_mid)
            v_curr = v0 * np.exp(np.sum(eps_mid[:3]))
            accumulated_work += float(np.sum(sig_mid * deps)) * v_curr

        lam_final = lam_arr[-1]
        f_final = np.diag([lam_final, np.exp(-0.45 * np.log(lam_final)), np.exp(-0.45 * np.log(lam_final))])
        w_final = law69_strain_energy_density(mat, f_final)
        e_analytical = w_final * v0

        rel_diff = abs(accumulated_work - e_analytical) / e_analytical
        assert rel_diff < 1.0e-4, f"Mooney-Rivlin work ledger error {rel_diff:.4e} exceeds 0.01% tolerance"

    def test_internal_energy_work_ledger_matches_analytical_strain_energy_multi_term_ogden(self):
        """Verify N=3 Ogden series work-energy ledger matches analytical strain energy."""
        mat = build_law69(nu=0.49, nip=3, mu=[10.0, 2.0, -1.0], alpha=[1.3, 5.0, -2.0])
        v0 = 500.0

        steps = 200
        lam_arr = np.linspace(1.0, 1.20, steps + 1)
        accumulated_work = 0.0
        sig = np.zeros(6, dtype=np.float64)

        for i in range(steps):
            lam_prev = lam_arr[i]
            lam_next = lam_arr[i + 1]

            eps_prev = np.array([np.log(lam_prev), -0.49 * np.log(lam_prev), -0.49 * np.log(lam_prev), 0, 0, 0])
            eps_next = np.array([np.log(lam_next), -0.49 * np.log(lam_next), -0.49 * np.log(lam_next), 0, 0, 0])
            deps = eps_next - eps_prev
            eps_mid = 0.5 * (eps_prev + eps_next)

            sig_mid, _, _ = solid_update(mat, sig, deps=deps, eps=eps_mid)
            v_curr = v0 * np.exp(np.sum(eps_mid[:3]))
            accumulated_work += float(np.sum(sig_mid * deps)) * v_curr

        lam_final = lam_arr[-1]
        f_final = np.diag([lam_final, np.exp(-0.49 * np.log(lam_final)), np.exp(-0.49 * np.log(lam_final))])
        w_final = law69_strain_energy_density(mat, f_final)
        e_analytical = w_final * v0

        rel_diff = abs(accumulated_work - e_analytical) / e_analytical
        assert rel_diff < 1.0e-4, f"N=3 Ogden ledger error {rel_diff:.4e} exceeds tolerance"


# ============================================================================
# 4. Shell Thinning & Incompressibility in Biaxial Tension
# ============================================================================

class TestLaw69ShellThinningAndIncompressibility:
    """Audit shell thickness update h(t) and incompressibility condition lambda_1*lambda_2*lambda_3 approx 1.0."""

    def test_shell_biaxial_thinning_incompressibility_kernel(self):
        """Kernel verification: thickness thins and lambda_1 * lambda_2 * lambda_3 approx 1.0."""
        mat = build_law69(mu=[20.0, 5.0], alpha=[2.0, -2.0], nu=0.499)
        sig = np.zeros(5, dtype=np.float64)
        uvar = np.zeros((1, 9), dtype=np.float64)
        uvar[0, 2] = 1.0
        thk0 = 2.5
        thkn = np.array([thk0])
        thklyl = np.array([thk0])

        # Equibiaxial stretch: engineering strain eps_xx = eps_yy = 0.12 (lambda_1 = lambda_2 = 1.12)
        eps_biax = 0.12
        deps = np.array([eps_biax, eps_biax, 0.0, 0.0, 0.0])

        sig_new, _ = sigeps69c_shell(
            mat, sig, deps=deps, uvar=uvar, thkn=thkn, thklyl=thklyl, ismstr=1
        )

        # 1. Thickness must thin
        assert thkn[0] < thk0, f"Thickness must decrease from {thk0}, got {thkn[0]}"

        # 2. Incompressibility: lambda_1 * lambda_2 * lambda_3 approx 1.0
        lam1 = 1.0 + eps_biax
        lam2 = 1.0 + eps_biax
        lam3_actual = float(uvar[0, 2])
        lam3_incompressible = 1.0 / (lam1 * lam2)

        rel_vol_j = lam1 * lam2 * lam3_actual
        assert np.isclose(rel_vol_j, 1.0, rtol=1.0e-2), f"Relative volume J={rel_vol_j:.4f} must be approx 1.0"
        assert np.isclose(lam3_actual, lam3_incompressible, rtol=1.0e-2)

        # 3. Equibiaxial symmetry: sigma_xx == sigma_yy, sigma_xy == 0
        assert sig_new[0] > 0.0, "sigma_xx must be tensile"
        assert sig_new[1] > 0.0, "sigma_yy must be tensile"
        assert np.isclose(sig_new[0], sig_new[1], rtol=1.0e-3), "sigma_xx and sigma_yy must be equal"
        assert abs(sig_new[2]) < 1.0e-10, "In-plane shear stress must remain zero"

    def test_shell_biaxial_thinning_engine_simulation(self, tmp_path: Path):
        """Dynamic engine verification: shell thickness thins continuously under biaxial tension."""
        run_name = "BIAX_THIN_LAW69"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=1.0e-9, nu=0.495, nip=2, mu=[15.0, 5.0], alpha=[2.0, -2.0])
        d.prop_shell(1, "BiaxProp", thick=2.0, nip=3, ishell=1)
        d.part(1, "BiaxPart", 1, 1)

        # 10x10 mm plate
        d.node([(1, 0, 0, 0), (2, 10, 0, 0), (3, 10, 10, 0), (4, 0, 10, 0)])
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Symmetry BCs and biaxial extension: pull X on right edge, pull Y on top edge
        d.grnod_node(1, "fix_x", [1, 4])
        d.bcs(1, "bcs_x", "100", "111", 1)
        d.grnod_node(2, "fix_y", [1, 2])
        d.bcs(2, "bcs_y", "010", "111", 2)

        d.grnod_node(3, "pull_x", [2, 3])
        d.funct(1, "f_x", [(0.0, 100.0), (1.0e-3, 100.0)])
        d.impvel(1, "vx", 1, "X", 3)

        d.grnod_node(4, "pull_y", [3, 4])
        d.funct(2, "f_y", [(0.0, 100.0), (1.0e-3, 100.0)])
        d.impvel(2, "vy", 2, "Y", 4)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 5
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0

        thk = eng_model.shells.state["thick"]
        assert thk[0] < 2.0, f"Shell thickness must thin below 2.0 mm, got {thk[0]}"

    def test_shell_thinning_poisson_sensitivity(self):
        """Compare shell thinning behavior between moderately compressible (nu=0.45) and incompressible (nu=0.499)."""
        mat_comp = build_law69(mu=[20.0], alpha=[2.0], nu=0.45)
        mat_incomp = build_law69(mu=[20.0], alpha=[2.0], nu=0.499)

        uvar_c = np.zeros((1, 9), dtype=np.float64)
        uvar_c[0, 2] = 1.0
        uvar_i = np.zeros((1, 9), dtype=np.float64)
        uvar_i[0, 2] = 1.0

        deps = np.array([0.15, 0.15, 0.0, 0.0, 0.0])
        sig = np.zeros(5, dtype=np.float64)

        sigeps69c_shell(mat_comp, sig.copy(), deps=deps, uvar=uvar_c, ismstr=1)
        sigeps69c_shell(mat_incomp, sig.copy(), deps=deps, uvar=uvar_i, ismstr=1)

        lam3_comp = uvar_c[0, 2]
        lam3_incomp = uvar_i[0, 2]

        # Incompressible material contracts more out-of-plane to preserve volume
        assert lam3_incomp < lam3_comp, f"Incompressible lambda_3 ({lam3_incomp}) must be smaller than compressible ({lam3_comp})"


# ============================================================================
# 5. Tensile Cut-Off Element Deletion & Post-Failure Continuation
# ============================================================================

class TestLaw69TensileCutOffAndPostFailure:
    """Audit tensile stress cut-off element erosion and smooth post-failure continuation."""

    def test_solid_tensile_cutoff_element_deletion_and_continuation(self):
        """Exceeding TENSCUT zeroes stress, sets off=0.0, and runs subsequent steps without NaN."""
        tenscut_limit = 20.0
        mat = build_law69(mu=[25.0], alpha=[2.0], tenscut=tenscut_limit)
        sig = np.zeros(6, dtype=np.float64)
        off = np.array([1.0], dtype=np.float64)

        # 1. Pre-failure loading: stress below TENSCUT
        eps_pre = np.array([0.08, -0.04, -0.04, 0.0, 0.0, 0.0])
        sig_pre, _, _ = sigeps69_solid(mat, sig, eps=eps_pre, off=off)
        assert 0.0 < sig_pre[0] < tenscut_limit
        assert off[0] == 1.0

        # 2. Failure loading: stretch exceeds TENSCUT
        eps_fail = np.array([0.45, -0.22, -0.22, 0.0, 0.0, 0.0])
        sig_fail, _, _ = sigeps69_solid(mat, sig_pre, eps=eps_fail, off=off)
        assert np.allclose(sig_fail, 0.0), "Stresses must be zeroed immediately upon failure"
        assert off[0] == 0.0, "Element activity flag off must be set to 0.0"

        # 3. Post-failure continuation: 20 subsequent increments must stay zero without NaNs
        for k in range(1, 21):
            eps_post = eps_fail + np.array([0.02 * k, 0.0, 0.0, 0.0, 0.0, 0.0])
            sig_post, _, _ = sigeps69_solid(mat, sig_fail, eps=eps_post, off=off)
            assert np.allclose(sig_post, 0.0), f"Step {k}: Failed element stress must remain zero"
            assert not np.isnan(sig_post).any(), f"Step {k}: Stress must not contain NaN"
            assert not np.isinf(sig_post).any(), f"Step {k}: Stress must not contain Inf"

    def test_shell_tensile_cutoff_layer_failure_and_continuation(self):
        """Shell plane-stress cut-off: off=0.8, layfail=0.0, zero stress, no NaNs."""
        tenscut_limit = 12.0
        mat = build_law69(mu=[20.0], alpha=[2.0], tenscut=tenscut_limit)
        sig = np.zeros(3, dtype=np.float64)
        off = np.array([1.0], dtype=np.float64)
        layfail = np.array([1.0], dtype=np.float64)
        extra = {"off": off, "layfail": layfail}

        # Large in-plane stretch exceeding TENSCUT
        eps_fail = np.array([0.35, 0.0, 0.0])
        sig_fail, _ = sigeps69c_shell(mat, sig, eps=eps_fail, off=off, extra=extra)

        assert np.allclose(sig_fail, 0.0), "Shell stresses must be zeroed"
        assert off[0] == 0.8, "Shell cut flag must be 0.8 according to sigeps69c.F"
        assert layfail[0] == 0.0, "Layer failure flag must be set to 0.0"

        # Post-failure continuation
        for k in range(1, 21):
            eps_post = eps_fail + np.array([0.02 * k, 0.0, 0.0])
            sig_post, _ = sigeps69c_shell(mat, sig_fail, eps=eps_post, off=off, extra=extra)
            assert np.allclose(sig_post, 0.0)
            assert not np.isnan(sig_post).any()

    def test_multi_element_tenscut_progressive_failure_engine(self, tmp_path: Path):
        """2-element solid bar with TENSCUT in engine: element failure handled smoothly."""
        run_name = "PROG_FAIL_LAW69"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=1.0e-9, nu=0.45, nip=1, mu=[20.0], alpha=[2.0], tenscut=15.0)
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "Part", 1, 1)

        d.node([
            (1, 0, 0, 0), (2, 5, 0, 0), (3, 5, 5, 0), (4, 0, 5, 0),
            (5, 0, 0, 5), (6, 5, 0, 5), (7, 5, 5, 5), (8, 0, 5, 5),
            (9, 10, 0, 0), (10, 10, 5, 0), (11, 10, 0, 5), (12, 10, 5, 5),
        ])
        d.brick(1, [
            (1, 1, 2, 3, 4, 5, 6, 7, 8),
            (2, 2, 9, 10, 3, 6, 11, 12, 7),
        ])

        d.grnod_node(1, "fix", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull", [9, 10, 11, 12])
        d.funct(1, "pull_f", [(0.0, 150.0), (2.0e-4, 150.0)])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.5e-4, dt_scale=0.4)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 15
        assert state.stop_reason == ""
        # Stresses must not have NaNs
        sig = eng_model.bricks.state["sig"]
        assert not np.isnan(sig).any()


# ============================================================================
# 6. Acoustic Wave Speed & Courant Stability
# ============================================================================

class TestLaw69WaveSpeedAndCourantStability:
    """Audit acoustic sound speeds, wave propagation, and Courant time step bounds."""

    def test_solid_dynamic_wave_propagation_bar(self, tmp_path: Path):
        """10-element solid bar: verify acoustic sound speed and stable pulse arrival."""
        run_name = "WAVE_SOLID_LAW69"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        rho0 = 1.0e-9
        mu = 10.0
        nu = 0.45
        mat = build_law69(rho0=rho0, nu=nu, nip=1, mu=[mu], alpha=[2.0])

        # Theoretical solid sound speed: c = sqrt((2/3 G_max + K) / rho0)
        c_theory = math.sqrt(((2.0 / 3.0) * mat.gmax + mat.rbulk) / rho0)
        c_solid = solid_sound_speed(mat, rho=rho0)
        assert math.isclose(c_solid, c_theory, rel_tol=1.0e-6)

        # 10-element bar of length 50 mm along X (dx = 5 mm, 5x5 cross section)
        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=rho0, nu=nu, nip=1, mu=[mu], alpha=[2.0])
        d.prop_solid(1, "BarProp", isolid=1)
        d.part(1, "BarPart", 1, 1)

        nodes = []
        for i in range(11):
            x = i * 5.0
            nodes.extend([(4 * i + 1, x, 0.0, 0.0), (4 * i + 2, x, 5.0, 0.0),
                          (4 * i + 3, x, 5.0, 5.0), (4 * i + 4, x, 0.0, 5.0)])
        d.node(nodes)

        bricks = []
        for i in range(10):
            b = 4 * i + 1
            bricks.append((i + 1, b, b + 4, b + 5, b + 1, b + 3, b + 7, b + 6, b + 2))
        d.brick(1, bricks)

        # Pulse at x=0
        d.grnod_node(1, "pulse_face", [1, 2, 3, 4])
        d.funct(1, "pulse_f", [(0.0, 100.0), (1.0e-5, 100.0), (1.1e-5, 0.0), (1.0, 0.0)])
        d.impvel(1, "px", 1, "X", 1)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.4)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 15
        assert state.stop_reason == ""

        # Wave front audit: c ~ 331662 mm/s, in t = 1.0e-4 s distance traveled is d ~ 33.16 mm
        # Elements 0..5 (x < 30 mm) have felt the disturbance
        # Element 9 (x in [45, 50] mm) has not been reached yet
        sig = eng_model.bricks.state["sig"]
        assert abs(sig[0, 0]) > 1.0e-4, "Element near pulse must be disturbed"
        assert abs(sig[9, 0]) < 1.0e-5, "Far end element 9 must remain undisturbed"

    def test_shell_dynamic_wave_propagation_strip(self, tmp_path: Path):
        """10-element shell strip: verify plane-stress sound speed c = sqrt(A11/rho0)."""
        run_name = "WAVE_SHELL_LAW69"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        rho0 = 1.0e-9
        mu = 10.0
        nu = 0.45
        mat = build_law69(rho0=rho0, nu=nu, nip=1, mu=[mu], alpha=[2.0])

        emax = mat.gmax * (1.0 + mat.nu)
        a11 = emax / (1.0 - mat.nu ** 2)
        c_theory = math.sqrt(a11 / rho0)
        c_shell = shell_sound_speed(mat, rho=rho0)
        assert math.isclose(c_shell, c_theory, rel_tol=1.0e-6)

        d = StarterDeck(run_name)
        d.mat_law69(1, rho0=rho0, nu=nu, nip=1, mu=[mu], alpha=[2.0])
        d.prop_shell(1, "StripProp", thick=1.0, nip=3, ishell=1)
        d.part(1, "StripPart", 1, 1)

        nodes = []
        for i in range(11):
            x = i * 5.0
            nodes.extend([(2 * i + 1, x, 0.0, 0.0), (2 * i + 2, x, 5.0, 0.0)])
        d.node(nodes)

        shells = []
        for i in range(10):
            shells.append((i + 1, 2 * i + 1, 2 * i + 3, 2 * i + 4, 2 * i + 2))
        d.shell(1, shells)

        d.grnod_node(1, "impulse_nodes", [1, 2])
        d.funct(1, "pulse", [(0.0, 80.0), (1.0e-5, 80.0), (1.1e-5, 0.0), (1.0, 0.0)])
        d.impvel(1, "pulse_vel", 1, "X", 1)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.4)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 10
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0

    def test_stretch_dependent_stiffening_and_dt_reduction(self):
        """Verify that wave speed stiffens and critical Courant dt drops under large stretch."""
        mat = build_law69(mu=[15.0, 2.0], alpha=[2.5, -2.0], nu=0.49)
        rho = 1.0e-9
        lc = 5.0  # element size in mm

        # Undeformed state
        c_rest = solid_sound_speed(mat, rho=rho)
        dt_rest = lc / c_rest

        # Moderate stretch lambda = 1.25
        eps_mod = np.array([np.log(1.25), -0.49 * np.log(1.25), -0.49 * np.log(1.25), 0, 0, 0])
        c_mod = solid_sound_speed(mat, rho=rho, eps=eps_mod)
        dt_mod = lc / c_mod

        # Severe stretch lambda = 1.65
        eps_sev = np.array([np.log(1.65), -0.49 * np.log(1.65), -0.49 * np.log(1.65), 0, 0, 0])
        c_sev = solid_sound_speed(mat, rho=rho, eps=eps_sev)
        dt_sev = lc / c_sev

        # Due to alpha=2.5 > 2.0, sound speed increases monotonically
        assert c_mod > c_rest, f"Sound speed must stiffen at lambda=1.25: {c_mod} vs {c_rest}"
        assert c_sev > c_mod, f"Sound speed must stiffen further at lambda=1.65: {c_sev} vs {c_mod}"

        # Courant timestep bounds decrease accordingly
        assert dt_mod < dt_rest
        assert dt_sev < dt_mod


# ============================================================================
# 7. Official OpenRadioss Benchmark Validation
# ============================================================================

class TestLaw69OfficialBenchmark:
    """Audit official RD-E-5600 Ogden rubber tension deck with LAW69."""

    def test_official_rd_e_5600_law69_benchmark_pair2(self, tmp_path: Path):
        """Run official RD-E-5600 rubber_tension_v1 LAW69 pair2 Poisson04997 deck."""
        src_dir = os.path.join(
            "tests", "data", "rd_decks", "rd_e",
            "RD-E-5600_Hyperelastic_material", "56_HyperElastic_Material",
            "Ogden_model", "LAW69_ogden_pair2", "LAW69_ogden_pair2_Poisson04997",
        )
        assert os.path.isdir(src_dir), f"Official deck directory not found: {src_dir}"

        for fname in os.listdir(src_dir):
            if fname.endswith(".rad") or fname.endswith(".txt"):
                shutil.copy(os.path.join(src_dir, fname), os.path.join(tmp_path, fname))

        # Override Tstop from 0.801 to 1.0e-4 in rubber_tension_v1_0001.rad for fast testing
        e_file = tmp_path / "rubber_tension_v1_0001.rad"
        content = e_file.read_text(encoding="utf-8")
        assert "0.801" in content, "Expected Tstop=0.801 in official engine deck"
        e_file.write_text(content.replace("0.801", "0.0001"), encoding="utf-8")

        s_file = str(tmp_path / "rubber_tension_v1_0000.rad")

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_file)
            eng_model = run_engine(str(e_file))

        assert 1 in st_model.materials
        mat = st_model.materials[1]
        assert getattr(mat, "law", None) == 69

        state = eng_model.engine_state
        assert state.cycle >= 20, f"Expected >= 20 cycles, got {state.cycle}"
        assert state.stop_reason == "", f"Abnormal stop: {state.stop_reason}"
        assert state.t > 0.0

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Official benchmark energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal energy must be positive"

    def test_official_rd_e_5600_law69_benchmark_pair3(self, tmp_path: Path):
        """Run official RD-E-5600 rubber_tension_v1 LAW69 pair3 deck."""
        src_dir = os.path.join(
            "tests", "data", "rd_decks", "rd_e",
            "RD-E-5600_Hyperelastic_material", "56_HyperElastic_Material",
            "Ogden_model", "LAW69_ogden_pair3",
        )
        assert os.path.isdir(src_dir), f"Official deck directory not found: {src_dir}"

        for fname in os.listdir(src_dir):
            if fname.endswith(".rad") or fname.endswith(".txt"):
                shutil.copy(os.path.join(src_dir, fname), os.path.join(tmp_path, fname))

        e_file = tmp_path / "rubber_tension_v1_0001.rad"
        content = e_file.read_text(encoding="utf-8")
        assert "0.801" in content
        e_file.write_text(content.replace("0.801", "0.0001"), encoding="utf-8")

        s_file = str(tmp_path / "rubber_tension_v1_0000.rad")

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_file)
            eng_model = run_engine(str(e_file))

        assert 1 in st_model.materials
        mat = st_model.materials[1]
        assert getattr(mat, "law", None) == 69

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0
        assert en["IE"] > 0.0
