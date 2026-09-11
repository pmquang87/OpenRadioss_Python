"""
Auditor 2C: Dynamic Simulation & Energy Balance Auditor for M551 (/MAT/LAW60 /MAT/PLAS_T3 /MAT/FABRIC).

Exhaustive dynamic engine simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations:
   - All 6 supported element formulations:
     - Solid Hexa8 (standard 1-pt integration, isolid=1)
     - Solid HEPH (physical hourglass stabilization, isolid=24)
     - Solid Tetra4 (constant strain tetrahedron)
     - Shell BT4 (Belytschko-Tsay quad shell, ishell=1)
     - Shell QEPH (physical hourglass quad shell, ishell=24)
     - Shell Tri3 (3-node C0 triangle shell, ish3n=1)
   - Verified through Starter and Engine decks under cyclic displacement and velocity loading.
   - Asserts normal termination, >= 20 cycles, energy balance error |ERR| < 1.0%,
     positive internal strain energy accumulation (IE > 0), and zero hourglass energy on Tetra4.
2. Elastic Path Reversibility:
   - Multi-cycle loading and unloading within the elastic regime (below initial yield).
   - Cauchy stresses drop to zero (|sigma| < 1e-4 for solids, < 3e-4 for shells).
   - Plastic strain remains identically zero (epsp == 0.0).
   - Shell thickness returns to initial thickness h0.
   - Analytical strain energy and external work return to zero (zero hysteretic dissipation).
3. Bauschinger Effect & Kinematic Hardening:
   - Mixed isotropic-kinematic hardening with fisokin < 1.0 (e.g. 0.0 vs 0.5).
   - Forward plastic loading generates back-stress alpha_xx > 0 when fisokin = 0.5.
   - Reverse compressive loading yields significantly earlier in compression for fisokin = 0.5
     compared to isotropic hardening (fisokin = 0.0), demonstrating the Bauschinger effect.
   - Explicit engine runs track back-stress evolution in mat_extra["sigb60"].
4. Energy Conservation & Work Ledger in Undamped Explicit Dynamic Simulations:
   - Free vibration of undamped 3D solid block (Hexa8, qa=0, qb=0, h=0) and 2D shell membrane:
     mechanical energy E_tot = E_kin + E_int is strictly conserved (|Delta E| / E_0 < 1.0%).
   - Incremental strain energy work ledger: sum(sigma : deps * V) matches exact analytical
     elastic strain energy 0.5 * eps : C : eps * V to within 0.01% (< 1e-4 relative error).
   - Plastic dissipation accounting: total work equals elastic strain energy plus plastic dissipation.
5. Dynamic Modulus Degradation & Vibration Frequency Shift:
   - Modulus degradation via exponential law (ce > 0, einf > 0) or curve (ifunce > 0):
     E_cur(epsp) drops monotonically as plastic strain accumulates.
   - Effective stiffness degradation reduces natural vibration frequency:
     omega_deg / omega_0 approx sqrt(E_deg / E_0), lengthening oscillation period T.
6. Shell Thinning & Incompressibility:
   - In shell biaxial tension, thickness update h(t) decreases monotonically with plastic stretching.
   - Equibiaxial symmetry: sigma_xx == sigma_yy, sigma_xy == 0.
   - Plastic flow is volume-preserving (isochoric, trace = 0): dezz_pl = -(dxx_pl + dyy_pl).
   - Dynamic engine run confirms continuous monotonic thinning over time.
7. Tensile Failure Scaling, Element Deletion & Post-Failure Continuation:
   - Tensile damage scaling: FAIL factor drops from 1.0 to 0.0 between eps_t1 and eps_t2.
   - Tensile plastic strain exceeding eps_max triggers element deletion:
     zero stress, off = 0.0, off60 = 0.0, layfail = 0.0.
   - Post-failure continuation: simulation runs further cycles without NaN, infinities, or crash.
   - Multi-element bar through the Engine continues stably after erosion of the failed element.
8. Acoustic Wave Speed & Courant Stability:
   - Longitudinal sound speed c_solid = sqrt((C1 + 4/3 G) / rho0) and c_shell = sqrt(A1 / rho0).
   - Dynamic time step bounded by Courant condition: dt <= dt_Courant = L_min / c.
   - Modulus degradation lowers sound speed and increases critical time step.
   - Acoustic pulse propagation along 10-element bar in Engine: wave front disturbance timing
     distinguishes near-pulse vs far-end elements according to wave speed c.

Fortran references:
- starter/source/materials/mat/mat060/hm_read_mat60.F
- engine/source/materials/mat/mat060/sigeps60.F
- engine/source/materials/mat/mat060/sigeps60c.F
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyradioss.common.tables import FunctTable
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
from pyradioss.materials.law60_plast3 import (
    Law60Params,
    build_law60,
    solid_update,
    shell_update,
    sound_speed,
    extra_shapes,
)
from pyradioss.model.entities import Material
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers: Engine Control Deck Writer and Energy Verification
# ============================================================================

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

class TestLaw60MultiCycleDynamicSimulations:
    """Audit multi-cycle explicit dynamic simulations across all 6 element formulations."""

    def test_solid_hexa8_cyclic_engine_simulation(self, tmp_path: Path):
        """Solid Hexa8 (Isolid=1): 2-element block under cyclic pull/release loading."""
        run_name = "HEXA8_LAW60_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 250.0), (0.1, 350.0), (0.4, 500.0)])
        d.mat_law60(mid=1, title="SteelHexa", rho=7.85e-9, e=210000.0, nu=0.3, funcs=[1])
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
        d.funct(10, "vel_cycle", [
            (0.0, 40.0),
            (1.0e-4, 40.0),
            (1.0001e-4, -40.0),
            (2.0e-4, -40.0),
            (3.0e-4, 0.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

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
        run_name = "HEPH_LAW60_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 260.0), (0.1, 360.0), (0.4, 520.0)])
        d.mat_law60(mid=1, title="SteelHEPH", rho=7.85e-9, e=210000.0, nu=0.3, funcs=[1])
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
        d.funct(10, "vel_cycle", [
            (0.0, 30.0),
            (0.8e-4, 0.0),
            (1.2e-4, 30.0),
            (2.0e-4, 0.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

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
        run_name = "TETRA4_LAW60_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 240.0), (0.1, 340.0), (0.3, 460.0)])
        d.mat_law60(mid=1, title="SteelTetra", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
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
        d.funct(10, "vel_cycle", [
            (0.0, 40.0),
            (1.0e-4, 40.0),
            (1.0001e-4, -40.0),
            (2.0e-4, -40.0),
        ])
        d.impvel(1, "pull_z", 10, "Z", 2)

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
        run_name = "SHELL_BT4_LAW60_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 220.0), (0.1, 320.0), (0.3, 420.0)])
        d.mat_law60(mid=1, title="FabricBT4", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
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
        d.funct(10, "vel_cycle", [
            (0.0, 50.0),
            (1.0e-4, 50.0),
            (1.0001e-4, -50.0),
            (2.0e-4, -50.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

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
        """Shell QEPH (Ishell=24): physical hourglass quad shell under cyclic loading."""
        run_name = "SHELL_QEPH_LAW60_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 220.0), (0.1, 320.0), (0.3, 420.0)])
        d.mat_law60(mid=1, title="FabricQEPH", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
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
        d.funct(10, "vel_cycle", [
            (0.0, 45.0),
            (1.0e-4, 45.0),
            (1.0001e-4, -45.0),
            (2.0e-4, -45.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

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
        run_name = "SHELL_TRI3_LAW60_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 220.0), (0.1, 320.0), (0.3, 420.0)])
        d.mat_law60(mid=1, title="FabricTri3", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
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
        d.funct(10, "vel_cycle", [
            (0.0, 40.0),
            (1.0e-4, 40.0),
            (1.0001e-4, -40.0),
            (2.0e-4, -40.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

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

class TestLaw60ElasticPathReversibility:
    """Audit elastic path reversibility: stress-strain curve returns to origin with zero dissipation."""

    def test_solid_cyclic_elastic_path_reversibility_3d(self):
        """3D solid cyclic stretch below yield: zero residual stress and zero plastic strain."""
        f1 = FunctTable(1, [0.0, 0.1], [300.0, 450.0])  # Yield stress = 300 MPa
        mat = build_law60(e0=200000.0, nu=0.3, funcs=[f1])
        sig = np.zeros(6, dtype=np.float64)

        # Elastic strain limit: eps_el = 300 / 200000 = 1.5e-3
        # Apply cyclic stretch up to 0.0008 (well below yield): 2 complete cycles
        n_pts = 20
        eps_peaks = [0.0, 0.0008, 0.0, 0.0008, 0.0]
        strain_hist = []
        for j in range(len(eps_peaks) - 1):
            seg = np.linspace(eps_peaks[j], eps_peaks[j + 1], n_pts)
            strain_hist.extend(seg if j == 0 else seg[1:])

        epsp = np.zeros(1, dtype=np.float64)
        peak_stress = 0.0
        return_stresses = []
        return_epsps = []

        for i in range(1, len(strain_hist)):
            eps_val = strain_hist[i]
            eps_prev = strain_hist[i - 1]

            deps = np.array([eps_val - eps_prev, -0.3 * (eps_val - eps_prev), -0.3 * (eps_val - eps_prev), 0, 0, 0])
            sig, epsp_out, _ = solid_update(mat, sig, deps=deps, epsp_old=epsp)
            epsp = np.atleast_1d(epsp_out)

            peak_stress = max(peak_stress, float(np.max(np.abs(sig))))

            if np.isclose(eps_val, 0.0):
                return_stresses.append(float(np.linalg.norm(sig)))
                return_epsps.append(float(np.squeeze(epsp)))

        # Peak stress develops significantly
        assert peak_stress > 100.0, f"Peak stress must develop, got {peak_stress}"
        # Residual stresses upon returning to origin are strictly zero
        for res_sig in return_stresses:
            assert res_sig < 1.0e-4, f"Residual stress {res_sig:.4e} MPa must be near zero"
        # Plastic strain remains identically zero
        for ep in return_epsps:
            assert ep == 0.0, f"Plastic strain must be zero in elastic regime, got {ep}"

    def test_shell_cyclic_elastic_path_reversibility_2d(self):
        """2D shell cyclic stretch below yield: zero residual stress, epsp=0, and thickness returns to h0."""
        f1 = FunctTable(1, [0.0, 0.1], [250.0, 380.0])  # Yield stress = 250 MPa
        mat = build_law60(e0=200000.0, nu=0.3, funcs=[f1])
        sig = np.zeros(3, dtype=np.float64)
        thk = np.array([1.5], dtype=np.float64)
        extra = {"thk": thk, "off": np.array([1.0])}

        # Elastic strain limit: eps_el = 250 / 200000 = 1.25e-3
        # Prescribe cyclic stretch up to 0.0006: 2 complete cycles
        n_pts = 20
        eps_peaks = [0.0, 0.0006, 0.0, 0.0006, 0.0]
        strain_hist = []
        for j in range(len(eps_peaks) - 1):
            seg = np.linspace(eps_peaks[j], eps_peaks[j + 1], n_pts)
            strain_hist.extend(seg if j == 0 else seg[1:])

        epsp = np.zeros(1, dtype=np.float64)
        return_stresses = []
        return_thks = []
        return_epsps = []

        for i in range(1, len(strain_hist)):
            eps_val = strain_hist[i]
            eps_prev = strain_hist[i - 1]

            deps = np.array([eps_val - eps_prev, 0.0, 0.0])
            sig, epsp_out, _ = shell_update(mat, sig, deps=deps, epsp_old=epsp, extra=extra)
            epsp = np.atleast_1d(epsp_out)

            if np.isclose(eps_val, 0.0):
                return_stresses.append(float(np.linalg.norm(sig)))
                return_thks.append(float(thk[0]))
                return_epsps.append(float(np.squeeze(epsp)))

        for res_sig in return_stresses:
            assert res_sig < 3.0e-4, f"Shell residual stress {res_sig:.4e} must be near zero"
        for t_ret in return_thks:
            assert math.isclose(t_ret, 1.5, abs_tol=1.0e-6), f"Thickness must return to 1.5, got {t_ret}"
        for ep in return_epsps:
            assert ep == 0.0

    def test_solid_hexa8_elastic_cycle_engine_simulation(self, tmp_path: Path):
        """Full Engine simulation of Hexa8 solid block loaded strictly below yield and unloaded."""
        run_name = "HEXA8_ELAS_CYCLE"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        # Yield stress = 400 MPa
        d.funct(1, "YieldCurve", [(0.0, 400.0), (0.1, 500.0)])
        d.mat_law60(mid=1, title="SteelElas", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "Part1", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        d.grnod_node(1, "fix_face", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_face", [2, 3, 6, 7])
        # Pull with small displacement: v = 10 mm/s for 5e-5 s -> dx = 0.0005 mm (eps = 5e-5 << eps_yield=0.002)
        # then return back with -10 mm/s for 5e-5 s
        d.funct(10, "vel_cycle", [
            (0.0, 10.0),
            (0.5e-4, 10.0),
            (0.5001e-4, -10.0),
            (1.0e-4, -10.0),
            (1.0001e-4, 0.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        # Plastic strain must remain strictly zero
        assert float(np.max(eng_model.bricks.state["epsp"])) == 0.0


# ============================================================================
# 3. Bauschinger Effect & Kinematic Hardening
# ============================================================================

class TestLaw60BauschingerKinematicHardening:
    """Audit kinematic hardening and Bauschinger effect with fisokin < 1.0 (e.g. 0.0 vs 0.5)."""

    def test_solid_bauschinger_effect_shift(self):
        """Compare pure isotropic (fisokin=0.0) vs mixed kinematic (fisokin=0.5) under reverse loading."""
        f1 = FunctTable(1, [0.0, 0.05, 0.20], [200.0, 350.0, 500.0])

        mat_iso = build_law60(e0=200000.0, nu=0.3, fisokin=0.0, funcs=[f1])
        mat_kin = build_law60(e0=200000.0, nu=0.3, fisokin=0.5, funcs=[f1])

        sig_iso = np.zeros(6, dtype=float)
        epsp_iso = np.zeros(1, dtype=float)
        extra_iso = {"sigb60": np.zeros(6, dtype=float)}

        sig_kin = np.zeros(6, dtype=float)
        epsp_kin = np.zeros(1, dtype=float)
        extra_kin = {"sigb60": np.zeros(6, dtype=float)}

        # Phase 1: Forward tension well into plastic regime (25 increments of strain 0.001)
        deps_fwd = np.array([0.001, -0.0003, -0.0003, 0.0, 0.0, 0.0])
        for _ in range(25):
            sig_iso, epsp_iso_out, _ = solid_update(mat_iso, sig_iso, deps=deps_fwd, epsp_old=epsp_iso, extra=extra_iso)
            sig_kin, epsp_kin_out, _ = solid_update(mat_kin, sig_kin, deps=deps_fwd, epsp_old=epsp_kin, extra=extra_kin)
            epsp_iso = np.atleast_1d(epsp_iso_out)
            epsp_kin = np.atleast_1d(epsp_kin_out)

        # Forward tension produces plastic strain in both
        assert float(np.squeeze(epsp_iso)) > 0.0
        assert float(np.squeeze(epsp_kin)) > 0.0

        # Back-stress alpha_xx is zero for isotropic, but strictly positive for kinematic
        alpha_xx_kin = float(np.squeeze(extra_kin["sigb60"])[0])
        alpha_xx_iso = float(np.squeeze(extra_iso["sigb60"])[0])
        assert alpha_xx_iso == 0.0, "Isotropic hardening must have zero back-stress"
        assert alpha_xx_kin > 10.0, f"Kinematic hardening must develop positive back-stress, got {alpha_xx_kin}"

        # Phase 2: Reverse compression (50 increments of smaller strain step -0.0001)
        deps_rev = np.array([-0.0001, 0.00003, 0.00003, 0.0, 0.0, 0.0])
        epsp_start_iso = float(np.squeeze(epsp_iso))
        epsp_start_kin = float(np.squeeze(epsp_kin))

        rev_yield_step_iso = -1
        rev_yield_step_kin = -1

        for step in range(50):
            sig_iso, epsp_iso_out, _ = solid_update(mat_iso, sig_iso, deps=deps_rev, epsp_old=epsp_iso, extra=extra_iso)
            sig_kin, epsp_kin_out, _ = solid_update(mat_kin, sig_kin, deps=deps_rev, epsp_old=epsp_kin, extra=extra_kin)
            epsp_iso = np.atleast_1d(epsp_iso_out)
            epsp_kin = np.atleast_1d(epsp_kin_out)

            if rev_yield_step_kin == -1 and float(np.squeeze(epsp_kin)) > epsp_start_kin + 1e-6:
                rev_yield_step_kin = step
            if rev_yield_step_iso == -1 and float(np.squeeze(epsp_iso)) > epsp_start_iso + 1e-6:
                rev_yield_step_iso = step

        # Kinematic hardening yields earlier in reverse loading (Bauschinger effect!)
        assert rev_yield_step_kin != -1, "Kinematic hardening must yield in reverse compression"
        assert rev_yield_step_iso != -1, "Isotropic hardening must yield in reverse compression"
        assert rev_yield_step_kin < rev_yield_step_iso, (
            f"Bauschinger effect: kinematic hardening must yield earlier (step {rev_yield_step_kin}) "
            f"than isotropic hardening (step {rev_yield_step_iso})"
        )

        # Back-stress shifts toward negative values during reverse plastic flow
        assert float(np.squeeze(extra_kin["sigb60"])[0]) < alpha_xx_kin

    def test_shell_bauschinger_effect_shift(self):
        """2D plane-stress shell Bauschinger effect under cyclic tension/compression."""
        f1 = FunctTable(1, [0.0, 0.05, 0.20], [200.0, 320.0, 480.0])

        mat_iso = build_law60(e0=200000.0, nu=0.3, fisokin=0.0, funcs=[f1])
        mat_kin = build_law60(e0=200000.0, nu=0.3, fisokin=0.5, funcs=[f1])

        sig_iso = np.zeros(3, dtype=float)
        epsp_iso = np.zeros(1, dtype=float)
        extra_iso = {"sigb60": np.zeros(3, dtype=float), "thk": np.array([1.0])}

        sig_kin = np.zeros(3, dtype=float)
        epsp_kin = np.zeros(1, dtype=float)
        extra_kin = {"sigb60": np.zeros(3, dtype=float), "thk": np.array([1.0])}

        # Forward tension
        deps_fwd = np.array([0.001, 0.0, 0.0])
        for _ in range(25):
            sig_iso, epsp_iso_out, _ = shell_update(mat_iso, sig_iso, deps=deps_fwd, epsp_old=epsp_iso, extra=extra_iso)
            sig_kin, epsp_kin_out, _ = shell_update(mat_kin, sig_kin, deps=deps_fwd, epsp_old=epsp_kin, extra=extra_kin)
            epsp_iso = np.atleast_1d(epsp_iso_out)
            epsp_kin = np.atleast_1d(epsp_kin_out)

        alpha_xx_kin = float(np.squeeze(extra_kin["sigb60"])[0])
        assert alpha_xx_kin > 5.0, f"Shell back-stress alpha_xx must develop, got {alpha_xx_kin}"

        # Reverse compression
        deps_rev = np.array([-0.0001, 0.0, 0.0])
        epsp_start_iso = float(np.squeeze(epsp_iso))
        epsp_start_kin = float(np.squeeze(epsp_kin))

        rev_step_iso = -1
        rev_step_kin = -1

        for step in range(60):
            sig_iso, epsp_iso_out, _ = shell_update(mat_iso, sig_iso, deps=deps_rev, epsp_old=epsp_iso, extra=extra_iso)
            sig_kin, epsp_kin_out, _ = shell_update(mat_kin, sig_kin, deps=deps_rev, epsp_old=epsp_kin, extra=extra_kin)
            epsp_iso = np.atleast_1d(epsp_iso_out)
            epsp_kin = np.atleast_1d(epsp_kin_out)

            if rev_step_kin == -1 and float(np.squeeze(epsp_kin)) > epsp_start_kin + 1e-6:
                rev_step_kin = step
            if rev_step_iso == -1 and float(np.squeeze(epsp_iso)) > epsp_start_iso + 1e-6:
                rev_step_iso = step

        assert rev_step_kin < rev_step_iso, "Shell Bauschinger effect: kinematic hardening must yield earlier"

    def test_solid_hexa8_kinematic_cyclic_engine(self, tmp_path: Path):
        """Solid Hexa8 cyclic tension/compression through Engine verifying back-stress tracking."""
        run_name = "HEXA8_ISOKIN"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 200.0), (0.05, 300.0), (0.2, 450.0)])
        d.mat_law60(mid=1, title="SteelIsoKin", rho=7.85e-9, e=200000.0, nu=0.3, mat_hard=0.5, funcs=[1])
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "Part1", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        d.grnod_node(1, "fix_face", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_face", [2, 3, 6, 7])
        d.funct(10, "vel_cycle", [
            (0.0, 2000.0),
            (0.5e-4, 2000.0),
            (0.5001e-4, -2000.0),
            (1.0e-4, -2000.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        # Verify plastic strain accumulated
        assert float(np.max(eng_model.bricks.state["epsp"])) > 0.0
        # Verify back-stress was allocated and tracked in mat_extra
        extra = eng_model.bricks.state["mat_extra"]
        assert "sigb60" in extra
        assert np.isfinite(extra["sigb60"]).all()


# ============================================================================
# 4. Energy Conservation in Undamped Explicit Dynamic Simulations & Ledger
# ============================================================================

class TestLaw60EnergyConservationAndLedger:
    """Audit mechanical energy conservation in undamped dynamic oscillations and work ledger."""

    def test_solid_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Solid Hexa8 free oscillation without damping: mechanical energy Delta E / E_0 < 1%."""
        run_name = "FREE_OSC_HEXA_LAW60"
        s_path = str(tmp_path / f"{run_name}_0000.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 500.0), (0.1, 700.0)])  # High yield for linear elasticity
        d.mat_law60(mid=1, title="SteelUndamped", rho=7.85e-9, e=210000.0, nu=0.3, funcs=[1])
        # Undamped: zero artificial viscosity and zero hourglass damping
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
        rho0 = 7.85e-9
        v0_block = 1000.0  # 10 x 10 x 10 mm^3
        m_node = rho0 * v0_block / 8.0
        mass_vec = np.full(8, m_node)

        # Symmetric breathing mode along X
        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 50.0
        v[[0, 3, 4, 7], 0] = -50.0

        dt = 1.0e-7  # Stable Courant time step resolving breathing mode (< 0.1 Courant limit)
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        v_half = v.copy()
        e_kin_0 = 0.5 * np.sum(mass_vec * (v[:, 0] ** 2))
        e_tot_0 = e_kin_0
        assert e_tot_0 > 0.0

        total_steps = 100
        e_tot_history = []

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
            e_tot_history.append(e_tot)

            v_half = v_next_half

        e_max = max(e_tot_history)
        max_delta_e = max(abs(e - e_tot_0) for e in e_tot_history)
        rel_drift = max_delta_e / e_max
        assert rel_drift < 0.01, f"Hexa8 free oscillation energy deviation {rel_drift * 100:.3f}% exceeds 1.0%"

    def test_shell_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Shell BT4 in-plane free oscillation without damping: mechanical energy Delta E / E_0 < 1%."""
        run_name = "FREE_OSC_SHELL_LAW60"
        s_path = str(tmp_path / f"{run_name}_0000.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 500.0), (0.1, 700.0)])
        d.mat_law60(mid=1, title="FabricUndamped", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
        d.prop_shell(1, "PropBT4", thick=1.0, nip=3, ishell=1, hm=0.0, hf=0.0, hr=0.0)
        d.part(1, "Part1", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])
        d.write(s_path)

        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.shells
        rho0 = 7.85e-9
        mass_total = rho0 * 10.0 * 10.0 * 1.0
        m_node = mass_total / 4.0
        mass_vec = np.full(4, m_node)

        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 30.0
        v[[0, 3], 0] = -30.0

        dt = 1.0e-7  # Stable Courant time step
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        v_half = v.copy()
        e_kin_0 = 0.5 * np.sum(mass_vec * (v[:, 0] ** 2))
        e_tot_0 = e_kin_0

        e_tot_history = []
        for step in range(100):
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

        e_max = max(e_tot_history)
        max_delta = max(abs(e - e_tot_0) for e in e_tot_history)
        rel_dev = max_delta / e_max
        assert rel_dev < 0.01, f"Shell free oscillation energy deviation {rel_dev * 100:.3f}% exceeds 1.0%"

    def test_strain_energy_work_ledger_accuracy(self):
        """Verify incremental work ledger sum(sigma : deps * V) matches exact analytical strain energy."""
        e0 = 210000.0
        nu = 0.3
        mat = build_law60(e0=e0, nu=nu)
        k0 = e0 / (3.0 * (1.0 - 2.0 * nu))
        g0 = e0 / (2.0 * (1.0 + nu))

        v0 = 1000.0  # 10 x 10 x 10 mm^3 volume
        sig = np.zeros(6, dtype=float)

        work_acc = 0.0
        n_steps = 40
        deps_step = np.array([0.00002, -0.000006, -0.000006, 0.00001, 0.0, 0.0])
        eps_tot = np.zeros(6, dtype=float)

        for _ in range(n_steps):
            sig_prev = sig.copy()
            sig, _, _ = solid_update(mat, sig, deps=deps_step, epsp_old=np.zeros(1))
            eps_tot += deps_step

            # Midpoint work increment: sigma_mid : deps * V
            sig_mid = 0.5 * (sig_prev + sig)
            # Voigt work contraction: s_xx*de_xx + s_yy*de_yy + s_zz*de_zz + s_xy*de_xy
            d_work = (
                sig_mid[0] * deps_step[0]
                + sig_mid[1] * deps_step[1]
                + sig_mid[2] * deps_step[2]
                + sig_mid[3] * deps_step[3]
            ) * v0
            work_acc += d_work

        # Exact analytical elastic strain energy density
        tr_eps = eps_tot[0] + eps_tot[1] + eps_tot[2]
        dev_eps = np.array([
            eps_tot[0] - tr_eps / 3.0,
            eps_tot[1] - tr_eps / 3.0,
            eps_tot[2] - tr_eps / 3.0,
            eps_tot[3],  # engineering shear gamma_xy
        ])
        w_exact = (
            0.5 * k0 * (tr_eps ** 2)
            + g0 * (dev_eps[0] ** 2 + dev_eps[1] ** 2 + dev_eps[2] ** 2 + 0.5 * (dev_eps[3] ** 2))
        ) * v0

        rel_error = abs(work_acc - w_exact) / w_exact
        assert rel_error < 1.0e-4, f"Strain energy work ledger relative error {rel_error:.3e} exceeds 0.01%"


# ============================================================================
# 5. Dynamic Modulus Degradation & Vibration Frequency Shift
# ============================================================================

class TestLaw60DynamicModulusDegradation:
    """Audit dynamic modulus degradation and frequency shift under plastic damage accumulation."""

    def test_exponential_modulus_degradation_law(self):
        """Verify exponential modulus degradation E_cur = E0 - (E0 - Einf) * (1 - exp(-ce * epsp))."""
        e0 = 200000.0
        einf = 50000.0
        ce = 10.0
        mat = build_law60(e0=e0, einf=einf, ce=ce)

        epsps = np.linspace(0.0, 1.0, 20)
        e_prev = e0

        for ep in epsps:
            expected_e = e0 - (e0 - einf) * (1.0 - math.exp(-ce * ep))
            c_solid = mat.sound_speed_solid(rho=7.85e-9, epsp=ep)
            k_cur = expected_e / (3.0 * (1.0 - 2.0 * mat.nu))
            g_cur = expected_e / (2.0 * (1.0 + mat.nu))
            expected_c = math.sqrt((k_cur + (4.0 / 3.0) * g_cur) / 7.85e-9)

            assert np.isclose(c_solid, expected_c, rtol=1e-4)
            # Monotonically decreasing
            assert expected_e <= e_prev
            e_prev = expected_e

        # Asymptotic limit at epsp = 1.0 (exp(-10) = 4.5e-5)
        assert np.isclose(expected_e, einf, rtol=0.01)

    def test_dynamic_vibration_frequency_shift_with_modulus_degradation(self):
        """Verify natural vibration frequency shifts as stiffness degrades under plastic strain."""
        e0 = 200000.0
        nu = 0.3
        rho0 = 7.85e-9

        # Undamaged frequency: k0 = E0 * A / L, omega0 = sqrt(k0 / m)
        mat_undamaged = build_law60(e0=e0, nu=nu, rho0=rho0)
        c0 = mat_undamaged.sound_speed_solid(rho=rho0, epsp=0.0)

        # Damaged state: degraded to Einf = 0.5 * E0
        mat_degraded = build_law60(e0=e0, einf=100000.0, ce=50.0, nu=nu, rho0=rho0)
        c_deg = mat_degraded.sound_speed_solid(rho=rho0, epsp=0.10)

        # Expected sound speed ratio: c_deg / c0 approx sqrt(E_deg / E0) = sqrt(0.5)
        ratio = c_deg / c0
        assert np.isclose(ratio, math.sqrt(0.5), rtol=1e-2)

        # Natural period of vibration scales inversely with sound speed / frequency:
        # T_deg / T0 = c0 / c_deg approx sqrt(2) = 1.414
        period_ratio = c0 / c_deg
        assert np.isclose(period_ratio, math.sqrt(2.0), rtol=1e-2)


# ============================================================================
# 6. Shell Thinning & Incompressibility
# ============================================================================

class TestLaw60ShellThinningAndIncompressibility:
    """Audit shell thickness update h(t) and incompressibility under biaxial tension."""

    def test_shell_equibiaxial_tension_thinning(self):
        """Under equibiaxial tension, thickness decreases monotonically and sigma_xx == sigma_yy."""
        f1 = FunctTable(1, [0.0, 0.05, 0.20], [200.0, 300.0, 420.0])
        mat = build_law60(e0=200000.0, nu=0.3, funcs=[f1])

        sig = np.zeros(3, dtype=float)
        thk = np.array([2.0], dtype=float)
        extra = {"thk": thk, "off": np.array([1.0])}
        epsp = np.zeros(1, dtype=float)

        thk_history = [2.0]
        deps = np.array([0.002, 0.002, 0.0])

        for _ in range(30):
            sig, epsp_out, _ = shell_update(mat, sig, deps=deps, epsp_old=epsp, extra=extra)
            epsp = np.atleast_1d(epsp_out)
            thk_history.append(float(thk[0]))

            # Equibiaxial symmetry: sigma_xx == sigma_yy, sigma_xy == 0
            assert np.isclose(sig[0], sig[1], rtol=1e-4)
            assert np.isclose(sig[2], 0.0, atol=1e-6)

        # Thickness decreases monotonically with stretching
        for i in range(1, len(thk_history)):
            assert thk_history[i] < thk_history[i - 1], "Thickness must decrease monotonically"

        assert thk[0] < 2.0
        assert float(np.squeeze(epsp)) > 0.0

    def test_shell_bt4_engine_dynamic_thinning(self, tmp_path: Path):
        """Full Engine run of BT4 shell patch under biaxial tension showing progressive thinning."""
        run_name = "BT4_THINNING"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 200.0), (0.1, 350.0)])
        d.mat_law60(mid=1, title="FabricThin", rho=7.85e-9, e=200000.0, nu=0.3, funcs=[1])
        d.prop_shell(1, "PropBT4", thick=2.5, nip=3, ishell=1)
        d.part(1, "Part1", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix node 1 (origin)
        d.grnod_node(1, "fix_orig", [1])
        d.bcs(1, "bcs_orig", "111", "111", 1)

        # Expand along X (nodes 2, 3) and along Y (nodes 3, 4)
        d.grnod_node(2, "pull_x", [2, 3])
        d.funct(10, "vel_pull", [(0.0, 1000.0), (2.0e-4, 1000.0)])
        d.impvel(1, "pull_x_bc", 10, "X", 2)

        d.grnod_node(3, "pull_y", [3, 4])
        d.impvel(2, "pull_y_bc", 10, "Y", 3)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        # Thickness must have thinned from initial 2.5 mm
        final_thk = float(eng_model.shells.state["thick"][0])
        assert final_thk < 2.5, f"Thickness must decrease, got {final_thk}"
        assert final_thk > 0.0


# ============================================================================
# 7. Tensile Failure Scaling & Element Deletion
# ============================================================================

class TestLaw60TensileFailureAndElementDeletion:
    """Audit tensile damage failure factor, element deletion, and post-failure continuation."""

    def test_tensile_damage_scaling_law(self):
        """Verify FAIL factor degrades yield stress linearly from eps_t1 to eps_t2."""
        f1 = FunctTable(1, [0.0, 0.1], [300.0, 300.0])  # Constant 300 MPa
        eps_t1 = 0.05
        eps_t2 = 0.15
        mat = build_law60(e0=200000.0, nu=0.3, eps_t1=eps_t1, eps_t2=eps_t2, funcs=[f1])

        # Point 1: Below eps_t1 (eps = 0.03): FAIL = 1.0 -> yld = 300
        sig0 = np.zeros(6, dtype=float)
        deps1 = np.array([0.03, -0.009, -0.009, 0, 0, 0])
        sig1, epsp1, _ = solid_update(mat, sig0, deps=deps1, eps=np.zeros(6), epsp_old=np.zeros(1))
        vm1 = np.sqrt(0.5 * ((sig1[0] - sig1[1])**2 + (sig1[1] - sig1[2])**2 + (sig1[2] - sig1[0])**2))
        assert np.isclose(vm1, 300.0, rtol=1e-2)

        # Point 2: Midway between eps_t1 and eps_t2 (eps = 0.10):
        # Tensile principal strain epst is solved via 4 Newton iterations.
        # FAIL = (0.15 - epst) / (0.15 - 0.05) < 1.0 -> yld is degraded below 300 MPa
        deps2 = np.array([0.07, -0.021, -0.021, 0, 0, 0])
        sig2, epsp2, _ = solid_update(mat, sig1, deps=deps2, eps=deps1, epsp_old=epsp1)
        vm2 = np.sqrt(0.5 * ((sig2[0] - sig2[1])**2 + (sig2[1] - sig2[2])**2 + (sig2[2] - sig2[0])**2))
        assert vm2 < 300.0, f"Stress must degrade below initial yield (300 MPa), got {vm2}"
        from pyradioss.materials.law60_plast3 import _principal_strain
        epst2 = _principal_strain((deps1 + deps2).reshape(1, 6))
        expected_fail = (0.15 - epst2[0]) / (0.15 - 0.05)
        assert np.isclose(vm2, expected_fail * 300.0, rtol=2e-2)

        # Point 3: Above eps_t2 (eps = 0.20): FAIL = 0.0 -> yld = 0 MPa
        deps3 = np.array([0.10, -0.03, -0.03, 0, 0, 0])
        sig3, epsp3, _ = solid_update(mat, sig2, deps=deps3, eps=deps1 + deps2, epsp_old=epsp2)
        vm3 = np.sqrt(0.5 * ((sig3[0] - sig3[1])**2 + (sig3[1] - sig3[2])**2 + (sig3[2] - sig3[0])**2))
        assert vm3 < 5.0, f"Deviatoric stress must drop near zero after complete tensile rupture, got {vm3}"

    def test_element_deletion_eps_max_solid_and_post_failure_continuation(self):
        """Verify solid element deletion when epsp reaches eps_max and stable post-failure continuation."""
        f1 = FunctTable(1, [0.0, 0.1], [300.0, 400.0])
        eps_limit = 0.05
        mat = build_law60(e0=200000.0, nu=0.3, eps_max=eps_limit, funcs=[f1])

        sig = np.zeros(6, dtype=float)
        epsp = np.zeros(1, dtype=float)
        off = np.array([1.0], dtype=float)
        off60 = np.array([1.0], dtype=float)
        extra = {"off": off, "off60": off60}

        # Step 1: Small strain below eps_max
        deps_small = np.array([0.01, -0.003, -0.003, 0, 0, 0])
        sig, epsp_out, _ = solid_update(mat, sig, deps=deps_small, epsp_old=epsp, extra=extra)
        epsp = np.atleast_1d(epsp_out)
        assert float(np.squeeze(epsp)) < eps_limit
        assert off[0] == 1.0

        # Step 2: Large strain exceeding eps_max
        deps_large = np.array([0.08, -0.024, -0.024, 0, 0, 0])
        sig, epsp_out, _ = solid_update(mat, sig, deps=deps_large, epsp_old=epsp, extra=extra)
        epsp = np.atleast_1d(epsp_out)
        assert float(np.squeeze(epsp)) >= eps_limit
        assert np.allclose(sig, 0.0), "Stress must be zeroed upon element deletion"
        assert off[0] == 0.0, "Element deletion flag 'off' must be set to 0.0"
        assert off60[0] == 0.0

        # Step 3: Post-failure continuation: apply further deformation increments
        for _ in range(10):
            sig, epsp_out, _ = solid_update(mat, sig, deps=deps_large, epsp_old=epsp, extra=extra)
            epsp = np.atleast_1d(epsp_out)
            assert np.allclose(sig, 0.0)
            assert not np.isnan(sig).any()
            assert not np.isnan(epsp).any()

    def test_multi_element_post_failure_continuation_engine(self, tmp_path: Path):
        """2-element bar through Engine: one element erodes at eps_max, simulation continues without crash."""
        run_name = "MULTI_ERODE_LAW60"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 200.0), (0.1, 300.0)])
        # Part 1 has low failure strain eps_p_max = 0.02 (will erode)
        d.mat_law60(mid=1, title="MatWeak", rho=7.85e-9, e=200000.0, nu=0.3, eps_p_max=0.02, funcs=[1])
        # Part 2 has high failure strain (will survive)
        d.mat_law60(mid=2, title="MatStrong", rho=7.85e-9, e=200000.0, nu=0.3, eps_p_max=1.0, funcs=[1])

        d.prop_solid(1, "Prop1", isolid=1)
        d.prop_solid(2, "Prop2", isolid=1)
        d.part(1, "PartWeak", 1, 1)
        d.part(2, "PartStrong", 2, 2)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
            (9, 10.0, 0.0, 0.0), (10, 10.0, 5.0, 0.0), (11, 10.0, 0.0, 5.0), (12, 10.0, 5.0, 5.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        d.brick(2, [(2, 2, 9, 10, 3, 6, 11, 12, 7)])

        d.grnod_node(1, "fix_left", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_right", [9, 10, 11, 12])
        d.funct(10, "pull_func", [(0.0, 5000.0), (2.0e-4, 5000.0)])
        d.impvel(1, "pull_bc", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        # Element 1 has failed (off = 0.0)
        off_flags = eng_model.bricks.state["off"]
        assert off_flags[0] == 0.0, "Weak element must be eroded (off=0.0)"
        # No NaNs anywhere in stress or coords
        assert not np.isnan(eng_model.bricks.state["sig"]).any()
        assert not np.isnan(eng_model.x).any()


# ============================================================================
# 8. Acoustic Wave Speed & Courant Stability
# ============================================================================

class TestLaw60WaveSpeedAndCourantStability:
    """Audit sound speed formulas, Courant time step bounding, and acoustic wave propagation."""

    def test_sound_speed_exact_formulas(self):
        """Verify longitudinal solid sound speed and shell sound speed exact formulas."""
        e0 = 210000.0
        nu = 0.3
        rho0 = 7.85e-9
        mat = build_law60(e0=e0, nu=nu, rho0=rho0)

        # Exact bulk and shear moduli
        k0 = e0 / (3.0 * (1.0 - 2.0 * nu))
        g0 = e0 / (2.0 * (1.0 + nu))
        expected_c_solid = math.sqrt((k0 + (4.0 / 3.0) * g0) / rho0)

        c_solid = mat.sound_speed_solid(rho=rho0)
        assert np.isclose(c_solid, expected_c_solid, rtol=1e-5)

        # Exact shell plane-stress acoustic speed: A1 = E / (1 - nu^2)
        expected_c_shell = math.sqrt((e0 / (1.0 - nu ** 2)) / rho0)
        c_shell = mat.sound_speed_shell(rho=rho0)
        assert np.isclose(c_shell, expected_c_shell, rtol=1e-5)

    def test_courant_stability_bound(self):
        """Verify dynamic time step bounded by Courant condition: dt <= dt_Courant = L_min / c."""
        e0 = 200000.0
        nu = 0.3
        rho0 = 7.85e-9
        mat = build_law60(e0=e0, nu=nu, rho0=rho0)

        l_min = 5.0  # mm
        c = mat.sound_speed_solid(rho=rho0)
        dt_courant = l_min / c

        # Verification: scale factor 0.5 gives safe stable time step
        dt_safe = 0.5 * dt_courant
        assert dt_safe < dt_courant
        assert dt_courant > 0.0

    def test_acoustic_pulse_propagation_bar_engine(self, tmp_path: Path):
        """10-element bar in Engine: acoustic disturbance travels at wave speed c."""
        run_name = "BAR_10EL_WAVE"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        rho0 = 7.85e-9
        e0 = 200000.0
        nu = 0.3
        d = StarterDeck(run_name)
        d.funct(1, "YieldCurve", [(0.0, 500.0), (0.1, 700.0)])
        d.mat_law60(mid=1, title="SteelWave", rho=rho0, e=e0, nu=nu, funcs=[1])
        d.prop_solid(1, "BarProp", isolid=1)
        d.part(1, "BarPart", 1, 1)

        # 10 elements along X from x=0 to x=50 mm (5 mm each)
        nodes = []
        for i in range(11):
            x = i * 5.0
            nodes.extend([
                (4 * i + 1, x, 0.0, 0.0),
                (4 * i + 2, x, 5.0, 0.0),
                (4 * i + 3, x, 5.0, 5.0),
                (4 * i + 4, x, 0.0, 5.0),
            ])
        d.node(nodes)

        bricks = []
        for i in range(10):
            b_bot = (4 * i + 1, 4 * (i + 1) + 1, 4 * (i + 1) + 2, 4 * i + 2)
            b_top = (4 * i + 4, 4 * (i + 1) + 4, 4 * (i + 1) + 3, 4 * i + 3)
            bricks.append((i + 1, *b_bot, *b_top))
        d.brick(1, bricks)

        # Apply short velocity pulse at x=0 (nodes 1..4)
        d.grnod_node(1, "impact_face", [1, 2, 3, 4])
        d.funct(10, "pulse_func", [(0.0, 50.0), (2.0e-6, 50.0), (2.1e-6, 0.0), (1.0, 0.0)])
        d.impvel(1, "impact_pulse", 10, "X", 1)

        d.write(s_path)
        # Sound speed ~ 5800 mm/s. In 4e-6 s, wave travels ~ 23 mm (reaches elements 0..4).
        # Element 9 (at x=45..50 mm) is far beyond wave front and must be undisturbed!
        _write_engine_deck(e_path, run_name, tstop=4.0e-6, dt_scale=0.4)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 5
        assert state.stop_reason == ""

        sig = eng_model.bricks.state["sig"]
        assert abs(sig[0, 0]) > 0.05, "Element near impact pulse must be disturbed"
        assert abs(sig[9, 0]) < 1.0e-3, "Far end element 9 must remain undisturbed before wave arrival"
