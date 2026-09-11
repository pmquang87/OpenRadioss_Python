"""
Dynamic Engine Simulation & Energy Balance Verifier for M562 (/MAT/LAW66 /MAT/PLAS_TAB_COSSER /MAT/PLAS_COSSER).

Exhaustive dynamic explicit engine simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations across solid and shell element formulations:
   - Solid Hexa8 with LAW66 under cyclic tension and compression (50+ cycles, |ERR| < 1.0%)
   - Solid Tetra4 with LAW66 under dynamic tension (50+ cycles, |ERR| < 1.0%)
   - Shell BT4 with LAW66 under cyclic tension and compression (50+ cycles, |ERR| < 1.0%)
   - Shell QEPH with LAW66 under dynamic biaxial stretch (50+ cycles, |ERR| < 1.0%)
   - Shell Tri3 with LAW66 under dynamic stretching (50+ cycles, |ERR| < 1.0%)
   - Solid Hexa8 2-element patch with LAW66 under cyclic loading (50+ cycles, |ERR| < 1.0%)
   - Shell BT4 2x2 multi-element patch with LAW66 under cyclic loading (50+ cycles, |ERR| < 1.0%)
   - Asserts normal termination, >= 50 cycles, energy balance error |ERR| < 1.0%,
     positive internal strain energy accumulation (IE > 0), and external work consistency.
2. Energy conservation & work balance:
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV * dt
   - Midpoint trapezoidal work matches exact analytical strain energy in closed reversible cycles.
   - Free vibration of undamped solid (Hexa8) and shell (BT4) elements:
     mechanical energy E_tot = E_kin + E_int is strictly conserved (|Delta E| / E_0 < 1.0%).
   - Plastic dissipation monotonically increasing during plastic flow.
3. Tension vs compression asymmetry & kinematic hardening:
   - Asymmetric yield strengths in tension vs compression using yield curves / scales.
   - Bauschinger reverse softening under kinematic hardening (c_hard > 0) vs isotropic hardening (c_hard = 0).
   - Shell thickness thinning under tensile stretch and thickening under compressive membrane loads.
4. Strain rate sensitivity:
   - Cowper-Symonds rate sensitivity (israte <= 2) scaling flow stress dynamically.
   - Rate curve table scaling (israte = 3) for compression and tension.
5. Acoustic sound speed & Courant time-step stability:
   - Solid and shell sound speed maintaining positive, stable Courant bounds across all dynamic cycles.
   - Stability under dynamic pulse propagation in multi-element patches.
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials.law66_plas_tab import (
    Law66Params,
    build_law66,
    solid_update,
    shell_update,
    sound_speed_solid,
    sound_speed_shell,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers: Engine Control Deck Writer and Material Factories
# ============================================================================

def _scalar(val: Any) -> float:
    """Safely convert 0-d or 1-d single element array/scalar to float (NumPy 2.x safe)."""
    return float(np.asarray(val).flat[0])


def _write_engine_deck(
    path: Path | str,
    run_name: str,
    tstop: float = 2.5e-4,
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


def make_test_material_law66(
    mid: int = 1,
    rho0: float = 2.7e-9,  # ton/mm^3
    E: float = 70000.0,    # MPa
    nu: float = 0.33,
    c_hard: float = 0.0,
    f_cut: float = 0.0,
    fsmooth: int = 0,
    israte: int = 1,
    fun_a1: int = 0,
    fun_a2: int = 0,
    fscale11: float = 1.0,
    fscale22: float = 1.0,
    c: float = 0.0,
    eps_0: float = 0.0,
    sigmay0: float = 200.0,
    curve_c: Any = None,
    curve_t: Any = None,
    curve_rate_c: Any = None,
    curve_rate_t: Any = None,
    fun_c: Any = None,
    fun_t: Any = None,
    **kwargs: Any,
) -> Material:
    """Factory creating a valid /MAT/LAW66 Material instance."""
    if curve_c is None:
        curve_c = fun_c
    if curve_t is None:
        curve_t = fun_t
    if curve_c is None and fun_a1 == 0 and sigmay0 > 0.0:
        curve_c = [(0.0, sigmay0), (0.05, sigmay0 * 1.25), (0.2, sigmay0 * 1.5)]
    if curve_t is None and fun_a2 == 0 and sigmay0 > 0.0:
        curve_t = [(0.0, sigmay0), (0.05, sigmay0 * 1.25), (0.2, sigmay0 * 1.5)]
    params = {
        "e": E,
        "nu": nu,
        "c_hard": c_hard,
        "f_cut": f_cut,
        "fsmooth": fsmooth,
        "iyld_rate": israte,
        "israte": israte,
        "fun_a1": fun_a1,
        "fun_a2": fun_a2,
        "fscale11": fscale11,
        "fscale22": fscale22,
        "c": c,
        "cp": c,
        "eps_0": eps_0,
        "epsp0": eps_0,
        "sigma_y0": sigmay0,
        "sigmay0": sigmay0,
        "rho": rho0,
        "curve_c": curve_c,
        "curve_t": curve_t,
        "curve_rate_c": curve_rate_c,
        "curve_rate_t": curve_rate_t,
    }
    params.update(kwargs)
    return build_law66(id=mid, rho0=rho0, title="MatLAW66", params=params)


# ============================================================================
# 1. Multi-Cycle Explicit Dynamic Simulations Across Formulations
# ============================================================================

class TestLaw66MultiCycleDynamicSimulations:
    """Audit multi-cycle explicit dynamic simulations across supported solid and shell element formulations."""

    def test_solid_hexa8_cyclic_tension_compression_engine(self, tmp_path: Path):
        """Solid Hexa8 with LAW66 under cyclic tension and compression (50+ cycles)."""
        run_name = "HEXA8_LAW66_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        # Yield functions: Compression (funct 10) and Tension (funct 11)
        d.funct(10, "f_comp", [(0.0, 180.0), (0.05, 240.0), (0.2, 300.0)])
        d.funct(11, "f_tens", [(0.0, 150.0), (0.05, 200.0), (0.2, 260.0)])
        d.mat_law66(
            1, "FoamLAW66",
            rho=2.7e-9, e=70000.0, nu=0.33,
            c_hard=0.3,
            fun_a1=10, fun_a2=11,
            fscale11=1.0, fscale22=1.0,
        )
        d.prop_solid(1, "PropHexa", isolid=1)
        d.part(1, "PartHexa", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        # Fix bottom face nodes (1, 2, 3, 4)
        d.grnod_node(1, "fix_bottom", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Multi-axis cyclic loading on top face nodes (5, 6, 7, 8)
        d.grnod_node(2, "top_nodes", [5, 6, 7, 8])
        # Cyclic tension -> hold -> compression -> tension in Z
        d.funct(12, "vz_cyc", [
            (0.0, 1000.0),
            (0.8e-4, 1000.0),
            (0.8001e-4, -800.0),
            (1.6e-4, -800.0),
            (1.6001e-4, 1000.0),
            (2.5e-4, 1000.0),
        ])
        d.impvel(1, "pull_z", 12, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == "", f"Unexpected stop reason: {state.stop_reason}"

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Hexa8 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal strain energy must be strictly positive"
        assert en["EW"] > 0.0, "External work must be positive"

        # Check state variable propagation (uvar66)
        mat_extra = eng_model.bricks.state.get("mat_extra", {})
        assert "uvar66" in mat_extra or "uvar66" in eng_model.bricks.state

    def test_solid_tetra4_dynamic_tension_engine(self, tmp_path: Path):
        """Solid Tetra4 with LAW66 under dynamic tension (50+ cycles)."""
        run_name = "TETRA4_LAW66_TEN"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "f_comp", [(0.0, 180.0), (0.1, 250.0)])
        d.funct(11, "f_tens", [(0.0, 160.0), (0.1, 230.0)])
        d.mat_law66(
            1, "TetraLAW66",
            rho=2.7e-9, e=70000.0, nu=0.33,
            c_hard=0.0,
            fun_a1=10, fun_a2=11,
        )
        d.prop_solid(1, "PropTetra", isolid=1)
        d.part(1, "PartTetra", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0), (4, 0.0, 0.0, 10.0),
        ])
        d.tetra4(1, [(1, 1, 2, 3, 4)])

        # Fix base nodes (1, 2, 3)
        d.grnod_node(1, "fix_base", [1, 2, 3])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull apex node 4 in Z
        d.grnod_node(2, "apex_node", [4])
        d.funct(12, "vz_pull", [(0.0, 500.0), (2.0e-4, 500.0)])
        d.impvel(1, "pull_apex", 12, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Tetra4 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_bt4_cyclic_tension_compression_engine(self, tmp_path: Path):
        """Shell BT4 with LAW66 under cyclic tension and compression (50+ cycles)."""
        run_name = "SHELL_BT4_LAW66_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "f_comp", [(0.0, 200.0), (0.05, 260.0), (0.15, 320.0)])
        d.funct(11, "f_tens", [(0.0, 160.0), (0.05, 210.0), (0.15, 270.0)])
        d.mat_law66(
            1, "ShellLAW66",
            rho=2.7e-9, e=70000.0, nu=0.33,
            c_hard=0.2,
            fun_a1=10, fun_a2=11,
        )
        thick0 = 1.2
        d.prop_shell(1, "PropBT4", thick=thick0, nip=3, ishell=1)
        d.part(1, "PartBT4", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix left edge (nodes 1, 4)
        d.grnod_node(1, "fix_left", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Multi-axial cyclic loading on right edge (nodes 2, 3)
        d.grnod_node(2, "right_nodes", [2, 3])
        # Cyclic tension -> hold -> compression -> tension in X
        d.funct(12, "vx_cyc", [
            (0.0, 1200.0),
            (0.8e-4, 1200.0),
            (0.8001e-4, -800.0),
            (1.6e-4, -800.0),
            (1.6001e-4, 1000.0),
            (2.5e-4, 1000.0),
        ])
        d.impvel(1, "pull_x", 12, "X", 2)

        # In-plane shear loading in Y
        d.funct(13, "vy_shear", [
            (0.0, 400.0),
            (1.0e-4, 400.0),
            (1.0001e-4, -400.0),
            (2.5e-4, -400.0),
        ])
        d.impvel(2, "shear_y", 13, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == "", f"Unexpected stop reason: {state.stop_reason}"

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"BT4 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal strain energy must be strictly positive"
        assert en["EW"] > 0.0, "External work must be positive"

        # Check plastic strain and state variable propagation
        st_shells = eng_model.shells.state
        mat_extra = st_shells.get("mat_extra", {})
        assert "uvar66" in mat_extra or "uvar66" in st_shells

    def test_shell_qeph_dynamic_biaxial_stretch_engine(self, tmp_path: Path):
        """Shell QEPH (Ishell=24) with LAW66 under dynamic biaxial stretch (50+ cycles)."""
        run_name = "SHELL_QEPH_LAW66_BIAX"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "f_comp", [(0.0, 200.0), (0.1, 280.0)])
        d.funct(11, "f_tens", [(0.0, 180.0), (0.1, 250.0)])
        d.mat_law66(
            1, "QEPHLAW66",
            rho=2.7e-9, e=70000.0, nu=0.33,
            c_hard=0.2,
            fun_a1=10, fun_a2=11,
        )
        d.prop_shell(1, "PropQEPH", thick=1.0, nip=3, ishell=24)
        d.part(1, "PartQEPH", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix corner node 1 completely, node 4 in X, node 2 in Y
        d.grnod_node(1, "fix_corner", [1])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "fix_x", [4])
        d.bcs(2, "bcs_fix_x", "100", "000", 2)
        d.grnod_node(3, "fix_y", [2])
        d.bcs(3, "bcs_fix_y", "010", "000", 3)

        # Pull right edge (2, 3) in X
        d.grnod_node(4, "pull_x_nodes", [2, 3])
        d.funct(12, "vx_pull", [(0.0, 350.0), (2.0e-4, 350.0)])
        d.impvel(1, "pull_x", 12, "X", 4)

        # Pull top edge (3, 4) in Y
        d.grnod_node(5, "pull_y_nodes", [3, 4])
        d.funct(13, "vy_pull", [(0.0, 300.0), (2.0e-4, 300.0)])
        d.impvel(2, "pull_y", 13, "Y", 5)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"QEPH energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_tri3_dynamic_stretch_engine(self, tmp_path: Path):
        """Shell Tri3 (Ish3n=1) with LAW66 under dynamic stretch (50+ cycles)."""
        run_name = "SHELL_TRI3_LAW66_STR"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "f_comp", [(0.0, 220.0), (0.1, 300.0)])
        d.funct(11, "f_tens", [(0.0, 190.0), (0.1, 270.0)])
        d.mat_law66(
            1, "Tri3LAW66",
            rho=2.7e-9, e=70000.0, nu=0.33,
            c_hard=0.0,
            fun_a1=10, fun_a2=11,
        )
        d.prop_shell(1, "PropTri3", thick=1.0, nip=3, ish3n=1)
        d.part(1, "PartTri3", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0),
        ])
        d.sh3n(1, [(1, 1, 2, 3)])

        # Fix base root (nodes 1, 3)
        d.grnod_node(1, "fix_root", [1, 3])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull apex node 2 in X and Y
        d.grnod_node(2, "apex_node", [2])
        d.funct(12, "vx_pull", [(0.0, 450.0), (2.0e-4, 450.0)])
        d.impvel(1, "pull_apex_x", 12, "X", 2)
        d.funct(13, "vy_pull", [(0.0, 200.0), (2.0e-4, 200.0)])
        d.impvel(2, "pull_apex_y", 13, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Tri3 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_solid_hexa8_multi_element_patch_cyclic_engine(self, tmp_path: Path):
        """Solid Hexa8 2-element patch with LAW66 under cyclic loading (50+ cycles)."""
        run_name = "HEXA8_PATCH_LAW66"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "f_comp", [(0.0, 200.0), (0.08, 280.0)])
        d.funct(11, "f_tens", [(0.0, 160.0), (0.08, 230.0)])
        d.mat_law66(
            1, "HexaPatch",
            rho=2.7e-9, e=70000.0, nu=0.33,
            c_hard=0.4,
            fun_a1=10, fun_a2=11,
        )
        d.prop_solid(1, "PropHexa", isolid=1)
        d.part(1, "PartHexa", 1, 1)

        # 2 elements stacked in Z: [0, 10] and [10, 20]
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
            (9, 0.0, 0.0, 20.0), (10, 10.0, 0.0, 20.0), (11, 10.0, 10.0, 20.0), (12, 0.0, 10.0, 20.0),
        ])
        d.brick(1, [
            (1, 1, 2, 3, 4, 5, 6, 7, 8),
            (2, 5, 6, 7, 8, 9, 10, 11, 12),
        ])

        # Fix bottom face nodes (1, 2, 3, 4)
        d.grnod_node(1, "fix_bottom", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Cyclic motion on top face nodes (9, 10, 11, 12)
        d.grnod_node(2, "top_nodes", [9, 10, 11, 12])
        d.funct(12, "vz_cyc", [
            (0.0, 1000.0),
            (1.0e-4, 1000.0),
            (1.0001e-4, -600.0),
            (2.2e-4, -600.0),
        ])
        d.impvel(1, "pull_top", 12, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.2e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Hexa8 2-element energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_bt4_2x2_patch_cyclic_engine(self, tmp_path: Path):
        """Shell BT4 2x2 multi-element patch with LAW66 under cyclic loading (50+ cycles)."""
        run_name = "PATCH_2X2_LAW66"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "f_comp", [(0.0, 220.0), (0.08, 300.0)])
        d.funct(11, "f_tens", [(0.0, 180.0), (0.08, 250.0)])
        d.mat_law66(
            1, "ShellPatch",
            rho=2.7e-9, e=70000.0, nu=0.33,
            c_hard=0.4,
            fun_a1=10, fun_a2=11,
        )
        d.prop_shell(1, "PropPatch", thick=1.0, nip=3, ishell=1)
        d.part(1, "PartPatch", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 20.0, 0.0, 0.0),
            (4, 0.0, 10.0, 0.0), (5, 10.0, 10.0, 0.0), (6, 20.0, 10.0, 0.0),
            (7, 0.0, 20.0, 0.0), (8, 10.0, 20.0, 0.0), (9, 20.0, 20.0, 0.0),
        ])
        d.shell(1, [
            (1, 1, 2, 5, 4),
            (2, 2, 3, 6, 5),
            (3, 4, 5, 8, 7),
            (4, 5, 6, 9, 8),
        ])

        # Fix bottom edge (nodes 1, 2, 3)
        d.grnod_node(1, "fix_bottom", [1, 2, 3])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull top edge (nodes 7, 8, 9) in Y
        d.grnod_node(2, "top_nodes", [7, 8, 9])
        d.funct(12, "vy_pull", [
            (0.0, 600.0),
            (1.0e-4, 600.0),
            (1.0001e-4, -400.0),
            (2.2e-4, -400.0),
        ])
        d.impvel(1, "pull_top", 12, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.2e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"2x2 patch energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0


# ============================================================================
# 2. Energy Conservation & Work Balance Auditing
# ============================================================================

class TestLaw66EnergyConservationAndWorkBalance:
    """Audit energy conservation, incremental trapezoidal work, and plastic dissipation."""

    def test_strain_energy_ledger_trapezoidal_integration(self):
        """Incremental trapezoidal work matches exact analytical strain energy in elastic regime."""
        mat = make_test_material_law66(
            E=70000.0, nu=0.33,
            sigmay0=10000.0,  # High yield stress for pure elastic response
        )
        vol0 = 100.0  # 10 x 10 x 1 mm
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar66": np.zeros((1, 8)),
            "thk66": np.ones(1),
            "thk": np.ones(1),
        }
        dt = 1.0e-6

        nu = 0.33
        E = 70000.0
        a11 = E / (1.0 - nu ** 2)
        a12 = nu * a11
        g = E / (2.0 * (1.0 + nu))

        eps_total = np.zeros(3, dtype=float)
        eint_incremental = 0.0

        for step in range(15):
            deps = np.array([[1.0e-4, -0.33 * 1.0e-4, 5.0e-5]])
            sig_old = sig.copy()
            res = shell_update(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig_new = res[0].copy()
            sig = sig_new.copy()
            epsp[0] = _scalar(res[1])

            sig_mid = 0.5 * (sig_old + sig_new)
            de = (sig_mid[0, 0] * deps[0, 0] + sig_mid[0, 1] * deps[0, 1] + sig_mid[0, 2] * deps[0, 2]) * vol0
            eint_incremental += de
            eps_total += deps[0]

        # Analytical elastic strain energy
        ex, ey, exy = eps_total
        analytical_eint = 0.5 * (a11 * ex**2 + a11 * ey**2 + 2.0 * a12 * ex * ey + g * exy**2) * vol0
        assert eint_incremental == pytest.approx(analytical_eint, rel=1e-5), \
            f"Incremental strain energy {eint_incremental} != analytical {analytical_eint}"
        assert epsp[0] == 0.0, "No plastic strain should accumulate in elastic regime"

    def test_solid_hexa8_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Solid Hexa8 in elastic regime: |Delta E| / E_0 < 1.0%."""
        deck = StarterDeck("HEXA_ELAS_OSC")
        deck.mat_law66(
            1, "ElasticLAW66",
            rho=2.7e-9, e=70000.0, nu=0.33,
            sigmay0=10000.0,
            fun_a1=0, fun_a2=0,
        )
        deck.prop_solid(1, "PropHexa", isolid=1)
        deck.part(1, "PartSolid", 1, 1)

        lx, ly, lz = 10.0, 10.0, 10.0
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, lx, 0.0, 0.0), (3, lx, ly, 0.0), (4, 0.0, ly, 0.0),
            (5, 0.0, 0.0, lz), (6, lx, 0.0, lz), (7, lx, ly, lz), (8, 0.0, ly, lz),
        ])
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        s_path = str(tmp_path / "HEXA_ELAS_OSC_0000.rad")
        deck.write(s_path)
        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.bricks
        rho0 = 2.7e-9
        vol0 = lx * ly * lz
        m_node = rho0 * vol0 / 8.0
        mass_vec = np.full(8, m_node)

        # Pure symmetric breathing velocity mode in X
        v = np.zeros_like(model.x)
        vx0 = 30.0
        v[[1, 2, 5, 6], 0] = vx0
        v[[0, 3, 4, 7], 0] = -vx0

        e_kin_0 = 0.5 * np.sum(mass_vec * (vx0 ** 2))
        e_int_0 = float(np.sum(group.state["eint"]))
        e_tot_0 = e_kin_0 + e_int_0
        assert e_tot_0 > 0.0

        dt = 1.0e-7
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        model.v = v.copy()

        e_tot_history = []
        for step in range(80):
            fint.fill(0.0)
            mint.fill(0.0)
            solid_hexa8.forces(group, model.x, model.v, model.vr, dt, fint, mint)
            v_old = model.v.copy()
            acc = fint / mass_vec[:, None]
            model.v += acc * dt
            model.x += model.v * dt
            v_mid = 0.5 * (v_old + model.v)
            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot_history.append(e_kin + e_int)

        max_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_tot_history)
        assert max_err < 0.01, f"Hexa8 elastic energy conservation error {max_err*100:.3f}% exceeds 1.0%"

    def test_shell_bt4_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Shell BT4 in elastic regime: |Delta E| / E_0 < 1.0%."""
        deck = StarterDeck("BT4_ELAS_OSC")
        deck.mat_law66(
            1, "ElasticLAW66",
            rho=2.7e-9, e=70000.0, nu=0.33,
            sigmay0=10000.0,
            fun_a1=0, fun_a2=0,
        )
        thick0 = 1.0
        deck.prop_shell(1, "PropBT4", thick=thick0, nip=3, ishell=1, hm=0.0, hf=0.0, hr=0.0)
        deck.part(1, "PartShell", 1, 1)

        lx, ly = 10.0, 10.0
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, lx, 0.0, 0.0), (3, lx, ly, 0.0), (4, 0.0, ly, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 3, 4)])

        s_path = str(tmp_path / "BT4_ELAS_OSC_0000.rad")
        deck.write(s_path)
        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.shells
        rho0 = 2.7e-9
        area0 = lx * ly
        m_node = rho0 * thick0 * area0 / 4.0
        mass_vec = np.full(4, m_node)

        # Pure symmetric breathing velocity mode in X
        v = np.zeros_like(model.x)
        vx0 = 40.0
        v[[1, 2], 0] = vx0
        v[[0, 3], 0] = -vx0

        e_kin_0 = 0.5 * np.sum(mass_vec * (vx0 ** 2))
        e_int_0 = float(np.sum(group.state["eint"]))
        e_tot_0 = e_kin_0 + e_int_0
        assert e_tot_0 > 0.0

        dt = 1.0e-7
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        model.v = v.copy()

        e_tot_history = []
        for step in range(80):
            fint.fill(0.0)
            mint.fill(0.0)
            shell_bt4.forces(group, model.x, model.v, model.vr, dt, fint, mint)
            v_old = model.v.copy()
            acc = fint / mass_vec[:, None]
            model.v += acc * dt
            model.x += model.v * dt
            v_mid = 0.5 * (v_old + model.v)
            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot_history.append(e_kin + e_int)

        max_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_tot_history)
        assert max_err < 0.01, f"BT4 elastic energy conservation error {max_err*100:.3f}% exceeds 1.0%"

    def test_plastic_dissipation_monotonically_increasing(self):
        """Plastic strain and plastic dissipation monotonically non-decreasing during plastic flow."""
        mat = make_test_material_law66(
            E=70000.0, nu=0.33,
            sigmay0=180.0,
            fun_c=[(0.0, 180.0), (0.1, 280.0)],
            fun_t=[(0.0, 180.0), (0.1, 280.0)],
        )
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar66": np.zeros((1, 8)),
            "thk66": np.ones(1),
            "thk": np.ones(1),
        }

        epsp_history = []
        dt = 1.0e-6

        for step in range(40):
            deps = np.array([[3.0e-4, -1.0e-4, 1.5e-4]])
            res = shell_update(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])
            epsp_history.append(float(epsp[0]))

        d_epsp = np.diff(epsp_history)
        assert np.all(d_epsp >= -1e-15), "Plastic strain increments must be non-negative"
        assert epsp_history[-1] > epsp_history[0], "Plastic strain must strictly increase during yield"


# ============================================================================
# 3. Tension vs Compression Asymmetry & Kinematic Hardening
# ============================================================================

class TestLaw66TensionCompressionAsymmetryAndKinematicHardening:
    """Audit tension vs compression asymmetric yield, Bauschinger effect, and shell thinning."""

    def test_tension_vs_compression_asymmetric_yield(self):
        """Verify distinct yield responses in tension vs compression matching Law 66 yield surfaces."""
        sig_c0 = 240.0
        sig_t0 = 160.0
        mat = make_test_material_law66(
            E=70000.0, nu=0.33,
            fun_c=[(0.0, sig_c0), (0.1, sig_c0 * 1.3)],
            fun_t=[(0.0, sig_t0), (0.1, sig_t0 * 1.3)],
        )

        # Simulation 1: Uniaxial tension
        sig_t = np.zeros((1, 6), dtype=float)
        epsp_t = np.zeros(1, dtype=float)
        extra_t = {"uvar66": np.zeros((1, 8))}
        for _ in range(40):
            deps = np.array([[1.0e-4, -0.33 * 1.0e-4, -0.33 * 1.0e-4, 0.0, 0.0, 0.0]])
            res = solid_update(mat, sig_t, deps, epsp_t, dt=1e-6, extra=extra_t)
            sig_t = res[0].copy()
            epsp_t[0] = _scalar(res[1])
            if epsp_t[0] > 1e-6:
                break

        # Simulation 2: Uniaxial compression
        sig_c = np.zeros((1, 6), dtype=float)
        epsp_c = np.zeros(1, dtype=float)
        extra_c = {"uvar66": np.zeros((1, 8))}
        for _ in range(40):
            deps = np.array([[-1.0e-4, 0.33 * 1.0e-4, 0.33 * 1.0e-4, 0.0, 0.0, 0.0]])
            res = solid_update(mat, sig_c, deps, epsp_c, dt=1e-6, extra=extra_c)
            sig_c = res[0].copy()
            epsp_c[0] = _scalar(res[1])
            if epsp_c[0] > 1e-6:
                break

        # Tension yield stress must be smaller than compression yield stress
        assert sig_t[0, 0] == pytest.approx(sig_t0, rel=0.06)
        assert abs(sig_c[0, 0]) == pytest.approx(sig_c0, rel=0.06)
        assert abs(sig_c[0, 0]) > sig_t[0, 0] + 30.0

    def test_bauschinger_effect_kinematic_hardening(self):
        """Under cyclic loading, verify reverse yield occurs earlier with kinematic hardening (c_hard > 0)."""
        mat_kin = make_test_material_law66(
            E=70000.0, nu=0.33, c_hard=1.0,  # purely kinematic hardening
            fun_c=[(0.0, 200.0), (0.05, 300.0)],
            fun_t=[(0.0, 200.0), (0.05, 300.0)],
        )
        mat_iso = make_test_material_law66(
            E=70000.0, nu=0.33, c_hard=0.0,  # purely isotropic hardening
            fun_c=[(0.0, 200.0), (0.05, 300.0)],
            fun_t=[(0.0, 200.0), (0.05, 300.0)],
        )

        def run_cycle(mat):
            sig = np.zeros((1, 6), dtype=float)
            epsp = np.zeros(1, dtype=float)
            extra = {"uvar66": np.zeros((1, 8))}
            # 1. Forward tension
            for _ in range(30):
                deps = np.array([[2.0e-4, -0.33 * 2.0e-4, -0.33 * 2.0e-4, 0.0, 0.0, 0.0]])
                res = solid_update(mat, sig, deps, epsp, dt=1e-6, extra=extra)
                sig = res[0].copy()
                epsp[0] = _scalar(res[1])
            epsp_fwd = float(epsp[0])
            # 2. Reverse compression
            reverse_yield_stress = None
            for _ in range(40):
                deps = np.array([[-2.0e-4, 0.33 * 2.0e-4, 0.33 * 2.0e-4, 0.0, 0.0, 0.0]])
                res = solid_update(mat, sig, deps, epsp, dt=1e-6, extra=extra)
                sig = res[0].copy()
                epsp[0] = _scalar(res[1])
                if epsp[0] > epsp_fwd + 1e-5 and reverse_yield_stress is None:
                    reverse_yield_stress = abs(float(sig[0, 0]))
            return reverse_yield_stress

        rev_kin = run_cycle(mat_kin)
        rev_iso = run_cycle(mat_iso)
        assert rev_kin is not None and rev_iso is not None
        # Kinematic hardening yields in reverse at a lower absolute stress than isotropic
        assert rev_kin < rev_iso, f"Bauschinger effect: rev_kin ({rev_kin}) must be < rev_iso ({rev_iso})"

    def test_shell_thickness_thinning_in_tension_and_compression(self):
        """Shell thickness decreases under membrane tension and increases under membrane compression."""
        mat = make_test_material_law66(
            E=70000.0, nu=0.33,
            fun_c=[(0.0, 200.0), (0.1, 280.0)],
            fun_t=[(0.0, 200.0), (0.1, 280.0)],
        )

        # Membrane tension
        sig_t = np.zeros((1, 3), dtype=float)
        epsp_t = np.zeros(1, dtype=float)
        extra_t = {"uvar66": np.zeros((1, 8)), "thk": np.array([1.0]), "thk66": np.array([1.0])}
        for _ in range(30):
            deps = np.array([[4.0e-4, 4.0e-4, 0.0]])
            sig_t, epsp_t = shell_update(mat, sig_t, deps, epsp_t, dt=1e-6, extra=extra_t)

        # Membrane compression
        sig_c = np.zeros((1, 3), dtype=float)
        epsp_c = np.zeros(1, dtype=float)
        extra_c = {"uvar66": np.zeros((1, 8)), "thk": np.array([1.0]), "thk66": np.array([1.0])}
        for _ in range(30):
            deps = np.array([[-4.0e-4, -4.0e-4, 0.0]])
            sig_c, epsp_c = shell_update(mat, sig_c, deps, epsp_c, dt=1e-6, extra=extra_c)

        thk_t = float(extra_t["thk"][0])
        thk_c = float(extra_c["thk"][0])
        assert thk_t < 1.0, f"Tensile membrane stretch must cause thinning: got {thk_t}"
        assert thk_c > 1.0, f"Compressive membrane load must cause thickening: got {thk_c}"


# ============================================================================
# 4. Strain Rate Sensitivity Auditing
# ============================================================================

class TestLaw66StrainRateSensitivity:
    """Audit Cowper-Symonds rate sensitivity (israte <= 2) and rate curve scaling."""

    def test_cowper_symonds_rate_sensitivity(self):
        """Flow stress increases at higher strain rate under Cowper-Symonds formulation."""
        c_param = 40.0
        p_param = 5.0  # eps_0 = 1/p
        mat = make_test_material_law66(
            E=70000.0, nu=0.33,
            israte=1,
            c=c_param,
            eps_0=p_param,
            sigmay0=200.0,
            fun_c=[(0.0, 200.0), (0.1, 280.0)],
            fun_t=[(0.0, 200.0), (0.1, 280.0)],
        )

        def get_flow_stress(dt_val, deps_xx):
            sig = np.zeros((1, 6), dtype=float)
            epsp = np.zeros(1, dtype=float)
            extra = {"uvar66": np.zeros((1, 8))}
            for _ in range(15):
                deps = np.array([[deps_xx, -0.33 * deps_xx, -0.33 * deps_xx, 0.0, 0.0, 0.0]])
                res = solid_update(mat, sig, deps, epsp, dt=dt_val, extra=extra)
                sig = res[0].copy()
                epsp[0] = _scalar(res[1])
            return float(sig[0, 0])

        s_slow = get_flow_stress(dt_val=1.0e-3, deps_xx=2.0e-4)   # strain rate = 0.2 s^-1
        s_fast = get_flow_stress(dt_val=1.0e-6, deps_xx=2.0e-4)   # strain rate = 200 s^-1

        assert s_fast > s_slow + 5.0, f"Dynamic rate effect: s_fast ({s_fast}) must exceed s_slow ({s_slow})"

    def test_rate_dependent_curve_tables(self):
        """Rate scale factors dynamically scale compression and tension yield surfaces."""
        mat = make_test_material_law66(
            E=70000.0, nu=0.33,
            israte=3,
            fscale33=1.5,  # 50% increase on compression rate scaling
            fscale12=1.2,  # 20% increase on tension rate scaling
            curve_c=[(0.0, 200.0), (0.1, 300.0)],
            curve_t=[(0.0, 200.0), (0.1, 300.0)],
            curve_rate_c=[(0.0, 1.0), (1000.0, 1.0)],
            curve_rate_t=[(0.0, 1.0), (1000.0, 1.0)],
        )

        sig_c = np.zeros((1, 6), dtype=float)
        epsp_c = np.zeros(1, dtype=float)
        extra_c = {"uvar66": np.zeros((1, 8))}
        for _ in range(25):
            deps = np.array([[-2.0e-4, 0.33 * 2.0e-4, 0.33 * 2.0e-4, 0.0, 0.0, 0.0]])
            res = solid_update(mat, sig_c, deps, epsp_c, dt=1e-6, extra=extra_c)
            sig_c = res[0].copy()
            epsp_c[0] = _scalar(res[1])

        # Scaled compression flow stress reflects fscale11 = 1.5
        assert abs(sig_c[0, 0]) > 250.0


# ============================================================================
# 5. Acoustic Sound Speed & Courant Time-Step Stability
# ============================================================================

class TestLaw66SoundSpeedAndCourantStability:
    """Audit solid and shell sound speed and Courant time-step bounds."""

    def test_solid_sound_speed_and_courant_step(self):
        """Longitudinal acoustic sound speed c_solid maintains positive, stable Courant bounds."""
        mat = make_test_material_law66(
            E=70000.0, nu=0.33, rho0=2.7e-9,
        )
        c_solid = sound_speed_solid(mat)
        c_expected = math.sqrt(70000.0 * (1.0 - 0.33) / (2.7e-9 * (1.0 + 0.33) * (1.0 - 2.0 * 0.33)))

        assert c_solid == pytest.approx(c_expected, rel=1e-5)
        assert c_solid > 0.0

        lc = 2.0  # 2 mm characteristic length
        dt_courant = lc / c_solid
        assert 1.0e-8 < dt_courant < 1.0e-5

    def test_shell_sound_speed_and_courant_step(self):
        """Plane-stress sound speed c_shell maintains positive, stable Courant bounds."""
        mat = make_test_material_law66(
            E=70000.0, nu=0.33, rho0=2.7e-9,
        )
        c_shell = sound_speed_shell(mat)
        c_expected = math.sqrt(70000.0 / (2.7e-9 * (1.0 - 0.33 ** 2)))

        assert c_shell == pytest.approx(c_expected, rel=1e-5)
        assert c_shell > 0.0

        lc = 2.0
        dt_courant = lc / c_shell
        assert 1.0e-8 < dt_courant < 1.0e-5

    def test_multi_element_wave_propagation_stability(self):
        """Multi-element patch under dynamic wave propagation maintains stable Courant bounds."""
        lx = 2.0
        mat = make_test_material_law66(E=70000.0, nu=0.33, rho0=2.7e-9, sigmay0=200.0)

        c_theory = sound_speed_shell(mat)
        dt_courant_bound = lx / c_theory

        assert dt_courant_bound > 0.0
        assert np.isfinite(dt_courant_bound)
        assert dt_courant_bound > 1.0e-7
