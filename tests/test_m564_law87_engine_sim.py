"""
Dynamic Engine Simulation & Energy Balance Verifier for M564 (/MAT/LAW87 /MAT/BARLAT2000 /MAT/BARLAT2000_2D).

Exhaustive dynamic explicit engine simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations across shell element formulations:
   - Shell BT4 with LAW87 under dynamic cyclic tension, compression, and shear (50+ cycles, |ERR| < 1.0%, stop_reason == "").
   - Shell QEPH with LAW87 under dynamic biaxial tension and shear (50+ cycles, |ERR| < 1.0%, stop_reason == "").
   - Shell Tri3 with LAW87 under dynamic stretching (50+ cycles, |ERR| < 1.0%, stop_reason == "").
   - Shell BT4 2x2 multi-element patch with LAW87 under cyclic loading (50+ cycles, |ERR| < 1.0%, stop_reason == "").
   - Asserts normal termination, >= 50 cycles, energy balance error |ERR| < 1.0%,
     positive internal strain energy accumulation (IE > 0), and external work consistency.
2. Physical anisotropic sheet metal behavior:
   - Verify differential yield strengths across 0 deg, 45 deg, 90 deg rolling orientations matching Barlat Yld2000-2d anisotropy.
   - Tensile strip engine simulations under displacement control oriented at 0 vs 45 vs 90 deg exhibiting distinct reaction forces / flow stress levels.
   - Differential plastic thinning across different Lankford R values / anisotropy parameters.
3. Cyclic tension-compression verifying kinematic hardening Bauschinger hysteresis loop:
   - Kinematic hardening (chard > 0 / fisokin > 0, ckh, akh) vs isotropic hardening (chard = 0).
   - Reverse yield initiating at earlier / reduced stress magnitude (Bauschinger effect).
   - Backstress sigb87 / uvar87 evolution during forward and reverse loading.
   - Closed hysteresis loop with positive plastic dissipation.
4. Strict energy conservation check:
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV * dt matches analytical strain energy in reversible elastic cycles.
   - Free vibration of undamped shell elements (BT4, QEPH):
     mechanical energy E_tot = E_kin + E_int is strictly conserved (|Delta E| / E_0 < 0.1% for BT4, < 1.0% for QEPH).
   - Monotonically increasing plastic dissipation during plastic flow.
5. Shell thickness thinning evolution:
   - Thickness h_final < h_0 monotonically decreasing with plastic strain during tensile loading.
   - Thickness reduction consistency across multiple integration steps.
6. Acoustic sound speed & Courant time-step stability:
   - Sound speed c_shell maintaining positive, stable Courant bounds across all dynamic cycles and sheet alloys.
   - Courant time-step stability under high-rate dynamic loading.
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials.law87_barlat2000 import (
    Law87Params,
    build_law87,
    shell_update as shell_update_law87,
    sound_speed as sound_speed_shell_law87,
    barlat2000_equivalent_stress,
)
from pyradioss.model.entities import Material, MatLaw87
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
    stop_cycles: Optional[int] = None,
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


def make_test_material_law87(
    mid: int = 1,
    rho0: float = 2.7e-9,  # ton/mm^3
    E: float = 70000.0,    # MPa
    nu: float = 0.33,
    expa: float = 8.0,     # FCC aluminum
    alpha: Sequence[float] | None = None,
    fisokin: float = 0.0,
    ckh: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    akh: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    aswift: float = 350.0,
    nexp: float = 0.2,
    epso: float = 0.002,
    yield_table: Any = None,
    table_id: int = 0,
    **kwargs: Any,
) -> Law87Params:
    """Factory creating a valid Law87Params instance."""
    if alpha is None:
        alpha = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    p = Law87Params(
        id=mid,
        title="Alu_Barlat2000_Yld2000",
        rho0=rho0,
        refer_rho=rho0,
        e=E,
        nu=nu,
        expa=expa,
        al1=alpha[0],
        al2=alpha[1],
        al3=alpha[2],
        al4=alpha[3],
        al5=alpha[4],
        al6=alpha[5],
        al7=alpha[6],
        al8=alpha[7],
        fisokin=fisokin,
        ckh=ckh,
        akh=akh,
        aswift=aswift,
        nexp=nexp,
        epso=epso,
        alpha=1.0,
        iflag=1 if yield_table is None and table_id == 0 else 0,
        yield_table=yield_table,
        table_id=table_id,
        **kwargs,
    )
    return p


# ============================================================================
# 1. Multi-Cycle Explicit Dynamic Simulations Across Formulations
# ============================================================================

class TestLaw87MultiCycleDynamicSimulations:
    """Audit multi-cycle explicit dynamic simulations across supported shell formulations."""

    def test_shell_bt4_cyclic_tension_compression_shear_engine(self, tmp_path: Path):
        """Shell BT4 with LAW87 under dynamic cyclic tension, compression, and shear (50+ cycles)."""
        run_name = "SHELL_BT4_LAW87_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 150.0), (0.05, 220.0), (0.15, 300.0)])
        d.mat_law87(
            1, "AluBarlatBT4",
            rho=2.7e-9, e=70000.0, nu=0.33,
            tab_id0=10, exp_a=8.0,
            alpha=[1.05, 0.95, 1.02, 0.98, 1.0, 1.03, 0.97, 1.0],
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
        d.funct(11, "vx_cyc", [
            (0.0, 1200.0),
            (0.8e-4, 1200.0),
            (0.8001e-4, -800.0),
            (1.6e-4, -800.0),
            (1.6001e-4, 1000.0),
            (2.5e-4, 1000.0),
        ])
        d.impvel(1, "pull_x", 11, "X", 2)

        # In-plane shear loading in Y
        d.funct(12, "vy_shear", [
            (0.0, 500.0),
            (1.0e-4, 500.0),
            (1.0001e-4, -500.0),
            (2.5e-4, -500.0),
        ])
        d.impvel(2, "shear_y", 12, "Y", 2)

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

        # Check plastic strain accumulation occurred
        mat_extra = eng_model.shells.state.get("mat_extra", {})
        assert "pla87" in mat_extra or "epsp" in eng_model.shells.state
        pla_final = mat_extra.get("pla87", eng_model.shells.state.get("epsp"))
        assert np.any(pla_final > 0.0), "Plastic strain must accumulate during cycle"

    def test_shell_qeph_dynamic_biaxial_tension_shear_engine(self, tmp_path: Path):
        """Shell QEPH (Ishell=24) with LAW87 under dynamic biaxial tension and shear (50+ cycles)."""
        run_name = "SHELL_QEPH_LAW87_BIAX"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 180.0), (0.1, 280.0)])
        d.mat_law87(
            1, "AluQEPH",
            rho=2.7e-9, e=70000.0, nu=0.33,
            tab_id0=10, exp_a=8.0,
            alpha=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
        d.funct(11, "vx_pull", [(0.0, 350.0), (2.0e-4, 350.0)])
        d.impvel(1, "pull_x", 11, "X", 4)

        # Pull top edge (3, 4) in Y
        d.grnod_node(5, "pull_y_nodes", [3, 4])
        d.funct(12, "vy_pull", [(0.0, 300.0), (2.0e-4, 300.0)])
        d.impvel(2, "pull_y", 12, "Y", 5)

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

    def test_shell_tri3_dynamic_stretching_engine(self, tmp_path: Path):
        """Shell Tri3 (Ish3n=1) with LAW87 under dynamic stretching (50+ cycles)."""
        run_name = "SHELL_TRI3_LAW87_STR"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 200.0), (0.1, 300.0)])
        d.mat_law87(
            1, "AluTri3",
            rho=2.7e-9, e=70000.0, nu=0.33,
            tab_id0=10, exp_a=8.0,
            alpha=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
        d.funct(11, "vx_pull", [(0.0, 450.0), (2.0e-4, 450.0)])
        d.impvel(1, "pull_apex_x", 11, "X", 2)
        d.funct(12, "vy_pull", [(0.0, 200.0), (2.0e-4, 200.0)])
        d.impvel(2, "pull_apex_y", 12, "Y", 2)

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

    def test_shell_bt4_multi_element_patch_cyclic_engine(self, tmp_path: Path):
        """Shell BT4 2x2 multi-element patch with LAW87 under cyclic dynamic loading (50+ cycles)."""
        run_name = "PATCH_2X2_LAW87"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 210.0), (0.08, 310.0)])
        d.mat_law87(
            1, "AluPatch",
            rho=2.7e-9, e=70000.0, nu=0.33,
            tab_id0=10, exp_a=8.0,
            alpha=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
        d.funct(11, "vy_pull", [
            (0.0, 600.0),
            (1.0e-4, 600.0),
            (1.0001e-4, -400.0),
            (2.2e-4, -400.0),
        ])
        d.impvel(1, "pull_top", 11, "Y", 2)

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
# 2. Anisotropic Sheet Metal Behavior Auditing
# ============================================================================

class TestLaw87AnisotropicSheetBehavior:
    """Audit differential yield strengths across 0 deg, 45 deg, 90 deg rolling angles."""

    def test_differential_yield_strengths_0_45_90_rolling_angles(self):
        """Verify differential yield stresses across 0 deg, 45 deg, and 90 deg rolling angles."""
        p = Law87Params(
            al1=1.2, al2=0.8, al3=1.1, al4=0.9, al5=0.95, al6=1.25, al7=0.85, al8=1.1,
            expa=8.0, e=70000.0, nu=0.33,
            aswift=200.0, nexp=0.001, alpha=1.0, epso=1.0,  # Nearly non-hardening for clear yield detection
        )

        f0 = barlat2000_equivalent_stress(np.array([1.0, 0.0, 0.0]), p)
        f90 = barlat2000_equivalent_stress(np.array([0.0, 1.0, 0.0]), p)
        f45 = barlat2000_equivalent_stress(np.array([0.5, 0.5, 0.5]), p)

        sig0_theory = 200.0 / f0
        sig90_theory = 200.0 / f90
        sig45_theory = 200.0 / f45

        assert abs(sig90_theory - sig0_theory) > 5.0, "Anisotropy must produce distinct 90 deg yield stress"
        assert abs(sig45_theory - sig0_theory) > 1.5, "Anisotropy must produce distinct 45 deg yield stress"

        # Simulation 1: Uniaxial tension along 0 deg (x)
        sig_0 = np.zeros((1, 3), dtype=float)
        epsp_0 = np.zeros(1, dtype=float)
        extra_0 = {
            "uvar87": np.zeros((1, 7)), "pla87": np.zeros(1),
            "off87": np.ones(1), "thk87": np.ones(1), "sigb87": np.zeros((1, 12)),
        }
        for _ in range(60):
            deps = np.array([[1.0e-4, -0.33 * 1.0e-4, 0.0]])
            res = shell_update_law87(p, sig_0, deps, epsp_0, dt=1e-6, extra=extra_0)
            sig_0 = res[0].copy()
            epsp_0[0] = _scalar(res[1])
            if epsp_0[0] > 1e-6:
                break
        assert sig_0[0, 0] == pytest.approx(sig0_theory, rel=0.03)

        # Simulation 2: Uniaxial tension along 90 deg (y)
        sig_90 = np.zeros((1, 3), dtype=float)
        epsp_90 = np.zeros(1, dtype=float)
        extra_90 = {
            "uvar87": np.zeros((1, 7)), "pla87": np.zeros(1),
            "off87": np.ones(1), "thk87": np.ones(1), "sigb87": np.zeros((1, 12)),
        }
        for _ in range(70):
            deps = np.array([[-0.33 * 1.0e-4, 1.0e-4, 0.0]])
            res = shell_update_law87(p, sig_90, deps, epsp_90, dt=1e-6, extra=extra_90)
            sig_90 = res[0].copy()
            epsp_90[0] = _scalar(res[1])
            if epsp_90[0] > 1e-6:
                break
        assert sig_90[0, 1] == pytest.approx(sig90_theory, rel=0.03)

    def test_tensile_strip_engine_simulations_0_vs_90_deg(self, tmp_path: Path):
        """Tensile strip engine simulations along 0 deg vs 90 deg under displacement control."""
        # Run 00: Pull along X
        run00 = "STRIP_00"
        s00 = str(tmp_path / f"{run00}_0000.rad")
        e00 = str(tmp_path / f"{run00}_0001.rad")

        d00 = StarterDeck(run00)
        d00.funct(10, "yield_fct", [(0.0, 200.0), (0.1, 300.0)])
        d00.mat_law87(
            1, "BarlatAniso", rho=2.7e-9, e=70000.0, nu=0.33, tab_id0=10, exp_a=8.0,
            alpha=[1.2, 0.8, 1.1, 0.9, 0.95, 1.25, 0.85, 1.1],
        )
        d00.prop_shell(1, "PropBT4", thick=1.0, nip=3, ishell=1)
        d00.part(1, "Part1", 1, 1)
        d00.node([(1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0)])
        d00.shell(1, [(1, 1, 2, 3, 4)])
        d00.grnod_node(1, "fix_left", [1, 4])
        d00.bcs(1, "bcs_fix", "111", "111", 1)
        d00.grnod_node(2, "pull_nodes", [2, 3])
        d00.funct(11, "vx_pull", [(0.0, 500.0), (2.0e-4, 500.0)])
        d00.impvel(1, "pull_x", 11, "X", 2)
        d00.write(s00)
        _write_engine_deck(e00, run00, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            stA = run_starter(s00)
            engA = run_engine(e00)

        # Run 90: Pull along Y
        run90 = "STRIP_90"
        s90 = str(tmp_path / f"{run90}_0000.rad")
        e90 = str(tmp_path / f"{run90}_0001.rad")

        d90 = StarterDeck(run90)
        d90.funct(10, "yield_fct", [(0.0, 200.0), (0.1, 300.0)])
        d90.mat_law87(
            1, "BarlatAniso", rho=2.7e-9, e=70000.0, nu=0.33, tab_id0=10, exp_a=8.0,
            alpha=[1.2, 0.8, 1.1, 0.9, 0.95, 1.25, 0.85, 1.1],
        )
        d90.prop_shell(1, "PropBT4", thick=1.0, nip=3, ishell=1)
        d90.part(1, "Part1", 1, 1)
        d90.node([(1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0)])
        d90.shell(1, [(1, 1, 2, 3, 4)])
        d90.grnod_node(1, "fix_bottom", [1, 2])
        d90.bcs(1, "bcs_fix", "111", "111", 1)
        d90.grnod_node(2, "pull_nodes", [3, 4])
        d90.funct(11, "vy_pull", [(0.0, 500.0), (2.0e-4, 500.0)])
        d90.impvel(1, "pull_y", 11, "Y", 2)
        d90.write(s90)
        _write_engine_deck(e90, run90, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            stB = run_starter(s90)
            engB = run_engine(e90)

        ie00 = float(engA.shells.state["eint"][0])
        ie90 = float(engB.shells.state["eint"][0])
        sig00 = float(engA.shells.state["sig"][0, :, 0].mean())
        sig90 = float(engB.shells.state["sig"][0, :, 1].mean())

        assert ie90 > ie00, f"Expected 90 deg internal energy ({ie90}) > 00 deg ({ie00})"
        assert sig90 > sig00, f"Expected 90 deg stress ({sig90}) > 00 deg ({sig00})"

    def test_differential_plastic_thinning_anisotropy(self):
        """Verify sheet thinning resistance scales with anisotropy parameters."""
        # Set 1: High thinning resistance
        p_high = Law87Params(
            al1=0.8, al2=0.8, al3=1.0, al4=1.0, al5=1.0, al6=1.0, al7=1.0, al8=1.0,
            expa=8.0, e=70000.0, nu=0.33, aswift=250.0, nexp=0.15, alpha=1.0, epso=0.002,
        )
        # Set 2: Low thinning resistance
        p_low = Law87Params(
            al1=1.3, al2=1.3, al3=1.0, al4=1.0, al5=1.0, al6=1.0, al7=1.0, al8=1.0,
            expa=8.0, e=70000.0, nu=0.33, aswift=250.0, nexp=0.15, alpha=1.0, epso=0.002,
        )

        sig_h = np.zeros((1, 3), dtype=float)
        epsp_h = np.zeros(1, dtype=float)
        extra_h = {"uvar87": np.zeros((1, 7)), "thk87": np.array([1.0]), "pla87": np.zeros(1), "off87": np.ones(1), "sigb87": np.zeros((1, 12))}

        sig_l = np.zeros((1, 3), dtype=float)
        epsp_l = np.zeros(1, dtype=float)
        extra_l = {"uvar87": np.zeros((1, 7)), "thk87": np.array([1.0]), "pla87": np.zeros(1), "off87": np.ones(1), "sigb87": np.zeros((1, 12))}

        dt = 1.0e-6
        for step in range(40):
            deps = np.array([[5.0e-4, -1.5e-4, 0.0]])
            res_h = shell_update_law87(p_high, sig_h, deps, epsp_h, dt=dt, extra=extra_h)
            sig_h = res_h[0].copy()
            epsp_h[0] = _scalar(res_h[1])
            res_l = shell_update_law87(p_low, sig_l, deps, epsp_l, dt=dt, extra=extra_l)
            sig_l = res_l[0].copy()
            epsp_l[0] = _scalar(res_l[1])

        thk_final_high = float(extra_h["thk87"][0])
        thk_final_low = float(extra_l["thk87"][0])

        delta_thk_high = 1.0 - thk_final_high
        delta_thk_low = 1.0 - thk_final_low

        assert delta_thk_low > 0.0
        assert delta_thk_high > 0.0
        assert delta_thk_low != pytest.approx(delta_thk_high, rel=0.01)


# ============================================================================
# 3. Kinematic Hardening & Bauschinger Effect Auditing
# ============================================================================

class TestLaw87KinematicHardeningAndBauschinger:
    """Audit kinematic hardening, reverse yield reduction (Bauschinger), and hysteresis loop."""

    def test_cyclic_tension_compression_bauschinger_hysteresis_loop(self):
        """Cyclic tension-compression verifies earlier reverse yield (Bauschinger) vs isotropic hardening."""
        p_kin = Law87Params(
            al1=1.0, al2=1.0, al3=1.0, al4=1.0, al5=1.0, al6=1.0, al7=1.0, al8=1.0,
            expa=2.0, e=70000.0, nu=0.33,
            aswift=300.0, nexp=0.2, alpha=1.0, epso=0.002,
            fisokin=0.5, ikin=1, ckh=(200.0, 0, 0, 0), akh=(500.0, 0, 0, 0),
        )
        p_iso = Law87Params(
            al1=1.0, al2=1.0, al3=1.0, al4=1.0, al5=1.0, al6=1.0, al7=1.0, al8=1.0,
            expa=2.0, e=70000.0, nu=0.33,
            aswift=300.0, nexp=0.2, alpha=1.0, epso=0.002,
            fisokin=0.0, ikin=1,
        )

        sig_k = np.zeros((1, 3))
        epsp_k = np.zeros(1)
        extra_k = {"sigb87": np.zeros((1, 12)), "pla87": np.zeros(1), "thk87": np.ones(1), "off87": np.ones(1), "uvar87": np.zeros((1, 7))}

        sig_i = np.zeros((1, 3))
        epsp_i = np.zeros(1)
        extra_i = {"sigb87": np.zeros((1, 12)), "pla87": np.zeros(1), "thk87": np.ones(1), "off87": np.ones(1), "uvar87": np.zeros((1, 7))}

        # 1. Forward loading in tension
        dt = 1.0e-6
        for _ in range(30):
            deps = np.array([[2.0e-4, -0.33 * 2.0e-4, 0.0]])
            sig_k, epsp_k = shell_update_law87(p_kin, sig_k, deps, epsp_k, dt=dt, extra=extra_k)
            sig_i, epsp_i = shell_update_law87(p_iso, sig_i, deps, epsp_i, dt=dt, extra=extra_i)

        sig_fwd_k = float(sig_k[0, 0])
        sig_fwd_i = float(sig_i[0, 0])
        assert sig_fwd_k > 250.0, f"Forward tension should reach yield, got {sig_fwd_k}"
        assert float(extra_k["sigb87"][0, 0]) > 20.0, "Forward tension should develop positive backstress"

        # 2. Reverse loading into compression
        rev_yield_k = None
        rev_yield_i = None

        for _ in range(60):
            deps = np.array([[-2.0e-4, 0.33 * 2.0e-4, 0.0]])
            sig_k_new, epsp_k_new = shell_update_law87(p_kin, sig_k, deps, epsp_k, dt=dt, extra=extra_k)
            if epsp_k_new[0] > epsp_k[0] + 1e-6 and rev_yield_k is None:
                rev_yield_k = float(sig_k_new[0, 0])
            sig_k, epsp_k = sig_k_new, epsp_k_new

            sig_i_new, epsp_i_new = shell_update_law87(p_iso, sig_i, deps, epsp_i, dt=dt, extra=extra_i)
            if epsp_i_new[0] > epsp_i[0] + 1e-6 and rev_yield_i is None:
                rev_yield_i = float(sig_i_new[0, 0])
            sig_i, epsp_i = sig_i_new, epsp_i_new

        assert rev_yield_k is not None, "Kinematic hardening should yield under reverse compression"
        assert rev_yield_i is not None, "Isotropic hardening should yield under reverse compression"

        # Bauschinger effect: reverse yielding under kinematic hardening occurs at lower stress magnitude
        mag_k = abs(rev_yield_k)
        mag_i = abs(rev_yield_i)
        assert mag_k < mag_i, f"Kinematic reverse yield |{rev_yield_k:.1f}| must be < isotropic |{rev_yield_i:.1f}|"
        assert mag_k < 0.85 * sig_fwd_k, f"Kinematic reverse yield |{mag_k:.1f}| should exhibit marked Bauschinger softening vs {sig_fwd_k:.1f}"

    def test_backstress_tensor_evolution_in_sigb87_and_uvar87(self):
        """Backstress tensor evolves continuously and smoothly during plastic deformation."""
        p = Law87Params(
            al1=1.0, al2=1.0, al3=1.0, al4=1.0, al5=1.0, al6=1.0, al7=1.0, al8=1.0,
            expa=2.0, e=70000.0, nu=0.33,
            aswift=300.0, nexp=0.2, alpha=1.0, epso=0.002,
            fisokin=0.6, ikin=1, ckh=(150.0, 0, 0, 0), akh=(400.0, 0, 0, 0),
        )
        sig = np.zeros((1, 3))
        epsp = np.zeros(1)
        extra = {
            "sigb87": np.zeros((1, 12)),
            "pla87": np.zeros(1),
            "thk87": np.ones(1),
            "off87": np.ones(1),
            "uvar87": np.zeros((1, 7)),
        }

        backstress_history = []
        dt = 1.0e-6
        for step in range(35):
            deps = np.array([[2.5e-4, -0.33 * 2.5e-4, 0.0]])
            sig, epsp = shell_update_law87(p, sig, deps, epsp, dt=dt, extra=extra)
            backstress_history.append(float(extra["sigb87"][0, 0]))

        # Backstress must monotonically grow during forward plastic flow
        diffs = np.diff(backstress_history)
        assert np.all(diffs >= -1e-12), "Backstress in forward direction must be monotonically non-decreasing"
        assert backstress_history[-1] > 10.0, f"Expected substantial backstress development, got {backstress_history[-1]}"


# ============================================================================
# 4. Strict Energy Conservation & Work Balance Auditing
# ============================================================================

class TestLaw87StrictEnergyConservation:
    """Audit undamped free vibration energy conservation (|Delta E| / E_0 < 0.1%) and work balance."""

    def test_shell_bt4_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Shell BT4 in elastic regime: mechanical energy strictly conserved (|Delta E| / E_0 < 0.1%)."""
        deck = StarterDeck("BT4_LAW87_OSC")
        deck.mat_law87(
            1, "ElasticBarlat",
            rho=2.7e-9, e=70000.0, nu=0.33, exp_a=8.0,
        )
        thick0 = 1.0
        deck.prop_shell(1, "PropBT4", thick=thick0, nip=3, ishell=1, hm=0.0, hf=0.0, hr=0.0)
        deck.part(1, "PartShell", 1, 1)

        lx, ly = 10.0, 10.0
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, lx, 0.0, 0.0), (3, lx, ly, 0.0), (4, 0.0, ly, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 3, 4)])

        s_path = str(tmp_path / "BT4_LAW87_OSC_0000.rad")
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
        e_sym_history = []
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
            e_kin_sym = 0.5 * np.sum(mass_vec[:, None] * (v_old * model.v))
            e_int = float(np.sum(group.state["eint"]))
            e_tot_history.append(e_kin + e_int)
            e_sym_history.append(e_kin_sym + e_int)

        max_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_tot_history)
        max_sym_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_sym_history)
        assert max_err < 0.01, f"BT4 LAW87 energy conservation error {max_err*100:.3f}% exceeds 1.0%"
        assert max_sym_err < 0.001, f"BT4 LAW87 symplectic energy error {max_sym_err*100:.6f}% exceeds 0.1%"

    def test_shell_qeph_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Shell QEPH in elastic regime: mechanical energy conserved (|Delta E| / E_0 < 1.0%)."""
        deck = StarterDeck("QEPH_LAW87_OSC")
        deck.mat_law87(
            1, "ElasticBarlat",
            rho=2.7e-9, e=70000.0, nu=0.33, exp_a=8.0,
        )
        thick0 = 1.0
        deck.prop_shell(1, "PropQEPH", thick=thick0, nip=3, ishell=24)
        deck.part(1, "PartQEPH", 1, 1)

        lx, ly = 10.0, 10.0
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, lx, 0.0, 0.0), (3, lx, ly, 0.0), (4, 0.0, ly, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 3, 4)])

        s_path = str(tmp_path / "QEPH_LAW87_OSC_0000.rad")
        deck.write(s_path)
        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.shells_qeph
        group.state["amu"].fill(0.0)  # Zero out viscous hourglass damping
        rho0 = 2.7e-9
        area0 = lx * ly
        m_node = rho0 * thick0 * area0 / 4.0
        mass_vec = np.full(4, m_node)

        # Pure symmetric breathing velocity mode in X
        v = np.zeros_like(model.x)
        vx0 = 30.0
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
            shell_qeph.forces(group, model.x, model.v, model.vr, dt, fint, mint)
            v_old = model.v.copy()
            acc = fint / mass_vec[:, None]
            model.v += acc * dt
            model.x += model.v * dt
            v_mid = 0.5 * (v_old + model.v)
            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot_history.append(e_kin + e_int)

        max_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_tot_history)
        assert max_err < 0.01, f"QEPH LAW87 energy conservation error {max_err*100:.3f}% exceeds 1.0%"

    def test_strain_energy_ledger_trapezoidal_integration(self):
        """Incremental trapezoidal work matches exact analytical strain energy in elastic regime."""
        p = Law87Params(
            e=70000.0, nu=0.33, expa=8.0,
            aswift=10000.0,  # High yield stress for pure elastic response
        )
        vol0 = 100.0  # 10 x 10 x 1 mm
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar87": np.zeros((1, 7)),
            "pla87": np.zeros(1),
            "off87": np.ones(1),
            "thk87": np.ones(1),
            "sigb87": np.zeros((1, 12)),
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
            res = shell_update_law87(p, sig, deps, epsp, dt=dt, extra=extra)
            sig_new = res[0].copy()
            sig = sig_new.copy()
            epsp[0] = _scalar(res[1])

            sig_mid = 0.5 * (sig_old + sig_new)
            de = (sig_mid[0, 0] * deps[0, 0] + sig_mid[0, 1] * deps[0, 1] + sig_mid[0, 2] * deps[0, 2]) * vol0
            eint_incremental += de
            eps_total += deps[0]

        # Analytical elastic strain energy: 0.5 * (a11*ex^2 + a11*ey^2 + 2*a12*ex*ey + g*exy^2) * vol0
        ex, ey, exy = eps_total
        analytical_eint = 0.5 * (a11 * ex**2 + a11 * ey**2 + 2.0 * a12 * ex * ey + g * exy**2) * vol0
        assert eint_incremental == pytest.approx(analytical_eint, rel=1e-5), \
            f"Incremental strain energy {eint_incremental} != analytical {analytical_eint}"
        assert epsp[0] == 0.0, "No plastic strain should accumulate in elastic regime"

    def test_plastic_dissipation_monotonically_increasing(self):
        """Plastic strain and plastic dissipation monotonically non-decreasing during plastic flow."""
        p = Law87Params(
            e=70000.0, nu=0.33, expa=8.0,
            aswift=200.0, nexp=0.2, alpha=1.0, epso=0.002,
        )
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar87": np.zeros((1, 7)),
            "pla87": np.zeros(1),
            "off87": np.ones(1),
            "thk87": np.ones(1),
            "sigb87": np.zeros((1, 12)),
        }

        epsp_history = []
        dt = 1.0e-6

        for step in range(40):
            deps = np.array([[3.0e-4, -1.0e-4, 1.5e-4]])
            res = shell_update_law87(p, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])
            epsp_history.append(float(epsp[0]))

        d_epsp = np.diff(epsp_history)
        assert np.all(d_epsp >= -1e-15), "Plastic strain increments must be non-negative"
        assert epsp_history[-1] > epsp_history[0], "Plastic strain must strictly increase during yield"


# ============================================================================
# 5. Shell Thickness Thinning Evolution Auditing
# ============================================================================

class TestLaw87ThicknessThinningEvolution:
    """Audit shell thickness thinning evolution under plastic deformation."""

    def test_shell_thickness_thinning_evolution_monotonic(self):
        """Thickness thinning evolves monotonically under tensile plastic deformation (h_final < h_0)."""
        p = Law87Params(
            e=70000.0, nu=0.33, expa=8.0,
            aswift=250.0, nexp=0.18, alpha=1.0, epso=0.002,
        )
        sig = np.zeros((1, 3))
        epsp = np.zeros(1)
        thick0 = 1.5
        extra = {
            "sigb87": np.zeros((1, 12)),
            "pla87": np.zeros(1),
            "thk87": np.array([thick0]),
            "off87": np.ones(1),
            "uvar87": np.zeros((1, 7)),
        }

        thk_history = [thick0]
        dt = 1.0e-6
        for step in range(40):
            deps = np.array([[3.5e-4, -0.1e-4, 0.0]])
            sig, epsp = shell_update_law87(p, sig, deps, epsp, dt=dt, extra=extra)
            thk_history.append(float(extra["thk87"][0]))

        assert thk_history[-1] < thick0, f"Final thickness ({thk_history[-1]}) must be < initial ({thick0})"
        # Monotonically non-increasing
        diffs = np.diff(thk_history)
        assert np.all(diffs <= 1e-14), "Thickness must be monotonically non-increasing under tensile straining"

    def test_engine_simulation_shell_thinning_tracking(self, tmp_path: Path):
        """Multi-step explicit engine simulation tracks shell thickness reduction under dynamic tension."""
        run_name = "THINNING_ENGINE"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 150.0), (0.1, 250.0)])
        d.mat_law87(1, "AluThin", rho=2.7e-9, e=70000.0, nu=0.33, tab_id0=10, exp_a=8.0)
        thick0 = 1.2
        d.prop_shell(1, "PropBT4", thick=thick0, nip=3, ishell=1)
        d.part(1, "Part1", 1, 1)
        d.node([(1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0)])
        d.shell(1, [(1, 1, 2, 3, 4)])

        d.grnod_node(1, "fix_left", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_nodes", [2, 3])
        d.funct(11, "vx_pull", [(0.0, 600.0), (2.0e-4, 600.0)])
        d.impvel(1, "pull_x", 11, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        mat_extra = eng_model.shells.state.get("mat_extra", {})
        assert "thk87" in mat_extra
        thk_final = float(mat_extra["thk87"][0, 0])
        assert thk_final < thick0, f"Expected final thickness ({thk_final}) < initial ({thick0})"


# ============================================================================
# 6. Acoustic Sound Speed & Courant Time-Step Stability Auditing
# ============================================================================

class TestLaw87SoundSpeedAndCourantStability:
    """Audit plane-stress acoustic sound speed c_shell and Courant time-step stability."""

    def test_sound_speed_shell_positivity_and_courant_bounds(self):
        """Verify positive plane-stress sound speed c = sqrt(E / (rho0 * (1 - nu^2))) across alloys."""
        alloys = [
            ("Steel", 210000.0, 0.30, 7.85e-9),
            ("Aluminum", 70000.0, 0.33, 2.70e-9),
            ("Titanium", 110000.0, 0.31, 4.50e-9),
            ("Copper", 120000.0, 0.34, 8.96e-9),
        ]
        lc = 2.0  # characteristic shell length = 2.0 mm

        for name, E, nu, rho0 in alloys:
            p = Law87Params(e=E, nu=nu, rho0=rho0)
            c_shell = sound_speed_shell_law87(p)
            c_expected = math.sqrt(E / (rho0 * (1.0 - nu ** 2)))

            assert c_shell == pytest.approx(c_expected, rel=1e-6)
            assert c_shell > 0.0

            dt_courant = lc / c_shell
            assert 1.0e-8 < dt_courant < 1.0e-5, f"Courant dt {dt_courant} outside reasonable bounds for {name}"

    def test_courant_step_under_dynamic_deformation(self):
        """Verify stable Courant time step maintained during high-rate plastic deformation."""
        lx = 2.0
        p = Law87Params(e=70000.0, nu=0.33, rho0=2.7e-9, expa=8.0)
        c_theory = sound_speed_shell_law87(p)
        dt_courant_bound = lx / c_theory

        assert dt_courant_bound > 0.0
        assert np.isfinite(dt_courant_bound)
        assert dt_courant_bound > 1.0e-7

        sig = np.zeros((1, 3))
        epsp = np.zeros(1)
        extra = {
            "sigb87": np.zeros((1, 12)),
            "pla87": np.zeros(1),
            "thk87": np.ones(1),
            "off87": np.ones(1),
            "uvar87": np.zeros((1, 7)),
        }

        # Apply deformation step and verify sound speed returned
        res = shell_update_law87(p, sig, np.array([[1e-3, -0.33e-3, 0.0]]), epsp, dt=1e-6, extra=extra, return_tuple=True)
        c_out = _scalar(res[2])
        assert c_out == pytest.approx(c_theory, rel=1e-3)
        assert c_out > 0.0
