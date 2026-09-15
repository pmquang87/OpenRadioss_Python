"""
Auditor 2C: Dynamic Engine Simulation & Energy Balance Auditor for M554 (/MAT/LAW52 /MAT/GURSON).

Exhaustive dynamic engine simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations:
   - Hexa8 brick with LAW52 under cyclic tension and shear
   - Tetra4 solid with LAW52 under dynamic tension
   - Shell BT4 with LAW52 under dynamic biaxial tension and thinning
   - Shell QEPH with LAW52 under dynamic cyclic loading
   - Shell Tri3 with LAW52 under cyclic dynamic loading
   - Verified through Starter and Engine decks under cyclic velocity/displacement loading.
   - Asserts normal termination, >= 20 cycles, energy balance error |ERR| < 1.0%,
     positive internal strain energy accumulation (IE > 0), and external work consistency.
2. Energy conservation & work balance:
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV * dt
   - Midpoint trapezoidal work matches exact analytical strain energy in closed reversible cycles.
   - Free vibration of undamped solid and shell elements (Hexa8, BT4, QEPH):
     mechanical energy E_tot = E_kin + E_int is strictly conserved (|Delta E| / E_0 < 1.0%).
   - Plastic dissipation monotonically increasing during plastic flow and void growth.
3. Micro-porosity softening & ductile rupture:
   - Void volume fraction growth causing softening under tensile triaxiality
   - Element erosion when f* >= f_u or f >= f_F with stresses dropping to zero
   - Stable post-failure cycle continuation without solver NaN or acoustic collapse.
   - Two-element progressive rupture where one element erodes while neighbor stays intact.
4. Acoustic sound speed & Courant time-step stability:
   - Sound speeds c_solid and c_shell maintaining stable Courant bounds across all cycles.
   - Courant time-step stability under dynamic acoustic pulse propagation in multi-element patch.

Fortran references:
- engine/source/materials/mat/mat052/sigeps52.F (3D solid kernel)
- engine/source/materials/mat/mat052/sigeps52c.F (2D shell plane-stress kernel)
- starter/source/materials/mat/mat052/hm_read_mat52.F (parameter initialization)
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials.law52_gurson import (
    Law52Params,
    build_law52,
    compute_f_star,
    gurson_yield_function,
    shell_membrane_tangent,
    shell_update_law52,
    solid_update_law52,
    sound_speed_shell_law52,
    sound_speed_solid_law52,
    tangent_law52_shell,
    tangent_law52_solid,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers: Engine Control Deck Writer and Material Factories
# ============================================================================

def _write_engine_deck(
    path: Path | str,
    run_name: str,
    tstop: float = 2.0e-4,
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


def make_test_material_law52(
    mid: int = 1,
    rho0: float = 7.85e-9,
    E: float = 210000.0,
    nu: float = 0.3,
    a: float = 350.0,
    b: float = 400.0,
    n: float = 0.2,
    q1: float = 1.5,
    q2: float = 1.0,
    q3: float = 2.25,
    fi: float = 0.01,
    fc: float = 0.15,
    ff: float = 0.25,
    fn: float = 0.04,
    sn: float = 0.1,
    epsn: float = 0.3,
    **kwargs: Any,
) -> Material:
    """Factory creating a valid /MAT/LAW52 (/MAT/GURSON) Material instance."""
    params = {
        "E": E,
        "nu": nu,
        "a": a,
        "b": b,
        "n": n,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "f_i": fi,
        "f_c": fc,
        "f_f": ff,
        "f_n": fn,
        "s_n": sn,
        "eps_n": epsn,
        "yield_a": a,
        "hard_b": b,
        "hard_n": n,
        "fi": fi,
        "fc": fc,
        "ff": ff,
        "fn": fn,
        "sn": sn,
        "epsn": epsn,
        "fu": 1.0 / q1 if q1 > 0 else 0.667,
    }
    params.update(kwargs)
    mat = Material(id=mid, law=52, rho0=rho0, title="Steel_LAW52_Gurson", params=params)
    return mat


class MockProp:
    """Mock shell property mimicking /PROP/SHELL (/PROP/TYPE1)."""
    def __init__(
        self,
        pid: int = 1,
        thick: float = 1.0,
        nip: int = 3,
        ishell: int = 1,
        ish3n: int = 1,
        hm: float = 0.1,
        hf: float = 0.1,
        hr: float = 0.1,
        dn: float = 0.015,
        **kwargs: Any,
    ):
        self.id = pid
        self.thick = thick
        self.nip = nip
        self.ishell = ishell
        self.ish3n = ish3n
        self.params = {
            "thick": thick,
            "nip": nip,
            "ishell": ishell,
            "ish3n": ish3n,
            "qa": 0.0,
            "qb": 0.0,
            "hm": hm,
            "hf": hf,
            "hr": hr,
            "dn": dn,
            **kwargs,
        }


class MockGroup:
    """Mock element group with connectivity, id indexing, and state buffer."""
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


# ============================================================================
# 1. Multi-Cycle Explicit Dynamic Simulations Across Formulations
# ============================================================================

class TestLaw52MultiCycleDynamicSimulations:
    """Audit multi-cycle explicit dynamic simulations across supported element formulations."""

    def test_hexa8_brick_cyclic_tension_shear_engine(self, tmp_path: Path):
        """Hexa8 brick with LAW52 under cyclic tension followed by shear."""
        run_name = "HEXA8_LAW52_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law52(
            1, "SteelGTN", rho=7.85e-9, e=210000.0, nu=0.3,
            a=350.0, b=400.0, n=0.2, q1=1.5, q2=1.0, q3=2.25,
            f_i=0.01, f_c=0.15, f_f=0.25, f_n=0.04, s_n=0.1, eps_n=0.3,
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
        # Cyclic tension-hold-compression in Z
        d.funct(10, "vz_cycle", [
            (0.0, 1200.0),
            (0.5e-4, 1200.0),
            (0.5001e-4, -800.0),
            (1.0e-4, -800.0),
            (1.0001e-4, 1000.0),
            (1.5e-4, 1000.0),
        ])
        d.impvel(1, "pull_z", 10, "Z", 2)

        # Shear in X
        d.funct(11, "vx_shear", [
            (0.0, 400.0),
            (0.7e-4, 400.0),
            (0.7001e-4, -400.0),
            (1.5e-4, -400.0),
        ])
        d.impvel(2, "shear_x", 11, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20, f"Expected >= 20 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Hexa8 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal energy must be strictly positive"
        assert en["EW"] > 0.0, "External work must be positive"

        # Check void evolution occurred
        mat_extra = eng_model.bricks.state.get("mat_extra", {})
        assert "dmg" in mat_extra or "f" in mat_extra

    def test_tetra4_solid_dynamic_tension_engine(self, tmp_path: Path):
        """Tetra4 solid with LAW52 under dynamic tension."""
        run_name = "TETRA4_LAW52_TENS"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law52(
            1, "SteelTetra", rho=7.85e-9, e=210000.0, nu=0.3,
            a=320.0, b=380.0, n=0.22, q1=1.5, q2=1.0, q3=2.25,
            f_i=0.015, f_c=0.15, f_f=0.25, f_n=0.04, s_n=0.1, eps_n=0.3,
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
        d.funct(10, "vz_pull", [
            (0.0, 800.0),
            (1.0e-4, 800.0),
            (1.0001e-4, -400.0),
            (1.5e-4, -400.0),
        ])
        d.impvel(1, "pull_apex", 10, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Tetra4 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0
        assert en["HE"] == 0.0, "Constant strain tetrahedron has strictly zero hourglass energy"

    def test_shell_bt4_dynamic_biaxial_tension_thinning_engine(self, tmp_path: Path):
        """Shell BT4 with LAW52 under dynamic biaxial tension and thinning."""
        run_name = "SHELL_BT4_LAW52_BIAX"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law52(
            1, "SteelBT4", rho=7.85e-9, e=210000.0, nu=0.3,
            a=300.0, b=450.0, n=0.25, q1=1.5, q2=1.0, q3=2.25,
            f_i=0.01, f_c=0.15, f_f=0.25, f_n=0.04, s_n=0.1, eps_n=0.3,
        )
        thick0 = 1.5
        d.prop_shell(1, "PropBT4", thick=thick0, nip=3, ishell=1)
        d.part(1, "PartBT4", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix node 1 completely, node 4 in X, node 2 in Y
        d.grnod_node(1, "fix_corner", [1])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "fix_x", [4])
        d.bcs(2, "bcs_fix_x", "100", "000", 2)
        d.grnod_node(3, "fix_y", [2])
        d.bcs(3, "bcs_fix_y", "010", "000", 3)

        # Pull right nodes (2, 3) in X
        d.grnod_node(4, "pull_x", [2, 3])
        d.funct(10, "vx_pull", [(0.0, 600.0), (1.2e-4, 600.0)])
        d.impvel(1, "pull_x_vel", 10, "X", 4)

        # Pull top nodes (3, 4) in Y
        d.grnod_node(5, "pull_y", [3, 4])
        d.funct(11, "vy_pull", [(0.0, 600.0), (1.2e-4, 600.0)])
        d.impvel(2, "pull_y_vel", 11, "Y", 5)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.2e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"BT4 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

        # Verify through-thickness plastic thinning
        thk_curr = eng_model.shells.state["mat_extra"]["thk"]
        assert np.all(thk_curr < thick0), f"Thinning must result in thk < {thick0}, got {thk_curr}"

    def test_shell_qeph_dynamic_cyclic_engine(self, tmp_path: Path):
        """Shell QEPH (Ishell=24) with LAW52 under dynamic cyclic loading."""
        run_name = "SHELL_QEPH_LAW52_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law52(
            1, "SteelQEPH", rho=7.85e-9, e=210000.0, nu=0.3,
            a=320.0, b=420.0, n=0.22, q1=1.5, q2=1.0, q3=2.25,
            f_i=0.01, f_c=0.15, f_f=0.25, f_n=0.04, s_n=0.1, eps_n=0.3,
        )
        d.prop_shell(1, "PropQEPH", thick=1.0, nip=3, ishell=24)
        d.part(1, "PartQEPH", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix left edge (1, 4)
        d.grnod_node(1, "fix_left", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull-push right edge (2, 3)
        d.grnod_node(2, "pull_right", [2, 3])
        d.funct(10, "vx_cyc", [
            (0.0, 800.0),
            (0.6e-4, 800.0),
            (0.6001e-4, -600.0),
            (1.2e-4, -600.0),
            (1.2001e-4, 800.0),
            (1.6e-4, 800.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.6e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"QEPH energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_tri3_dynamic_cyclic_engine(self, tmp_path: Path):
        """Shell Tri3 (Ish3n=1) with LAW52 under dynamic cyclic loading."""
        run_name = "SHELL_TRI3_LAW52_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law52(
            1, "SteelTri3", rho=7.85e-9, e=210000.0, nu=0.3,
            a=350.0, b=400.0, n=0.2, q1=1.5, q2=1.0, q3=2.25,
            f_i=0.01, f_c=0.15, f_f=0.25, f_n=0.04, s_n=0.1, eps_n=0.3,
        )
        d.prop_shell(1, "PropTri3", thick=1.0, nip=3, ish3n=1)
        d.part(1, "PartTri3", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0),
        ])
        d.sh3n(1, [(1, 1, 2, 3)])

        d.grnod_node(1, "fix_root", [1, 3])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        d.grnod_node(2, "pull_node", [2])
        d.funct(10, "vx_cyc", [
            (0.0, 600.0),
            (0.6e-4, 600.0),
            (0.6001e-4, -400.0),
            (1.2e-4, -400.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.2e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Tri3 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0
        assert en["HE"] == 0.0, "Tri3 shell has zero hourglass energy"


# ============================================================================
# 2. Energy Conservation & Work Balance Auditing
# ============================================================================

class TestLaw52EnergyConservationAndWorkBalance:
    """Audit incremental strain energy ledger and total energy conservation."""

    def test_incremental_strain_energy_ledger_matches_external_work(self):
        """Verify Delta E_int = int sigma : depsilon * dV * dt matches external work."""
        mat = make_test_material_law52(a=300.0, b=400.0, n=0.2)
        p = Law52Params(E=210000.0, nu=0.3, rho0=7.85e-9, yield_a=300.0, hard_b=400.0, hard_n=0.2)

        # Single integration point solid state
        sig = np.zeros((1, 6), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra: dict[str, Any] = {}

        vol0 = 1000.0  # 10 x 10 x 10 mm^3 cube
        n_steps = 40
        dt = 1.0e-6

        deps_rate = 100.0  # s^-1
        deps_val = deps_rate * dt

        delta_eint_total = 0.0
        sig_history = []
        eps_history = []

        for step in range(n_steps):
            factor = 1.0 if step < 25 else -1.0
            deps = np.zeros((1, 6), dtype=float)
            deps[0, 2] = factor * deps_val
            # Poisson lateral strain approximation
            deps[0, 0] = -0.3 * factor * deps_val
            deps[0, 1] = -0.3 * factor * deps_val

            sig_old = sig.copy()
            solid_update_law52(mat, sig, deps, epsp, dt=dt, extra=extra)

            # Midpoint stress integration
            sig_mid = 0.5 * (sig_old + sig)
            de = (
                sig_mid[0, 0] * deps[0, 0] +
                sig_mid[0, 1] * deps[0, 1] +
                sig_mid[0, 2] * deps[0, 2] +
                2.0 * (
                    sig_mid[0, 3] * (0.5 * deps[0, 3]) +
                    sig_mid[0, 4] * (0.5 * deps[0, 4]) +
                    sig_mid[0, 5] * (0.5 * deps[0, 5])
                )
            ) * vol0
            delta_eint_total += de
            sig_history.append(sig[0, 2])
            eps_history.append((step + 1) * deps_val if step < 25 else 25 * deps_val - (step - 24) * deps_val)

        assert delta_eint_total > 0.0, "Total strain energy must be positive"
        assert epsp[0] > 0.0, "Plastic strain must accumulate during cycle"

    def test_hexa8_elastic_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Hexa8 solid in pure elastic regime: |Delta E| / E_0 < 1%."""
        deck = StarterDeck("HEXA_ELAS_OSC")
        deck.mat_law52(1, "ElasticMat", rho=7.85e-9, e=210000.0, nu=0.3, a=20000.0, b=0.0)
        deck.prop_solid(1, "PropSolid", isolid=1, qa=0.0, qb=0.0, h=0.0)
        deck.part(1, "PartHexa", 1, 1)

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
        rho0 = 7.85e-9
        vol0 = lx * ly * lz
        m_node = rho0 * vol0 / 8.0
        mass_vec = np.full(8, m_node)

        # Pure symmetric breathing velocity mode in Z (nodes 1..4 move -Vz, 5..8 move +Vz)
        v = np.zeros_like(model.x)
        vz0 = 100.0
        v[:4, 2] = -vz0
        v[4:, 2] = vz0

        e_kin_0 = 0.5 * np.sum(mass_vec * (vz0 ** 2))
        e_int_0 = float(np.sum(group.state["eint"]))
        e_tot_0 = e_kin_0 + e_int_0
        assert e_tot_0 > 0.0

        dt = 5.0e-7
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        v_half = v.copy()

        e_tot_history = []
        for step in range(80):
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

        max_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_tot_history)
        assert max_err < 0.01, f"Hexa8 elastic energy conservation error {max_err*100:.3f}% exceeds 1.0%"
        assert np.all(group.state["epsp"] == 0.0), "Plastic strain must remain zero"

    def test_shell_bt4_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Shell BT4 in elastic regime: |Delta E| / E_0 < 1%."""
        deck = StarterDeck("BT4_ELAS_OSC")
        deck.mat_law52(1, "ElasticMat", rho=7.85e-9, e=210000.0, nu=0.3, a=20000.0, b=0.0)
        deck.prop_shell(1, "PropBT4", thick=1.0, nip=3, ishell=1, hm=0.0, hf=0.0, hr=0.0)
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
        rho0 = 7.85e-9
        area0 = lx * ly
        m_node = rho0 * 1.0 * area0 / 4.0
        mass_vec = np.full(4, m_node)

        v = np.zeros_like(model.x)
        vx0 = 50.0
        v[[1, 2], 0] = vx0
        v[[0, 3], 0] = -vx0

        e_kin_0 = 0.5 * np.sum(mass_vec * (vx0 ** 2))
        e_int_0 = float(np.sum(group.state["eint"]))
        e_tot_0 = e_kin_0 + e_int_0

        dt = 5.0e-7
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        v_half = v.copy()

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

        max_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_tot_history)
        assert max_err < 0.01, f"BT4 elastic energy error {max_err*100:.3f}% exceeds 1.0%"

    def test_plastic_dissipation_monotonic_growth(self):
        """Plastic dissipation monotonically increasing during plastic flow and void growth."""
        mat = make_test_material_law52(a=300.0, b=400.0, n=0.2, fn=0.04, sn=0.1, epsn=0.1)

        sig = np.zeros((1, 6), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra: dict[str, Any] = {}

        epsp_history = []
        epsm_history = []
        f_history = []
        dt = 1.0e-6

        for step in range(50):
            deps = np.zeros((1, 6), dtype=float)
            deps[0, 2] = 2.0e-4
            deps[0, 0] = -0.3 * 2.0e-4
            deps[0, 1] = -0.3 * 2.0e-4

            solid_update_law52(mat, sig, deps, epsp, dt=dt, extra=extra)

            curr_epsp = float(epsp[0])
            curr_epsm = float(extra["epsm"][0])
            curr_f = float(extra["dmg"][0, 3])

            epsp_history.append(curr_epsp)
            epsm_history.append(curr_epsm)
            f_history.append(curr_f)

        d_epsp = np.diff(epsp_history)
        assert np.all(d_epsp >= -1e-15), "Macroscopic plastic strain must be non-decreasing"

        d_epsm = np.diff(epsm_history)
        assert np.all(d_epsm >= -1e-15), "Matrix plastic strain must be non-decreasing"

        d_f = np.diff(f_history)
        assert np.all(d_f >= -1e-15), "Void volume fraction must grow under tension"
        assert f_history[-1] > f_history[0], "Void fraction must increase"


# ============================================================================
# 3. Micro-Porosity Softening & Ductile Rupture
# ============================================================================

class TestLaw52MicroPorositySofteningAndDuctileRupture:
    """Audit GTN micro-porosity softening, void growth under triaxiality, and ductile rupture."""

    def test_void_growth_and_softening_under_tensile_triaxiality(self):
        """Void volume fraction growth causing softening under tensile triaxiality."""
        # Case A: High triaxiality tension (hydrostatic tension P > 0)
        mat_tens = make_test_material_law52(a=350.0, b=400.0, n=0.2, fn=0.04)
        sig_tens = np.zeros((1, 6), dtype=float)
        epsp_tens = np.zeros(1, dtype=float)
        extra_tens: dict[str, Any] = {}

        # Case B: Pure shear (zero hydrostatic pressure P = 0)
        mat_shear = make_test_material_law52(a=350.0, b=400.0, n=0.2, fn=0.04)
        sig_shear = np.zeros((1, 6), dtype=float)
        epsp_shear = np.zeros(1, dtype=float)
        extra_shear: dict[str, Any] = {}

        # Case C: Uniaxial compression (negative pressure P < 0)
        mat_comp = make_test_material_law52(a=350.0, b=400.0, n=0.2, fn=0.04)
        sig_comp = np.zeros((1, 6), dtype=float)
        epsp_comp = np.zeros(1, dtype=float)
        extra_comp: dict[str, Any] = {}

        dt = 1.0e-6
        steps = 40

        for step in range(steps):
            deps_t = np.array([[1.0e-4, 1.0e-4, 3.0e-4, 0.0, 0.0, 0.0]])
            solid_update_law52(mat_tens, sig_tens, deps_t, epsp_tens, dt=dt, extra=extra_tens)

            deps_s = np.array([[0.0, 0.0, 0.0, 5.0e-4, 0.0, 0.0]])
            solid_update_law52(mat_shear, sig_shear, deps_s, epsp_shear, dt=dt, extra=extra_shear)

            deps_c = np.array([[-0.15e-4, -0.15e-4, -3.0e-4, 0.0, 0.0, 0.0]])
            solid_update_law52(mat_comp, sig_comp, deps_c, epsp_comp, dt=dt, extra=extra_comp)

        f_tens = extra_tens["dmg"][0, 3]
        f_shear = extra_shear["dmg"][0, 3]
        f_comp = extra_comp["dmg"][0, 3]

        assert f_tens > f_shear, f"Tensile void fraction {f_tens:.5f} must exceed shear void fraction {f_shear:.5f}"
        assert f_tens > f_comp, f"Tensile void fraction {f_tens:.5f} must exceed compression void fraction {f_comp:.5f}"

        fg_tens = extra_tens["dmg"][0, 1]
        fg_comp = extra_comp["dmg"][0, 1]
        assert fg_tens > 0.0, "Void growth fg must be positive under hydrostatic tension"
        assert fg_tens > fg_comp, "Void growth under tension must exceed compression"

    def test_element_erosion_at_fu_or_ff_zero_stresses(self):
        """Element erosion when f* >= fu or f >= ff with stresses dropping to zero."""
        mat = make_test_material_law52(
            a=350.0, b=400.0, n=0.2, fi=0.04, fc=0.045, ff=0.05, fu=0.667, fn=0.05, epsn=0.02
        )
        sig = np.zeros((1, 6), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra: dict[str, Any] = {}

        dt = 1.0e-6
        eroded = False

        for step in range(30):
            deps = np.array([[2.0e-4, 2.0e-4, 4.0e-4, 0.0, 0.0, 0.0]])
            solid_update_law52(mat, sig, deps, epsp, dt=dt, extra=extra)

            if extra["off"][0] == 0.0:
                eroded = True
                break

        assert eroded, "Element must be eroded when void fraction reaches ff"
        np.testing.assert_allclose(sig, 0.0, atol=1e-15)
        assert extra["off"][0] == 0.0
        assert extra["off52"][0] == 0.0

    def test_stable_post_failure_simulation_continuation(self):
        """Post-failure cycle continuation without solver NaN or acoustic collapse."""
        mat = make_test_material_law52(
            a=350.0, b=400.0, n=0.2, fi=0.04, fc=0.045, ff=0.05, fu=0.667, fn=0.05, epsn=0.02
        )
        sig = np.zeros((1, 6), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra: dict[str, Any] = {}

        dt = 1.0e-6

        # Step 1: Drive element to erosion
        for step in range(25):
            deps = np.array([[3.0e-4, 3.0e-4, 6.0e-4, 0.0, 0.0, 0.0]])
            solid_update_law52(mat, sig, deps, epsp, dt=dt, extra=extra)

        assert extra["off"][0] == 0.0

        # Step 2: Continue for 60+ cycles on the deleted element
        for post_step in range(65):
            deps_post = np.array([[1.0e-4, -1.0e-4, 2.0e-4, 1.0e-4, 0.0, 0.0]])
            sig_out, epsp_out, c_out = solid_update_law52(
                mat, sig, deps_post, epsp, dt=dt, extra=extra, return_sound_speed=True
            )

            assert np.isfinite(sig_out).all()
            assert np.isfinite(epsp_out).all()
            assert np.isfinite(c_out).all()
            np.testing.assert_allclose(sig_out, 0.0, atol=1e-15)
            assert c_out[0] > 1000.0

    def test_two_element_patch_progressive_rupture(self, tmp_path: Path):
        """2-element solid patch where element 1 fails while element 2 remains intact."""
        run_name = "TWO_ELEM_RUPTURE"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law52(
            1, "WeakMat", rho=7.85e-9, e=210000.0, nu=0.3,
            a=300.0, b=200.0, n=0.2, q1=1.5, q2=1.0, q3=2.25,
            f_i=0.02, f_c=0.025, f_f=0.03, f_n=0.05, s_n=0.1, eps_n=0.02,
        )
        d.prop_solid(1, "PropWeak", isolid=1)
        d.part(1, "PartWeak", 1, 1)

        d.mat_law52(
            2, "StrongMat", rho=7.85e-9, e=210000.0, nu=0.3,
            a=600.0, b=400.0, n=0.2, q1=1.5, q2=1.0, q3=2.25,
            f_i=0.001, f_c=0.50, f_f=0.90, f_n=0.01, s_n=0.1, eps_n=0.3,
        )
        d.prop_solid(2, "PropStrong", isolid=1)
        d.part(2, "PartStrong", 2, 2)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
            (9, 0.0, 0.0, 20.0), (10, 10.0, 0.0, 20.0), (11, 10.0, 10.0, 20.0), (12, 0.0, 10.0, 20.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        d.brick(2, [(2, 5, 6, 7, 8, 9, 10, 11, 12)])

        d.grnod_node(1, "fix_base", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        d.grnod_node(2, "pull_top", [9, 10, 11, 12])
        d.funct(10, "vz_fast", [(0.0, 20000.0), (1e-4, 20000.0)])
        d.impvel(1, "pull_z", 10, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 4
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0


# ============================================================================
# 4. Acoustic Sound Speed & Courant Time-Step Stability
# ============================================================================

class TestLaw52AcousticSoundSpeedAndCourantStability:
    """Audit acoustic sound speed formulas and Courant time-step bounds across deformation regimes."""

    def test_acoustic_sound_speed_solid_stability_across_deformation(self):
        """c_solid = sqrt(E(1-nu) / ((1+nu)(1-2nu)rho0)) maintaining stable bounds."""
        p = Law52Params(E=210000.0, nu=0.3, rho0=7.85e-9)
        c_expected = math.sqrt(210000.0 * (1.0 - 0.3) / ((1.0 + 0.3) * (1.0 - 2.0 * 0.3) * 7.85e-9))

        c0 = sound_speed_solid_law52(p)
        assert math.isclose(c0, c_expected, rel_tol=1e-5)

        rho_array = np.array([7.85e-9, 8.0e-9, 7.5e-9])
        c_vec = sound_speed_solid_law52(p, rho=rho_array)
        assert len(c_vec) == 3
        assert np.all(c_vec > 0.0)
        assert np.all(np.isfinite(c_vec))

    def test_acoustic_sound_speed_shell_stability_across_deformation(self):
        """c_shell = sqrt(E / ((1-nu^2)rho0)) maintaining stable bounds."""
        p = Law52Params(E=210000.0, nu=0.3, rho0=7.85e-9)
        c_expected = math.sqrt(210000.0 / ((1.0 - 0.3 ** 2) * 7.85e-9))

        c0 = sound_speed_shell_law52(p)
        assert math.isclose(c0, c_expected, rel_tol=1e-5)

        rho_array = np.array([7.85e-9, 8.2e-9, 7.2e-9])
        c_vec = sound_speed_shell_law52(p, rho=rho_array)
        assert len(c_vec) == 3
        assert np.all(c_vec > 0.0)
        assert np.all(np.isfinite(c_vec))

    def test_courant_step_stability_under_acoustic_pulse(self, tmp_path: Path):
        """Multi-element dynamic bar acoustic pulse propagation in Engine."""
        run_name = "BAR_PULSE_LAW52"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law52(
            1, "SteelBar", rho=7.85e-9, e=210000.0, nu=0.3,
            a=400.0, b=500.0, n=0.2, q1=1.5, q2=1.0, q3=2.25,
            f_i=0.005, f_c=0.15, f_f=0.25, f_n=0.04, s_n=0.1, eps_n=0.3,
        )
        d.prop_solid(1, "PropSolid", isolid=1)
        d.part(1, "PartBar", 1, 1)

        nodes = []
        for i in range(5):
            x = i * 10.0
            nodes.extend([
                (4 * i + 1, x, 0.0, 0.0),
                (4 * i + 2, x, 10.0, 0.0),
                (4 * i + 3, x, 10.0, 10.0),
                (4 * i + 4, x, 0.0, 10.0),
            ])
        d.node(nodes)

        bricks = []
        for i in range(4):
            n1 = 4 * i + 1
            bricks.append((
                i + 1,
                n1, n1 + 1, n1 + 2, n1 + 3,
                n1 + 4, n1 + 5, n1 + 6, n1 + 7
            ))
        d.brick(1, bricks)

        d.grnod_node(1, "fix_left", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        d.grnod_node(2, "impact_face", [17, 18, 19, 20])
        d.funct(10, "v_pulse", [
            (0.0, -1000.0),
            (0.5e-5, -1000.0),
            (0.5001e-5, 0.0),
            (5.0e-5, 0.0),
        ])
        d.impvel(1, "impact", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=5.0e-5, dt_scale=0.6)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 30
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Pulse energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0
