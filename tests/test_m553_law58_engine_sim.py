"""
Auditor 2C: Dynamic Engine Simulation & Energy Balance Auditor for M553 (/MAT/LAW58 /MAT/FABR_A /MAT/FABRIC_A).

Exhaustive dynamic engine simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations:
   - Shell BT4 (Ishell=1) with LAW58 fabric under cyclic tension, compression, and shear
   - Shell QEPH (Ishell=24) with LAW58 fabric under biaxial tension and Trellis shearing
   - Shell Tri3 (Ish3n=1) with LAW58 fabric under cyclic tension
   - Verified through Starter and Engine decks under cyclic velocity/displacement loading.
   - Asserts normal termination, >= 20 cycles, energy balance error |ERR| < 1.0%,
     positive internal strain energy accumulation (IE > 0), and external work consistency.
2. Energy conservation & work balance:
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV * dt
   - Midpoint trapezoidal work matches exact analytical strain energy in closed reversible cycles.
   - Free vibration of undamped 2D shell membrane (BT4 and QEPH, hm=0, hf=0, hr=0, df=0, ds=0):
     mechanical energy E_tot = E_kin + E_int is strictly conserved (|Delta E| / E_0 < 1.0%).
   - Viscous dissipation tracking when fiber damping Df, Ds or yarn friction tau_frot is active:
     dissipated energy monotonically increasing across cycles.
3. Trellis shear kinematics & locking:
   - Progressive Trellis shear showing smooth initial deformation (modulus G0) followed by
     sharp stiffening (modulus G_post) after lock angle phi_lock.
   - C0 stress continuity at lock angle boundary.
   - Dynamic cycle crossing lock angle threshold stably.
4. Fabric folding & zero-stress area:
   - Compression causing A/A0 <= A_rel resulting in zero-stress state without numerical instability.
   - Explicit engine simulation compressing through folding regime runs stably without crashing.
5. Acoustic sound speed & Courant time-step stability:
   - Sound speed c_shell = sqrt(max(Kc, Kt, G0) / rho0) remains strictly positive, finite,
     and stable across all deformation regimes (tension, compression, folding, Trellis locking).
   - Element stable time step Delta t = Le / c satisfies Courant criterion without crashing.
   - Acoustic pulse propagation along multi-element fabric patch in Engine.

Fortran references:
- starter/source/materials/mat/mat058/hm_read_mat58.F
- starter/source/materials/mat/mat058/cm58in3.F
- engine/source/materials/mat/mat058/sigeps58c.F
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials.law58_fabr_a import (
    Law58Params,
    FabricAMaterial,
    build_law58,
    crimp_interchange,
    extra_shapes,
    shell_membrane_tangent,
    shell_update_law58,
    solid_update_law58,
    sound_speed_shell_law58,
    tangent_law58_shell,
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
    tstop: float = 2.5e-3,
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


def make_test_material_law58(
    mid: int = 1,
    rho0: float = 1.0e-6,
    e1: float = 2000.0,
    e2: float = 1500.0,
    g0: float = 50.0,
    gt: float = 500.0,
    alphat: float = 30.0,
    df: float = 0.05,
    ds: float = 0.0,
    **kwargs: Any,
) -> Material:
    """Factory creating a valid /MAT/LAW58 (/MAT/FABR_A) Material instance."""
    params = {
        "e1": e1,
        "b1": kwargs.get("b1", 0.0),
        "e2": e2,
        "b2": kwargs.get("b2", 0.0),
        "flex": kwargs.get("flex", 1e-3),
        "g0": g0,
        "gt": gt,
        "alphat": alphat,
        "g5": kwargs.get("g5", 40.0),
        "df": df,
        "ds": ds,
        "gfrot": kwargs.get("gfrot", 20.0),
        "zero_stress": kwargs.get("zero_stress", 0.0),
        "arel": kwargs.get("arel", 0.0),
        "n1": kwargs.get("n1", 1),
        "n2": kwargs.get("n2", 1),
        "s1": kwargs.get("s1", 0.1),
        "s2": kwargs.get("s2", 0.1),
        "c4": kwargs.get("c4", 0.0),
        "c5": kwargs.get("c5", 0.0),
    }
    params.update(kwargs)
    mat = Material(id=mid, law=58, rho0=rho0, title="Fabric_LAW58", params=params)
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

class TestLaw58MultiCycleDynamicSimulations:
    """Audit multi-cycle explicit dynamic simulations across supported shell formulations."""

    def test_shell_bt4_cyclic_tension_compression_shear_engine(self, tmp_path: Path):
        """Shell BT4 (Ishell=1) with LAW58 fabric under cyclic tension, compression, and shear."""
        run_name = "SHELL_BT4_LAW58_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law58(
            mid=1, title="FabricBT4", rho=1.0e-6,
            e1=2500.0, e2=1800.0, g0=60.0, gi=600.0, alpha=32.0,
            df=0.02, ds=0.0, n1_warp=1, n2_weft=1, s1=0.1, s2=0.1,
        )
        d.prop_shell(1, "PropBT4", thick=1.0, nip=3, ishell=1)
        d.part(1, "PartBT4", 1, 1)

        # 2-element quad strip along X
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 20.0, 0.0, 0.0), (6, 20.0, 10.0, 0.0),
        ])
        d.shell(1, [
            (1, 1, 2, 3, 4),
            (2, 2, 5, 6, 3),
        ])

        # Boundary conditions: fix left edge (1, 4)
        d.grnod_node(1, "fix_edge", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Multi-axis cyclic loading: tension -> compression -> shear on right edge (5, 6)
        d.grnod_node(2, "load_edge", [5, 6])
        # Cyclic velocity in X: pull, push, pull
        d.funct(10, "vx_cycle", [
            (0.0, 15.0),
            (0.8e-3, 15.0),
            (0.8001e-3, -15.0),
            (1.6e-3, -15.0),
            (1.6001e-3, 15.0),
            (2.5e-3, 15.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

        # In-plane shear velocity in Y
        d.funct(11, "vy_cycle", [
            (0.0, 5.0),
            (1.2e-3, 5.0),
            (1.2001e-3, -5.0),
            (2.5e-3, -5.0),
        ])
        d.impvel(2, "shear_y", 11, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.5e-3, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20, f"Expected >= 20 cycles, got {state.cycle}"
        assert state.stop_reason == "", f"Abnormal engine termination: {state.stop_reason}"
        assert state.t > 0.0

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy balance error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal strain energy must accumulate"
        assert en["EW"] > 0.0, "External work must be positive"

    def test_shell_qeph_biaxial_tension_trellis_shearing_engine(self, tmp_path: Path):
        """Shell QEPH (Ishell=24): physical hourglass quad shell under biaxial tension and Trellis shear."""
        run_name = "SHELL_QEPH_LAW58_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law58(
            mid=1, title="FabricQEPH", rho=1.2e-6,
            e1=2200.0, e2=1600.0, g0=50.0, gi=500.0, alpha=30.0,
            df=0.03, ds=0.0, n1_warp=1, n2_weft=1, s1=0.1, s2=0.1,
        )
        d.prop_shell(1, "PropQEPH", thick=1.0, nip=3, ishell=24)
        d.part(1, "PartQEPH", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix node 1 completely
        d.grnod_node(1, "fix_corner", [1])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Nodes 2 and 3 pulled in X (warp tension)
        d.grnod_node(2, "pull_x_edge", [2, 3])
        d.funct(10, "vel_x", [
            (0.0, 12.0),
            (1.0e-3, 12.0),
            (1.0001e-3, -8.0),
            (2.5e-3, -8.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

        # Nodes 3 and 4 pulled in Y (weft tension) + shear
        d.grnod_node(3, "pull_y_edge", [3, 4])
        d.funct(11, "vel_y", [
            (0.0, 10.0),
            (1.2e-3, 10.0),
            (1.2001e-3, -6.0),
            (2.5e-3, -6.0),
        ])
        d.impvel(2, "pull_y", 11, "Y", 3)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.5e-3, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"QEPH energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_tri3_cyclic_engine_simulation(self, tmp_path: Path):
        """Shell Tri3 (Ish3n=1): 3-node C0 triangle shell with LAW58 fabric under cyclic loading."""
        run_name = "SHELL_TRI3_LAW58_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law58(
            mid=1, title="FabricTri3", rho=1.0e-6,
            e1=2400.0, e2=1700.0, g0=55.0, gi=450.0, alpha=28.0,
            df=0.04, ds=0.0,
        )
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
            (0.0, 15.0),
            (1.0e-3, 15.0),
            (1.0001e-3, -15.0),
            (2.0e-3, -15.0),
            (2.0001e-3, 10.0),
            (2.8e-3, 10.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.8e-3, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Tri3 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_bt4_cyclic_displacement_strain_reversal(self):
        """Verify BT4 element stress responses under cyclic tension, compression, and shear reversals."""
        coords = np.array([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        ])
        conn = np.array([[0, 1, 2, 3]])
        mat = make_test_material_law58(e1=2000.0, e2=1500.0, g0=50.0, df=0.0, ds=0.0)
        prop = MockProp(thick=1.0, nip=2)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model

        shell_bt4.init_group(group, model, None)
        fint = np.zeros_like(coords)
        mint = np.zeros_like(coords)

        # 1. Forward tension
        vel_fwd = np.zeros_like(coords)
        vel_fwd[[1, 2], 0] = 20.0
        dt = 1.0e-5
        curr_x = coords.copy()

        for _ in range(15):
            curr_x += vel_fwd * dt
            shell_bt4.forces(group, curr_x, vel_fwd, np.zeros_like(coords), dt, fint, mint)

        sig_fwd = group.state["sig"][0, :, 0].mean()
        assert sig_fwd > 0.0, f"Expected positive tension stress, got {sig_fwd}"

        # 2. Reverse compression
        vel_rev = -vel_fwd
        for _ in range(30):
            curr_x += vel_rev * dt
            shell_bt4.forces(group, curr_x, vel_rev, np.zeros_like(coords), dt, fint, mint)

        sig_rev = group.state["sig"][0, :, 0].mean()
        assert sig_rev < sig_fwd

        # 3. Cyclic in-plane shear
        vel_shear = np.zeros_like(coords)
        vel_shear[[2, 3], 0] = 25.0
        for _ in range(20):
            curr_x += vel_shear * dt
            shell_bt4.forces(group, curr_x, vel_shear, np.zeros_like(coords), dt, fint, mint)

        sig_xy = group.state["sig"][0, :, 2].mean()
        assert abs(sig_xy) > 0.0, "Shear stress must develop during shear kinematics"


# ============================================================================
# 2. Energy Conservation & Work Balance Ledger
# ============================================================================

class TestLaw58EnergyConservationAndWorkBalance:
    """Audit mechanical energy conservation, incremental work ledger, and viscous dissipation."""

    def test_strain_energy_work_ledger_accuracy(self):
        """Verify incremental work ledger Delta E_int = sum(sigma_mid : deps * V) matches closed loop."""
        mat = make_test_material_law58(e1=2400.0, e2=1800.0, g0=60.0, df=0.0, ds=0.0)
        extra: dict[str, Any] = {
            "eps58": np.zeros((1, 3)), "yc": np.zeros(1), "yt": np.zeros(1), "fn": np.zeros(1),
            "sigv_xy": np.zeros(1), "tan_phi": np.zeros(1), "sigi58": np.zeros((1, 3)), "t58": np.zeros(1),
            "area": np.array([100.0]), "thk": np.array([1.0]),
        }
        v0 = 100.0 * 1.0
        sig = np.zeros((1, 3))
        dt = 0.0  # static/reversible limit

        w_acc = 0.0
        n_steps = 40
        deps_fwd = np.array([[0.0002, 0.00015, 0.0001]])

        # Forward stretch
        for _ in range(n_steps):
            sig_old = sig.copy()
            sig, _ = shell_update_law58(mat, sig, deps_fwd, dt=dt, extra=extra)
            sig_mid = 0.5 * (sig_old + sig)
            d_work = (
                sig_mid[0, 0] * deps_fwd[0, 0]
                + sig_mid[0, 1] * deps_fwd[0, 1]
                + sig_mid[0, 2] * deps_fwd[0, 2]
            ) * v0
            w_acc += d_work

        assert w_acc > 0.0, "Strain energy must be positive after forward deformation"

        # Reverse back to origin
        deps_rev = -deps_fwd
        for _ in range(n_steps):
            sig_old = sig.copy()
            sig, _ = shell_update_law58(mat, sig, deps_rev, dt=dt, extra=extra)
            sig_mid = 0.5 * (sig_old + sig)
            d_work = (
                sig_mid[0, 0] * deps_rev[0, 0]
                + sig_mid[0, 1] * deps_rev[0, 1]
                + sig_mid[0, 2] * deps_rev[0, 2]
            ) * v0
            w_acc += d_work

        # In conservative regime, roundtrip work must return identically to zero
        assert abs(w_acc) < 1.0e-10, f"Residual work {w_acc} exceeds conservative threshold"
        assert np.linalg.norm(sig) < 1.0e-12, f"Residual stress {sig} must be zero at origin"

    def test_shell_undamped_free_oscillation_energy_conservation(self):
        """Shell BT4 undamped free vibration with LAW58 fabric: |Delta E| / E0 < 1.0%."""
        coords = np.array([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        ])
        conn = np.array([[0, 1, 2, 3]])
        mat = make_test_material_law58(e1=2000.0, e2=1500.0, g0=50.0, df=0.0, ds=0.0)
        prop = MockProp(thick=1.0, nip=1, hm=0.0, hf=0.0, hr=0.0)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model

        shell_bt4.init_group(group, model, None)
        rho0 = 1.0e-6
        m_node = rho0 * 100.0 * 1.0 / 4.0
        mass_vec = np.full(4, m_node)

        v = np.zeros_like(coords)
        v[[1, 2], 0] = 5.0
        v[[0, 3], 0] = -5.0

        dt = 1.0e-6
        fint = np.zeros_like(coords)
        mint = np.zeros_like(coords)

        curr_x = coords.copy()
        v_half = v.copy()
        e_kin_0 = 0.5 * np.sum(mass_vec[:, None] * (v ** 2))
        e_tot_0 = e_kin_0
        assert e_tot_0 > 0.0

        e_tot_history = []
        for step in range(100):
            curr_x += v_half * dt
            fint.fill(0.0)
            mint.fill(0.0)
            shell_bt4.forces(group, curr_x, v_half, np.zeros_like(coords), dt, fint, mint)

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
        assert rel_drift < 0.01, f"BT4 free oscillation energy deviation {rel_drift * 100:.4f}% exceeds 1.0%"

    def test_shell_qeph_undamped_free_oscillation_energy_conservation(self):
        """Shell QEPH undamped free vibration with LAW58 fabric: |Delta E| / E0 < 1.0%."""
        coords = np.array([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        ])
        conn = np.array([[0, 1, 2, 3]])
        mat = make_test_material_law58(e1=2000.0, e2=1500.0, g0=50.0, df=0.0, ds=0.0)
        prop = MockProp(thick=1.0, nip=1, ishell=24, dn=0.0)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model

        shell_qeph.init_group(group, model, None)
        rho0 = 1.0e-6
        m_node = rho0 * 100.0 * 1.0 / 4.0
        mass_vec = np.full(4, m_node)

        v = np.zeros_like(coords)
        v[[1, 2], 0] = 5.0
        v[[0, 3], 0] = -5.0

        dt = 1.0e-6
        fint = np.zeros_like(coords)
        mint = np.zeros_like(coords)

        curr_x = coords.copy()
        v_half = v.copy()
        e_kin_0 = 0.5 * np.sum(mass_vec[:, None] * (v ** 2))
        e_tot_0 = e_kin_0

        e_tot_history = []
        for step in range(100):
            curr_x += v_half * dt
            fint.fill(0.0)
            mint.fill(0.0)
            shell_qeph.forces(group, curr_x, v_half, np.zeros_like(coords), dt, fint, mint)

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
        assert rel_drift < 0.01, f"QEPH free oscillation energy deviation {rel_drift * 100:.4f}% exceeds 1.0%"

    def test_viscous_fiber_damping_dissipation_monotonicity(self):
        """Verify dissipated energy is strictly monotonically increasing when fiber damping Df > 0."""
        mat_df = make_test_material_law58(e1=2000.0, e2=1500.0, df=0.1, ds=0.0)
        extra = {
            "eps58": np.zeros((1, 3)), "yc": np.zeros(1), "yt": np.zeros(1), "fn": np.zeros(1),
            "sigv_xy": np.zeros(1), "tan_phi": np.zeros(1), "sigi58": np.zeros((1, 3)), "t58": np.zeros(1),
            "area": np.array([100.0]), "thk": np.array([1.0]), "ediss": np.zeros(1),
        }
        sig = np.zeros((1, 3))
        dt = 1.0e-5

        diss_history = []
        for k in range(35):
            dep_xx = 0.001 * math.sin(k * 0.4)
            dep_yy = -0.0003 * math.sin(k * 0.4)
            deps = np.array([[dep_xx, dep_yy, 0.0]])
            sig, _ = shell_update_law58(mat_df, sig, deps, dt=dt, extra=extra)
            diss_history.append(float(extra["ediss"][0]))

        for i in range(len(diss_history) - 1):
            assert diss_history[i + 1] >= diss_history[i] - 1e-15, (
                f"Fiber damping dissipation decreased at step {i}: {diss_history[i]} -> {diss_history[i+1]}"
            )
        assert diss_history[-1] > 0.0, "Positive viscous energy must be dissipated"

    def test_yarn_sliding_friction_dissipation_monotonicity(self):
        """Verify dissipated energy is monotonically increasing when yarn sliding friction is active."""
        mat_ds = make_test_material_law58(e1=2000.0, e2=1500.0, df=0.0, ds=0.25, gfrot=40.0)
        extra = {
            "eps58": np.zeros((1, 3)), "yc": np.zeros(1), "yt": np.zeros(1), "fn": np.zeros(1),
            "sigv_xy": np.zeros(1), "tan_phi": np.zeros(1), "sigi58": np.zeros((1, 3)), "t58": np.zeros(1),
            "area": np.array([100.0]), "thk": np.array([1.0]), "ediss": np.zeros(1),
        }
        dt = 1.0e-5

        deps_warp = np.array([[0.05, 0.0, 0.0]])
        sig, _ = shell_update_law58(mat_ds, np.zeros((1, 3)), deps_warp, dt=dt, extra=extra)
        assert extra["fn"][0] > 0.0, "Warp tension must generate contact normal force Fn > 0"

        diss_history = []
        for k in range(40):
            dep_xy = 0.08 * math.cos(k * 0.3)
            deps = np.array([[0.0, 0.0, dep_xy]])
            sig, _ = shell_update_law58(mat_ds, sig, deps, dt=dt, extra=extra)
            diss_history.append(float(extra["ediss"][0]))

        for i in range(len(diss_history) - 1):
            assert diss_history[i + 1] >= diss_history[i] - 1e-15, (
                f"Frictional dissipation decreased at step {i}: {diss_history[i]} -> {diss_history[i+1]}"
            )
        assert diss_history[-1] > 0.0, "Frictional sliding must dissipate energy"


# ============================================================================
# 3. Trellis Shear Kinematics & Locking
# ============================================================================

class TestLaw58TrellisShearKinematicsAndLocking:
    """Audit progressive Trellis shearing, locking angle transition, and C0 continuity."""

    def test_trellis_progressive_shear_and_lock_angle_transition(self):
        """Verify smooth initial shear (G0) followed by sharp stiffening (G_post) after phi_lock."""
        alphat = 30.0  # degrees -> tan_lock = tan(30 deg) = 0.57735
        g0 = 50.0
        gt = 500.0
        mat = make_test_material_law58(g0=g0, gt=gt, alphat=alphat, df=0.0, ds=0.0)

        tan_lock = math.tan(alphat * math.pi / 180.0)
        g_post = gt / (1.0 + tan_lock ** 2)
        gb = tan_lock * (g0 - g_post)

        # 1. Below lock angle: pure linear response with modulus G0
        angles_below = np.linspace(0.05, 0.45, 10)
        stresses_below = []
        for a in angles_below:
            extra = {
                "eps58": np.zeros((1, 3)), "yc": np.zeros(1), "yt": np.zeros(1), "fn": np.zeros(1),
                "sigv_xy": np.zeros(1), "tan_phi": np.zeros(1), "sigi58": np.zeros((1, 3)), "t58": np.zeros(1),
            }
            s, _ = shell_update_law58(mat, np.zeros((1, 3)), np.array([[0.0, 0.0, a]]), dt=0.0, extra=extra)
            stresses_below.append(float(s[0, 2]))

        slopes_below = np.diff(stresses_below) / np.diff(angles_below)
        assert np.allclose(slopes_below, g0, rtol=1e-3)

        # 2. Above lock angle: stiffened response with modulus G_post
        angles_above = np.linspace(0.65, 0.95, 10)
        stresses_above = []
        for a in angles_above:
            extra = {
                "eps58": np.zeros((1, 3)), "yc": np.zeros(1), "yt": np.zeros(1), "fn": np.zeros(1),
                "sigv_xy": np.zeros(1), "tan_phi": np.zeros(1), "sigi58": np.zeros((1, 3)), "t58": np.zeros(1),
            }
            s, _ = shell_update_law58(mat, np.zeros((1, 3)), np.array([[0.0, 0.0, a]]), dt=0.0, extra=extra)
            stresses_above.append(float(s[0, 2]))

        slopes_above = np.diff(stresses_above) / np.diff(angles_above)
        assert np.allclose(slopes_above, g_post, rtol=1e-3)
        assert g_post > g0 * 5.0, f"G_post ({g_post}) must be significantly higher than G0 ({g0})"

        # 3. Exact C0 continuity across lock angle boundary
        extra_minus = {
            "eps58": np.zeros((1, 3)), "yc": np.zeros(1), "yt": np.zeros(1), "fn": np.zeros(1),
            "sigv_xy": np.zeros(1), "tan_phi": np.zeros(1), "sigi58": np.zeros((1, 3)), "t58": np.zeros(1),
        }
        s_minus, _ = shell_update_law58(mat, np.zeros((1, 3)), np.array([[0.0, 0.0, tan_lock - 1e-6]]), dt=0.0, extra=extra_minus)

        extra_plus = {
            "eps58": np.zeros((1, 3)), "yc": np.zeros(1), "yt": np.zeros(1), "fn": np.zeros(1),
            "sigv_xy": np.zeros(1), "tan_phi": np.zeros(1), "sigi58": np.zeros((1, 3)), "t58": np.zeros(1),
        }
        s_plus, _ = shell_update_law58(mat, np.zeros((1, 3)), np.array([[0.0, 0.0, tan_lock + 1e-6]]), dt=0.0, extra=extra_plus)

        assert math.isclose(float(s_minus[0, 2]), float(s_plus[0, 2]), rel_tol=1e-4), (
            f"Shear stress discontinuity at lock angle: {s_minus[0, 2]} vs {s_plus[0, 2]}"
        )

    def test_trellis_shear_cyclic_engine_simulation(self, tmp_path: Path):
        """Engine simulation driving shell element across Trellis locking boundary dynamically."""
        run_name = "TRELLIS_LOCK_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law58(
            mid=1, title="FabricTrellis", rho=1.0e-6,
            e1=2000.0, e2=1500.0, g0=50.0, gi=600.0, alpha=25.0,
            df=0.02, ds=0.0,
        )
        d.prop_shell(1, "PropTrellis", thick=1.0, nip=3, ishell=1)
        d.part(1, "Part1", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix bottom edge
        d.grnod_node(1, "fix_bottom", [1, 2])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # High shear displacement across lock angle: vx on top edge
        d.grnod_node(2, "shear_top", [3, 4])
        d.funct(10, "shear_vx", [
            (0.0, 25.0),
            (1.0e-3, 25.0),
            (1.0001e-3, -25.0),
            (2.2e-3, -25.0),
        ])
        d.impvel(1, "shear_x", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.2e-3, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Trellis dynamic energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0


# ============================================================================
# 4. Fabric Folding & Zero-Stress Area
# ============================================================================

class TestLaw58FabricFoldingAndZeroStressArea:
    """Audit fabric folding, zero-stress area deactivation, and compression stability."""

    def test_folding_zero_stress_deactivation(self):
        """Verify normal and shear stresses vanish when A / A0 <= A_rel without sound speed collapse."""
        arel = 0.85
        mat = make_test_material_law58(e1=2000.0, e2=1500.0, g0=50.0, arel=arel)

        c_base = sound_speed_shell_law58(mat)
        assert c_base > 0.0

        extra = {
            "eps58": np.zeros((1, 3)), "yc": np.zeros(1), "yt": np.zeros(1), "fn": np.zeros(1),
            "sigv_xy": np.zeros(1), "tan_phi": np.zeros(1), "sigi58": np.zeros((1, 3)), "t58": np.zeros(1),
        }

        # Compressive biaxial strain: eps_xx = -0.25, eps_yy = -0.25 -> rel_area = exp(-0.5) = 0.6065 < 0.85
        deps_fold = np.array([[-0.25, -0.25, 0.1]])
        s_folded, _ = shell_update_law58(mat, np.zeros((1, 3)), deps_fold, dt=1.0e-5, extra=extra)

        # All membrane stresses must be zeroed in the folded region
        assert np.allclose(s_folded[0, :3], 0.0, atol=1e-10), f"Stresses must be zero in folded state: {s_folded}"

        # Sound speed must remain finite, positive, and unchanged
        c_folded = sound_speed_shell_law58(mat)
        assert c_folded > 0.0
        assert math.isclose(c_folded, c_base, rel_tol=1e-10)

    def test_folding_compression_engine_simulation(self, tmp_path: Path):
        """Explicit engine simulation compressing fabric through folding regime runs stably."""
        run_name = "FOLDING_ENGINE_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law58(
            mid=1, title="FabricFold", rho=1.0e-6,
            e1=2000.0, e2=1500.0, g0=50.0, arel=0.90,
        )
        d.prop_shell(1, "PropFold", thick=1.0, nip=3, ishell=1)
        d.part(1, "PartFold", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])

        d.grnod_node(1, "fix_edge", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Large compressive velocity causing area reduction past A_rel = 0.90
        d.grnod_node(2, "compress_edge", [2, 3])
        d.funct(10, "comp_vx", [
            (0.0, -15.0),
            (1.5e-3, -15.0),
            (1.5001e-3, 0.0),
            (2.5e-3, 0.0),
        ])
        d.impvel(1, "comp_x", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.5e-3, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Folding simulation energy error {en['ERR']}% exceeds 1.0%"


# ============================================================================
# 5. Acoustic Sound Speed & Courant Time-Step Stability
# ============================================================================

class TestLaw58SoundSpeedAndCourantStability:
    """Audit acoustic sound speed stability, positivity, and Courant time-step bounds."""

    def test_sound_speed_positivity_across_all_regimes(self):
        """Verify sound speed c_shell remains positive and finite across tension, compression, and shear."""
        mat = make_test_material_law58(
            rho0=1.2e-6, e1=2400.0, e2=1800.0, g0=60.0, gt=600.0, alphat=30.0, arel=0.85,
        )

        c0 = sound_speed_shell_law58(mat)
        assert c0 > 0.0
        assert math.isfinite(c0)

        # Regimes: large warp tension, large weft tension, large compression, high Trellis shear
        strain_states = [
            np.array([[0.20, 0.0, 0.0]]),    # High warp tension
            np.array([[0.0, 0.20, 0.0]]),    # High weft tension
            np.array([[-0.25, -0.25, 0.0]]), # High biaxial compression (folding)
            np.array([[0.0, 0.0, 0.85]]),    # High Trellis shear beyond lock angle
        ]

        for deps in strain_states:
            extra = {
                "eps58": np.zeros((1, 3)), "yc": np.zeros(1), "yt": np.zeros(1), "fn": np.zeros(1),
                "sigv_xy": np.zeros(1), "tan_phi": np.zeros(1), "sigi58": np.zeros((1, 3)), "t58": np.zeros(1),
            }
            s, _ = shell_update_law58(mat, np.zeros((1, 3)), deps, dt=1e-5, extra=extra)
            # Stresses must be finite (no NaNs or infinities)
            assert np.all(np.isfinite(s)), f"Non-finite stresses under strain {deps}: {s}"

            # Sound speed check
            c = sound_speed_shell_law58(mat)
            assert c > 0.0
            assert math.isfinite(c)

    def test_courant_timestep_bounds_and_pulse_propagation(self, tmp_path: Path):
        """Verify Courant condition dt <= Le / c and wave propagation across multi-element fabric strip."""
        run_name = "PULSE_LAW58_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        rho0 = 1.0e-6
        e1 = 2500.0
        e2 = 2000.0
        g0 = 50.0
        mat = make_test_material_law58(rho0=rho0, e1=e1, e2=e2, g0=g0)
        c_expected = sound_speed_shell_law58(mat)

        # 5-element strip: 5 elements of length 10.0 mm along X
        le = 10.0
        dt_courant = le / c_expected

        d = StarterDeck(run_name)
        d.mat_law58(
            mid=1, title="FabricPulse", rho=rho0,
            e1=e1, e2=e2, g0=g0, gi=500.0, alpha=30.0,
        )
        d.prop_shell(1, "PropPulse", thick=1.0, nip=3, ishell=1)
        d.part(1, "PartPulse", 1, 1)

        nodes = []
        for i in range(6):
            nodes.append((i * 2 + 1, i * 10.0, 0.0, 0.0))
            nodes.append((i * 2 + 2, i * 10.0, 10.0, 0.0))
        d.node(nodes)

        shells = []
        for i in range(5):
            n1 = i * 2 + 1
            n2 = (i + 1) * 2 + 1
            n3 = (i + 1) * 2 + 2
            n4 = i * 2 + 2
            shells.append((i + 1, n1, n2, n3, n4))
        d.shell(1, shells)

        # Fix far right end
        d.grnod_node(1, "fix_far", [11, 12])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Dynamic impulse at left end (nodes 1, 2)
        d.grnod_node(2, "impact_nodes", [1, 2])
        d.funct(10, "impulse_vx", [
            (0.0, 20.0),
            (2.0e-4, 20.0),
            (2.0001e-4, 0.0),
            (2.0e-3, 0.0),
        ])
        d.impvel(1, "impulse", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-3, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        # Verify element time steps satisfy Courant bound
        fint = np.zeros_like(eng_model.x)
        mint = np.zeros_like(eng_model.x)
        dt_step = shell_bt4.forces(eng_model.shells, eng_model.x, eng_model.v, eng_model.vr, 1.0e-6, fint, mint)
        assert np.all(dt_step <= dt_courant * 1.05), f"Element time step exceeded Courant bound {dt_courant}"
        assert np.all(dt_step > 0.0)

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0
        assert en["IE"] > 0.0
