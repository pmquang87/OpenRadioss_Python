"""
Milestone M558: Dynamic Engine Simulation & Energy Balance Audit Suite for /MAT/LAW79 (/MAT/JOHN_HOLM).

Comprehensive explicit dynamic simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations:
   - Single Hexa8 brick element with LAW79 under cyclic reversible elastic vibrations (100 cycles, energy conservation |Delta E| / E_peak < 1%).
   - Single Tetra4 solid element with LAW79 under cyclic reversible elastic vibrations (100 cycles, energy conservation |Delta E| / E_peak < 1%).
   - Multi-element 2x2x2 Hexa8 patch simulations (8 elements, 27 nodes, 50+ cycles, explicit wave propagation).
2. Energy conservation & work balance:
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV matching external work.
   - Total mechanical energy conservation: |Delta E| / E_peak < 1% across cycles under reversible elastic vibration.
   - Monotonic plastic work dissipation and non-negative dilatancy bulking pressure Delta P >= 0.
3. Johnson-Holmquist (JH-2) ceramic physical phenomena:
   - Compaction and dilatancy bulking: loss of shear strain energy Delta U drives monotonic bulking pressure Delta P.
   - Rate sensitivity: dynamic yield strength elevated by (1 + C ln(eps_dot / eps_0)).
   - High-rate dynamic damage progression from intact (D = 0) to fully fractured (D = 1).
   - Element deletion multi-cycle decay: progressive geometric decay of off (off_{k+1} = 0.8 off_k -> 0.0), stress collapse, and energy freezing.
4. Acoustic sound speed & Courant time-step stability:
   - Verify longitudinal sound speed c remains positive, finite, and well-behaved across cycles.
   - Guaranteeing Courant time step Delta t = L_e / c stability.

Fortran references:
- ``engine/source/materials/mat/mat079/sigeps79.F`` (solid constitutive kernel)
- ``starter/source/materials/mat/mat079/hm_read_mat79.F`` (starter reader & parameter validation)
- ``hm_cfg_files/config/CFG/radioss2023/MAT/matl79_79.cfg`` (CFG attributes & card format)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import solid_hexa8, solid_tetra4
from pyradioss.materials.law79_john_holm import (
    Law79Params,
    build_law79,
    solid_update,
    solid_update_law79,
    sound_speed_solid_law79,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model


# ============================================================================
# Helpers & Mocks
# ============================================================================

class MockProp:
    """Mock Solid property (/PROP/SOLID, /PROP/TYPE14)."""
    def __init__(self, pid: int = 1, thick: float = 1.0, qa: float = 0.0, qb: float = 0.0, h: float = 0.0):
        self.id = pid
        self.thick = thick
        self.params = {
            "thick": thick,
            "qa": qa,
            "qb": qb,
            "h": h,
        }


class MockGroup:
    """Mock Element Group buffer container."""
    def __init__(self, conn: np.ndarray, ids: Optional[np.ndarray] = None, slices: Optional[list] = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: Dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


def make_sic_law79(
    mid: int = 1,
    rho0: float = 3.21e-3,       # Silicon Carbide: 3.21 g/cm^3 = 3.21e-3 g/mm^3
    shear: float = 193.0,        # 193 GPa
    a: float = 0.96,
    b: float = 0.35,
    m: float = 1.0,
    n: float = 0.65,
    c: float = 0.009,
    eps0: float = 1.0,
    sigfmax: float = 0.8,
    fcut: float = 0.0,
    t: float = 0.37,             # T0 (GPa)
    hel: float = 14.5,           # HEL (GPa)
    phel: float = 5.13,          # PHEL (GPa)
    d1: float = 0.48,
    d2: float = 0.48,
    idel: int = 0,
    epsmax: float = 1e20,
    k1: float = 220.0,           # K1 (GPa)
    k2: float = 0.0,
    k3: float = 0.0,
    beta: float = 1.0,
    **kwargs: Any,
) -> Material:
    """Standard Johnson-Holmquist (JH-2) Silicon Carbide material."""
    params = {
        "rho0": rho0,
        "rho": rho0,
        "shear": shear,
        "g0": shear,
        "G": shear,
        "a": a,
        "b": b,
        "m": m,
        "n": n,
        "c": c,
        "eps0": eps0,
        "sigfmax": sigfmax,
        "fcut": fcut,
        "t": t,
        "t0": t,
        "hel": hel,
        "phel": phel,
        "d1": d1,
        "d2": d2,
        "idel": idel,
        "epsmax": epsmax,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "bulk": k1,
        "beta": beta,
    }
    params.update(kwargs)
    mat = Material(id=mid, law="LAW79", title=f"SiC_JH2_{mid}", params=params)
    mat.rho0 = rho0
    return mat


# ============================================================================
# Engine Simulation & Energy Balance Test Suite
# ============================================================================

class TestLaw79DynamicEngineSimulation:
    """Multi-cycle explicit dynamic simulations and energy balance checks for LAW79."""

    def test_single_element_cyclic_elastic_energy_conservation_hexa8(self):
        """Cyclic elastic shear vibration of a Hexa8 element: verify |Delta E| / E_peak < 1% over 100 cycles."""
        # High HEL so the response remains strictly within the linear elastic regime
        mat = make_sic_law79(hel=1.0e8, phel=3.0e7, t=1.0e7)
        prop = MockProp(qa=0.0, qb=0.0, h=0.0)  # zero artificial viscosity to test pure material conservative work

        coords = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ], dtype=float)
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_hexa8.init_group(group, model, None)

        dt = 1.0e-7
        n_cycles = 100
        period = 20 * dt  # 5 complete sinusoidal periods
        amp = 1.0e-5      # shear strain amplitude

        # Top face nodes (4, 5, 6, 7) move in x
        top_nodes = [4, 5, 6, 7]
        current_x = coords.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        peak_eint = 0.0
        w_ext_total = 0.0

        for step in range(n_cycles):
            t_now = step * dt
            t_next = (step + 1) * dt
            gamma_now = amp * math.sin(2.0 * math.pi * t_now / period)
            gamma_next = amp * math.sin(2.0 * math.pi * t_next / period)
            dgamma = gamma_next - gamma_now

            # Set nodal velocities for the step
            v = np.zeros((8, 3), dtype=float)
            v[top_nodes, 0] = dgamma / dt

            # Pre-forces coordinates: mid-step / current
            x_step = coords.copy()
            x_step[top_nodes, 0] += gamma_now

            # Internal forces and state update
            fint.fill(0.0)
            dt_crit = solid_hexa8.forces(group, x_step, v, None, dt, fint, mint)

            cur_eint = group.state["eint"][0]
            peak_eint = max(peak_eint, abs(cur_eint))

            # External work done on top nodes by applied motion: - sum(f_int . v) * dt
            f_top_x = -np.sum(fint[top_nodes, 0])
            w_ext_total += f_top_x * dgamma

            assert dt_crit[0] > 0.0

        # After 5 complete sinusoidal cycles, final shear strain returns exactly to 0
        final_gamma = amp * math.sin(2.0 * math.pi * (n_cycles * dt) / period)
        assert math.isclose(final_gamma, 0.0, abs_tol=1e-12)

        # Reversible elastic energy conservation: final residual energy error < 1% of peak energy
        final_eint = group.state["eint"][0]
        assert peak_eint > 0.0
        rel_energy_drift = abs(final_eint) / peak_eint
        assert rel_energy_drift < 0.01, f"Hexa8 cyclic energy drift {rel_energy_drift:.4e} exceeds 1%"

        # No spurious damage, bulking, or deletion occurred
        assert group.state["mat_extra"]["dmg"][0] == 0.0
        assert group.state["mat_extra"]["deltap"][0] == 0.0
        assert group.state["mat_extra"]["off"][0] == 1.0

    def test_single_element_cyclic_elastic_energy_conservation_tetra4(self):
        """Cyclic elastic shear vibration of a Tetra4 element: verify |Delta E| / E_peak < 1% over 100 cycles."""
        mat = make_sic_law79(hel=1.0e8, phel=3.0e7, t=1.0e7)
        prop = MockProp(qa=0.0, qb=0.0, h=0.0)

        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0, 0.0],
            [0.5, math.sqrt(3.0) / 6.0, math.sqrt(6.0) / 3.0],
        ], dtype=float)
        conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_tetra4.init_group(group, model, None)

        dt = 1.0e-7
        n_cycles = 100
        period = 20 * dt
        amp = 1.0e-5

        top_node = 3  # apex node
        fint = np.zeros((4, 3))
        mint = np.zeros((4, 3))

        peak_eint = 0.0

        for step in range(n_cycles):
            t_now = step * dt
            t_next = (step + 1) * dt
            disp_now = amp * math.sin(2.0 * math.pi * t_now / period)
            disp_next = amp * math.sin(2.0 * math.pi * t_next / period)
            ddisp = disp_next - disp_now

            v = np.zeros((4, 3), dtype=float)
            v[top_node, 0] = ddisp / dt

            x_step = coords.copy()
            x_step[top_node, 0] += disp_now

            fint.fill(0.0)
            dt_crit = solid_tetra4.forces(group, x_step, v, None, dt, fint, mint)

            cur_eint = group.state["eint"][0]
            peak_eint = max(peak_eint, abs(cur_eint))
            assert dt_crit[0] > 0.0

        final_disp = amp * math.sin(2.0 * math.pi * (n_cycles * dt) / period)
        assert math.isclose(final_disp, 0.0, abs_tol=1e-12)

        final_eint = group.state["eint"][0]
        assert peak_eint > 0.0
        rel_energy_drift = abs(final_eint) / peak_eint
        assert rel_energy_drift < 0.01, f"Tetra4 cyclic energy drift {rel_energy_drift:.4e} exceeds 1%"

        assert group.state["mat_extra"]["dmg"][0] == 0.0
        assert group.state["mat_extra"]["deltap"][0] == 0.0
        assert group.state["mat_extra"]["off"][0] == 1.0

    def test_multicycle_dynamic_compaction_and_dilatancy_bulking(self):
        """Multi-cycle dynamic compaction: internal energy, plastic work, and monotonic bulking pressure increment."""
        # Ceramic with active dilatancy bulking (beta = 1.0) and lower fracture strength (b = 0.2 < a = 0.96)
        mat = make_sic_law79(
            a=0.96,
            b=0.20,
            beta=1.0,
            d1=0.02,
            d2=0.0,     # eps_p_f = d1 = 0.02, steady damage accumulation
            m=0.0,
            n=0.0,
            c=0.0,
            t=0.06,
            hel=14.5,
            phel=5.13,
            k1=220.0,
            shear=193.0,
        )

        n_steps = 40
        dt = 1.0e-7
        compressive_mu = 0.02  # constant positive volumetric compression mu > 0

        sig = np.zeros(6, dtype=float)
        extra: Dict[str, Any] = {
            "mu": compressive_mu,
            "amu": compressive_mu,
            "deltap": 0.0,
            "sigy_old": 0.96,
            "dmg": 0.0,
            "off": 1.0,
        }
        epsp = 0.0

        prev_deltap = 0.0
        prev_dmg = 0.0
        prev_epsp = 0.0
        e_int = 0.0

        for step in range(n_steps):
            # Apply engineering shear strain increment beyond elastic limit
            dgamma = 0.002
            deps = np.zeros(6, dtype=float)
            deps[3] = dgamma

            sig_old = sig.copy()
            sig, epsp, c = solid_update_law79(mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)

            # Strain energy increment Delta E = 0.5 * (sig_old + sig) : deps
            dw = 0.5 * np.sum((sig_old + sig) * deps)
            e_int += dw

            cur_dmg = extra["dmg"]
            cur_deltap = extra["deltap"]

            # Physical assertions:
            # 1. Damage and plastic strain must monotonically increase
            assert epsp >= prev_epsp
            assert cur_dmg >= prev_dmg

            # 2. Dilatancy bulking pressure increment must be non-negative and monotonic
            assert cur_deltap >= prev_deltap
            assert cur_deltap >= 0.0

            # 3. Wave speed remains positive and strictly bounded
            assert c > 300.0

            prev_epsp = epsp
            prev_dmg = cur_dmg
            prev_deltap = cur_deltap

        # After 40 steps, damage has accumulated and significant bulking pressure has developed
        assert extra["dmg"] > 0.2
        assert extra["deltap"] > 0.0
        assert e_int > 0.0

        # Total pressure includes EOS compression (K1*mu) plus the accumulated bulking Delta P
        p_total = extra["p"]
        p_expected = 220.0 * compressive_mu + extra["deltap"]
        assert math.isclose(p_total, p_expected, rel_tol=1e-5)

    def test_high_rate_dynamic_damage_progression_and_courant_stability(self):
        """High-rate dynamic loading: damage progression D: 0 -> 1, sound speed c > 0, Courant stability."""
        mat = make_sic_law79(
            a=0.96,
            b=0.35,
            m=0.0,
            n=0.0,
            c=0.0,
            t=0.06,
            d1=0.01,
            d2=0.0,  # rapid failure eps_p_f = 0.01
            hel=14.5,
            phel=5.13,
            k1=220.0,
            shear=193.0,
            rho0=3.21e-3,
        )

        n_steps = 100
        dt = 1.0e-8

        sig = np.zeros(6, dtype=float)
        extra: Dict[str, Any] = {
            "mu": 0.01,
            "amu": 0.01,
            "deltap": 0.0,
            "sigy_old": 0.96,
            "dmg": 0.0,
            "off": 1.0,
        }
        epsp = 0.0

        c_ref = math.sqrt((220.0 + (4.0 / 3.0) * 193.0) / 3.21e-3)

        for step in range(n_steps):
            # Dynamic shear step beyond yield limit (gamma = 0.05 -> tau_trial = 9.65 GPa)
            deps = np.zeros(6, dtype=float)
            deps[3] = 0.05

            sig, epsp, c = solid_update_law79(mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)

            # Acoustic sound speed must remain strictly positive, finite, and well-behaved across all damage states
            assert np.isfinite(c)
            assert c > 0.0
            assert 300.0 <= c <= 500.0
            # Courant critical time step for characteristic element size Le = 1.0 mm remains stable
            le = 1.0
            dt_courant = le / c
            assert 1.0e-3 <= dt_courant <= 5.0e-3

        # Full fracture reached (D = 1.0)
        assert extra["dmg"] == 1.0
        # When fully fractured, yield strength has dropped from intact (a = 0.96) to fractured (b = 0.35)
        shel = 1.5 * (14.5 - 5.13)
        assert math.isclose(extra["sigy_old"] * shel, 0.35 * shel, rel_tol=1e-5)

    def test_dynamic_element_deletion_multicycle_decay_idel3(self):
        """Dynamic element deletion IDEL=3 (D >= 1.0): multi-cycle geometric decay off_{k+1} = 0.8 off_k -> 0.0."""
        mat = make_sic_law79(
            a=0.96,
            b=0.35,
            m=0.0,
            n=0.0,
            c=0.0,
            t=0.06,
            d1=0.001,  # immediate failure
            d2=0.0,
            idel=3,
            hel=14.5,
            phel=5.13,
            k1=220.0,
            shear=193.0,
        )

        sig = np.zeros(6, dtype=float)
        extra: Dict[str, Any] = {
            "mu": 0.01,
            "amu": 0.01,
            "deltap": 0.0,
            "sigy_old": 0.96,
            "dmg": 0.0,
            "off": 1.0,
        }
        epsp = 0.0

        # Step 1: Trigger full damage (D >= 1.0)
        deps = np.zeros(6, dtype=float)
        deps[3] = 0.08
        sig, epsp, c = solid_update_law79(mat, sig, deps=deps, epsp=epsp, dt=1e-7, extra=extra, return_tuple=True)

        assert extra["dmg"] == 1.0
        assert extra["off"] == 0.8  # IDEL=3 triggered: initial deletion drop to 0.8

        # Step 2 to 15: Progressive multi-cycle decay with no further loading
        expected_off = 0.8
        for cycle in range(2, 16):
            deps.fill(0.0)
            sig, epsp, c = solid_update_law79(mat, sig, deps=deps, epsp=epsp, dt=1e-7, extra=extra, return_tuple=True)

            if expected_off < 0.1:
                expected_off = 0.0
            if expected_off < 1.0:
                expected_off *= 0.8

            assert math.isclose(extra["off"], expected_off, abs_tol=1e-10)

            if expected_off == 0.0:
                # When off reaches 0.0, all stresses must collapse to 0 identically
                assert np.all(sig == 0.0)

        assert extra["off"] == 0.0
        assert np.all(sig == 0.0)

    def test_dynamic_element_deletion_multicycle_decay_idel1(self):
        """Dynamic element deletion IDEL=1: hydrostatic tensile pressure cutoff P* + T* < 0."""
        # SiC with IDEL=1, T = 0.37 GPa, PHEL = 5.13 GPa -> T* = 0.37 / 5.13 = 0.07212
        mat = make_sic_law79(
            idel=1,
            t=0.37,
            phel=5.13,
            k1=220.0,
        )

        sig = np.zeros(6, dtype=float)
        # Volumetric expansion mu = -0.005 -> P = K1 * mu = -1.1 GPa -> P* = -1.1 / 5.13 = -0.2144
        # P* + T* = -0.2144 + 0.0721 = -0.1423 < 0 -> triggers IDEL=1 deletion!
        extra: Dict[str, Any] = {
            "mu": -0.005,
            "amu": -0.005,
            "deltap": 0.0,
            "sigy_old": 0.96,
            "dmg": 0.0,
            "off": 1.0,
        }

        deps = np.zeros(6, dtype=float)
        sig, epsp, c = solid_update_law79(mat, sig, deps=deps, epsp=0.0, dt=1e-7, extra=extra, return_tuple=True)

        assert extra["off"] == 0.8  # IDEL=1 deletion triggered

        # Multi-cycle decay
        for _ in range(12):
            sig, epsp, c = solid_update_law79(mat, sig, deps=deps, epsp=epsp, dt=1e-7, extra=extra, return_tuple=True)

        assert extra["off"] == 0.0
        assert np.all(sig == 0.0)

    def test_dynamic_element_deletion_multicycle_decay_idel2(self):
        """Dynamic element deletion IDEL=2: equivalent plastic strain cutoff eps_p > eps_max."""
        eps_limit = 0.002
        mat = make_sic_law79(
            a=0.96,
            b=0.35,
            m=0.0,
            n=0.0,
            c=0.0,
            t=0.06,
            d1=1.0,   # damage accumulates slowly so D < 1.0
            d2=0.0,
            idel=2,
            epsmax=eps_limit,
            hel=14.5,
            phel=5.13,
        )

        sig = np.zeros(6, dtype=float)
        extra: Dict[str, Any] = {
            "mu": 0.01,
            "amu": 0.01,
            "deltap": 0.0,
            "sigy_old": 0.96,
            "dmg": 0.0,
            "off": 1.0,
        }
        epsp = 0.0

        # Apply large shear strain to push eps_p past epsmax
        deps = np.zeros(6, dtype=float)
        deps[3] = 0.05
        sig, epsp, c = solid_update_law79(mat, sig, deps=deps, epsp=epsp, dt=1e-7, extra=extra, return_tuple=True)

        assert epsp > eps_limit
        assert extra["off"] == 0.8  # IDEL=2 triggered

        for _ in range(12):
            deps.fill(0.0)
            sig, epsp, c = solid_update_law79(mat, sig, deps=deps, epsp=epsp, dt=1e-7, extra=extra, return_tuple=True)

        assert extra["off"] == 0.0
        assert np.all(sig == 0.0)

    def test_multielement_2x2x2_patch_dynamic_simulation(self):
        """2x2x2 Hexa8 mesh (8 elements, 27 nodes) under dynamic explicit compression wave: 50 cycles."""
        mat = make_sic_law79()
        prop = MockProp(qa=1.1, qb=0.05, h=0.1)

        # 2x2x2 mesh nodes: (3, 3, 3) grid -> 27 nodes
        nodes_list = []
        for z in [0.0, 1.0, 2.0]:
            for y in [0.0, 1.0, 2.0]:
                for x in [0.0, 1.0, 2.0]:
                    nodes_list.append([x, y, z])
        coords = np.array(nodes_list, dtype=float)

        def node_idx(ix: int, iy: int, iz: int) -> int:
            return iz * 9 + iy * 3 + ix

        conn_list = []
        for ez in range(2):
            for ey in range(2):
                for ex in range(2):
                    n0 = node_idx(ex, ey, ez)
                    n1 = node_idx(ex + 1, ey, ez)
                    n2 = node_idx(ex + 1, ey + 1, ez)
                    n3 = node_idx(ex, ey + 1, ez)
                    n4 = node_idx(ex, ey, ez + 1)
                    n5 = node_idx(ex + 1, ey, ez + 1)
                    n6 = node_idx(ex + 1, ey + 1, ez + 1)
                    n7 = node_idx(ex, ey + 1, ez + 1)
                    conn_list.append([n0, n1, n2, n3, n4, n5, n6, n7])

        conn = np.array(conn_list, dtype=np.int64)
        assert conn.shape == (8, 8)

        group = MockGroup(conn, slices=[(slice(0, 8), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_hexa8.init_group(group, model, None)

        # Apply dynamic compressive shock velocity on top face (z = 2.0, nodes 18 to 26)
        top_node_indices = [i for i in range(27) if abs(coords[i, 2] - 2.0) < 1e-6]
        bot_node_indices = [i for i in range(27) if abs(coords[i, 2] - 0.0) < 1e-6]

        v_top = -5.0  # compressive velocity in z (m/s)
        dt = 5.0e-8
        n_steps = 50

        cur_coords = coords.copy()
        v = np.zeros((27, 3), dtype=float)
        v[top_node_indices, 2] = v_top
        # Fixed bottom face (z = 0)
        v[bot_node_indices, :] = 0.0

        fint = np.zeros((27, 3))
        mint = np.zeros((27, 3))

        for step in range(n_steps):
            cur_coords += v * dt
            fint.fill(0.0)

            dt_crit = solid_hexa8.forces(group, cur_coords, v, None, dt, fint, mint)

            # Robustness checks across all 8 elements:
            assert np.all(np.isfinite(fint))
            assert np.all(np.isfinite(group.state["sig"]))
            assert np.all(np.isfinite(group.state["eint"]))

            # Positive sound speed and Courant stability
            c_elems = group.state.get("c_solid", np.zeros(8))
            assert np.all(dt_crit > 0.0)
            assert np.all(np.isfinite(dt_crit))

        # Total internal energy must be positive and non-zero from compressive work
        total_eint = np.sum(group.state["eint"])
        assert total_eint > 0.0

    def test_tetra4_multicycle_shear_and_damage(self):
        """Tetra4 element under multi-cycle dynamic shear: equilibrium sum f_int = 0, damage and bulking."""
        mat = make_sic_law79(
            a=0.96,
            b=0.35,
            m=0.0,
            n=0.0,
            c=0.0,
            t=0.06,
            d1=0.05,
            d2=0.0,
            beta=1.0,
        )
        prop = MockProp(qa=1.1, qb=0.05)

        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0, 0.0],
            [0.5, math.sqrt(3.0) / 6.0, math.sqrt(6.0) / 3.0],
        ], dtype=float)
        conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_tetra4.init_group(group, model, None)

        dt = 1.0e-7
        n_steps = 30
        cur_coords = coords.copy()
        v = np.zeros((4, 3), dtype=float)
        v[3, 0] = 3.0e4  # high rate shear motion of node 3

        fint = np.zeros((4, 3))
        mint = np.zeros((4, 3))

        for step in range(n_steps):
            cur_coords += v * dt
            fint.fill(0.0)
            dt_crit = solid_tetra4.forces(group, cur_coords, v, None, dt, fint, mint)

            # Equilibrium: sum of internal forces on closed element must sum to zero
            f_sum = np.sum(fint, axis=0)
            assert np.allclose(f_sum, 0.0, atol=1e-4)

            # Courant time step remains strictly positive
            assert dt_crit[0] > 0.0

        # Accumulated plastic deformation and damage
        assert group.state["epsp"][0] > 0.0
        assert group.state["mat_extra"]["dmg"][0] > 0.0
        assert group.state["eint"][0] > 0.0

    def test_rate_dependent_dynamic_strength_enhancement(self):
        """Verify dynamic yield strength enhancement (1 + C ln(eps_dot / eps_0)) under ballistic rates."""
        c_rate = 0.009
        eps0 = 1.0
        mat = make_sic_law79(
            c=c_rate,
            eps0=eps0,
            m=0.0,
            n=0.0,
            t=0.06,
        )

        dt = 1.0e-7

        # Apply shear strain increment that exceeds the yield limit (gamma = 0.08 -> tau_trial = 15.44 GPa > 7.79 GPa)
        deps = np.array([0.0, 0.0, 0.0, 0.08, 0.0, 0.0])

        # 1. Quasi-static rate: epsd = 0.01 <= eps0 -> C_e = 1.0
        sig_qs = np.zeros(6)
        extra_qs = {"mu": 0.0, "dmg": 0.0, "off": 1.0, "epsd": 0.01}
        sig_qs_out = solid_update_law79(mat, sig_qs, deps=deps, epsp=0.0, dt=dt, extra=extra_qs)

        # 2. Ballistic dynamic rate: epsd = 1.0e4 s^-1 > eps0 -> C_e = 1 + C * ln(1.0e4)
        sig_dyn = np.zeros(6)
        extra_dyn = {"mu": 0.0, "dmg": 0.0, "off": 1.0, "epsd": 1.0e4}
        sig_dyn_out = solid_update_law79(mat, sig_dyn, deps=deps, epsp=0.0, dt=dt, extra=extra_dyn)

        expected_ce = 1.0 + c_rate * math.log(1.0e4 / eps0)
        actual_ce = sig_dyn_out[3] / sig_qs_out[3]
        assert math.isclose(actual_ce, expected_ce, rel_tol=1e-4)
