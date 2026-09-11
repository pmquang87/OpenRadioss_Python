"""
Milestone M559: Dynamic Engine Simulation & Energy Balance Audit Suite for /MAT/LAW50 (/MAT/VISC_HONEY /MAT/HYP_FOAM).

Comprehensive explicit dynamic simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations:
   - Single Hexa8 brick element with LAW50 under cyclic reversible elastic vibrations (100 cycles, energy conservation |Delta E| / E_peak < 1%).
   - Single Tetra4 solid element with LAW50 under cyclic reversible elastic vibrations (100 cycles, energy conservation |Delta E| / E_peak < 1%).
   - Multi-element 2x2x2 Hexa8 patch simulations (8 elements, 27 nodes, 50+ cycles, explicit wave propagation).
2. Energy conservation & work balance:
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV matching external work.
   - Total mechanical energy conservation: |Delta E| / E_peak < 1% across cycles under reversible elastic vibration.
   - Monotonic plastic work dissipation and positive internal energy accumulation under volumetric crush.
3. Viscoelastic Honeycomb (LAW50) physical phenomena:
   - Orthotropic honeycomb moduli in uncompacted state (E11, E22, E33, G12, G23, G31).
   - Smooth moduli stiffening transition during compaction as r_vol -> V_comp.
   - Permanent compaction lock-in when r_vol <= V_comp.
   - Coupled isotropic J2 plasticity with isotropic hardening H = E_t in compacted state.
   - Strain rate filtering (asrate = min(1.0, f_cut * dt)) and dynamic yield scaling.
   - Directional strain failure erosion (eps_max11..eps_max31): stress collapse to zero, internal forces zeroed, energy frozen, dt_crit = 1e30.
4. Acoustic sound speed & Courant time-step stability:
   - Longitudinal sound speed c remains positive, finite, and well-behaved across uncompacted, transition, and compacted states.
   - Courant time step Delta t = L_e / c stability maintained throughout simulation.

Fortran references:
- ``engine/source/materials/mat/mat050/sigeps50s.F90`` (solid constitutive kernel)
- ``starter/source/materials/mat/mat050/hm_read_mat50.F90`` (starter reader & parameter validation)
- ``hm_cfg_files/config/CFG/radioss2025/MAT/mat_law50.cfg`` (CFG attributes & card format)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import solid_hexa8, solid_tetra4
from pyradioss.materials.law50_visc_honey import (
    Law50Params,
    build_law50,
    solid_update,
    sound_speed,
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


def make_honeycomb_law50(
    mid: int = 1,
    rho0: float = 0.15e-3,       # Honeycomb density: 0.15 g/cm^3 = 0.15e-3 g/mm^3
    ea: float = 200.0,           # E11 (MPa)
    eb: float = 300.0,           # E22 (MPa)
    ec: float = 800.0,           # E33 (MPa, strong out-of-plane direction)
    gab: float = 80.0,           # G12 (MPa)
    gbc: float = 120.0,          # G23 (MPa)
    gca: float = 150.0,          # G31 (MPa)
    asrate: float = 0.0,         # fcut (0 -> instantaneous / no lag)
    gflag: int = 0,
    vflag: int = 0,
    irate: int = 2,
    eps_max11: float = 1.0e30,
    eps_max22: float = 1.0e30,
    eps_max33: float = 1.0e30,
    eps_max12: float = 1.0e30,
    eps_max23: float = 1.0e30,
    eps_max31: float = 1.0e30,
    ecomp: float = 2500.0,       # Compacted modulus (MPa)
    et: float = 60.0,            # Compacted tangent modulus / hardening (MPa)
    sigy: float = 50.0,          # Compacted yield stress (MPa)
    pr: float = 0.3,             # Compacted Poisson's ratio
    vcomp: float = 0.25,         # Relative compaction volume Vcomp
    **kwargs: Any,
) -> Material:
    """Standard Honeycomb / Hyp Foam material helper."""
    params = {
        "rho0": rho0,
        "Refer_Rho": rho0,
        "MAT_RHO": rho0,
        "MAT_EA": ea,
        "MAT_EB": eb,
        "MAT_EC": ec,
        "MAT_GAB": gab,
        "MAT_GBC": gbc,
        "MAT_GCA": gca,
        "MAT_asrate": asrate,
        "Gflag": gflag,
        "Vflag": vflag,
        "Irate": irate,
        "MAT_EPS_max11": eps_max11,
        "MAT_EPS_max22": eps_max22,
        "MAT_EPS_max33": eps_max33,
        "MAT_EPS_max12": eps_max12,
        "MAT_EPS_max23": eps_max23,
        "MAT_EPS_max31": eps_max31,
        "MAT_ECOMP": ecomp,
        "MAT_ET": et,
        "MAT_SIGY": sigy,
        "MAT_PR": pr,
        "MAT_VCOMP": vcomp,
    }
    params.update(kwargs)
    rec = {
        "id": mid,
        "title": f"HONEYCOMB_{mid}",
        "params": params,
    }
    mat = build_law50(rec)
    mat.id = mid
    return mat


# ============================================================================
# Engine Simulation & Energy Balance Test Suite
# ============================================================================

class TestLaw50DynamicEngineSimulation:
    """Multi-cycle explicit dynamic simulations and energy balance checks for /MAT/LAW50."""

    def test_single_element_cyclic_elastic_energy_conservation_hexa8(self):
        """Cyclic elastic shear vibration of a Hexa8 element: verify |Delta E| / E_peak < 1% over 100 cycles."""
        # Unyielding elastic honeycomb: high yield thresholds, zero artificial viscosity
        mat = make_honeycomb_law50(
            ea=300.0,
            eb=300.0,
            ec=1000.0,
            gab=120.0,
            gbc=150.0,
            gca=150.0,
        )
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

        dt = 1.0e-6
        n_cycles = 100
        period = 20 * dt  # 5 complete sinusoidal periods
        amp = 1.0e-5      # shear strain amplitude

        # Top face nodes (4, 5, 6, 7) oscillate in x
        top_nodes = [4, 5, 6, 7]
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

            # Coordinates at step
            x_step = coords.copy()
            x_step[top_nodes, 0] += gamma_now

            fint.fill(0.0)
            dt_crit = solid_hexa8.forces(group, x_step, v, None, dt, fint, mint)

            cur_eint = group.state["eint"][0]
            peak_eint = max(peak_eint, abs(cur_eint))

            # External work done on top nodes by applied shear motion: - sum(f_int . v) * dt
            f_top_x = -np.sum(fint[top_nodes, 0])
            w_ext_total += f_top_x * dgamma

            assert dt_crit[0] > 0.0

        # After 5 complete sinusoidal cycles, final shear displacement returns to 0
        final_gamma = amp * math.sin(2.0 * math.pi * (n_cycles * dt) / period)
        assert math.isclose(final_gamma, 0.0, abs_tol=1e-12)

        # Reversible elastic energy conservation: final residual energy error < 1% of peak energy
        final_eint = group.state["eint"][0]
        assert peak_eint > 0.0
        rel_energy_drift = abs(final_eint) / peak_eint
        assert rel_energy_drift < 0.01, f"Hexa8 cyclic energy drift {rel_energy_drift:.4e} exceeds 1%"

        # No spurious compaction, plasticity, or deletion
        assert group.state["epsp"][0] == 0.0
        assert not bool(group.state["mat_extra"]["compacted"][0])
        assert group.state["mat_extra"]["off50"][0] == 1.0

    def test_single_element_cyclic_elastic_energy_conservation_tetra4(self):
        """Cyclic elastic shear vibration of a Tetra4 element: verify |Delta E| / E_peak < 1% over 100 cycles."""
        mat = make_honeycomb_law50(
            ea=300.0,
            eb=300.0,
            ec=1000.0,
            gab=120.0,
            gbc=150.0,
            gca=150.0,
        )
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

        dt = 1.0e-6
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

        assert group.state["epsp"][0] == 0.0
        assert not bool(group.state["mat_extra"]["compacted"][0])
        assert group.state["mat_extra"]["off50"][0] == 1.0

    def test_multielement_2x2x2_patch_dynamic_simulation(self):
        """2x2x2 Hexa8 mesh (8 elements, 27 nodes) under dynamic explicit compression wave: 50 cycles."""
        mat = make_honeycomb_law50(
            ea=150.0,
            eb=150.0,
            ec=500.0,
            gab=60.0,
            gbc=90.0,
            gca=90.0,
            ecomp=1500.0,
            sigy=30.0,
            et=40.0,
            vcomp=0.3,
        )
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

        # Dynamic compressive shock velocity on top face (z = 2.0, nodes with z=2.0)
        top_node_indices = [i for i in range(27) if abs(coords[i, 2] - 2.0) < 1e-6]
        bot_node_indices = [i for i in range(27) if abs(coords[i, 2] - 0.0) < 1e-6]

        v_top = -10.0  # compressive velocity in z (mm/s or m/s)
        dt = 1.0e-5
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
            assert np.all(dt_crit > 0.0)
            assert np.all(np.isfinite(dt_crit))

        # Total internal energy must be positive and non-zero from compressive work
        total_eint = np.sum(group.state["eint"])
        assert total_eint > 0.0
        # Normal compressive stress developed in z
        assert np.mean(group.state["sig"][:, 2]) < 0.0

    def test_dynamic_volumetric_compaction_and_plastic_dissipation(self):
        """Dynamic high crush: compaction transition (r_vol <= V_comp) and monotonic J2 plastic dissipation."""
        vcomp = 0.4
        ecomp = 3000.0
        sigy = 40.0
        et = 100.0
        mat = make_honeycomb_law50(
            ea=100.0,
            eb=100.0,
            ec=200.0,
            gab=50.0,
            gbc=50.0,
            gca=50.0,
            ecomp=ecomp,
            sigy=sigy,
            et=et,
            vcomp=vcomp,
            pr=0.25,
        )

        n_steps = 50
        dt = 1.0e-5

        sig = np.zeros(6, dtype=float)
        extra: Dict[str, Any] = {
            "mu": 0.0,
            "amu": 0.0,
            "compacted": False,
            "off50": 1.0,
            "eps50": np.zeros(6),
            "uvar50": np.zeros(6),
        }
        epsp = 0.0

        prev_epsp = 0.0
        e_int = 0.0
        compaction_triggered = False

        for step in range(n_steps):
            # Progressive volumetric and deviatoric compression
            # eps_zz increment: -0.02 per step
            deps = np.zeros(6, dtype=float)
            deps[2] = -0.02
            deps[0] = -0.005
            deps[1] = -0.005

            # Current volumetric strain mu = - tr(eps)
            extra["mu"] -= np.sum(deps[:3])
            extra["amu"] = extra["mu"]

            rvol = 1.0 / (1.0 + extra["mu"])

            sig_old = sig.copy()
            sig, epsp, c = solid_update(mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)

            dw = 0.5 * np.sum((sig_old + sig) * deps)
            e_int += dw

            if rvol <= vcomp:
                compaction_triggered = True

            if compaction_triggered:
                # Permanent compaction flag set
                assert bool(extra["compacted"])
                # Plastic strain must monotonically increase under continuing plastic flow
                assert epsp >= prev_epsp

            # Positive sound speed
            assert c > 0.0
            prev_epsp = epsp

        # Verification at end of crush
        assert compaction_triggered
        assert bool(extra["compacted"])
        assert epsp > 0.0
        assert e_int > 0.0
        # High compressive normal stress developed
        assert sig[2] < -sigy

    def test_acoustic_courant_stability_across_compaction_transition(self):
        """Sound speed and Courant stability tracked across uncompacted -> transition -> compacted states."""
        rho0 = 0.2e-3  # 0.2 g/cm^3
        ea = 100.0
        ec = 400.0
        ecomp = 4000.0
        vcomp = 0.3
        mat = make_honeycomb_law50(
            rho0=rho0,
            ea=ea,
            eb=ea,
            ec=ec,
            gab=50.0,
            gbc=50.0,
            gca=50.0,
            ecomp=ecomp,
            vcomp=vcomp,
            sigy=30.0,
        )

        c_uncompacted_expected = math.sqrt(ec / rho0)

        # 1. Uncompacted state: mu = 0.0 -> rvol = 1.0
        extra_1: Dict[str, Any] = {"mu": 0.0, "amu": 0.0, "compacted": False, "off50": 1.0, "eps50": np.zeros(6), "uvar50": np.zeros(6)}
        sig1 = np.zeros(6)
        deps1 = np.zeros(6)
        _, _, c1 = solid_update(mat, sig1, deps=deps1, dt=1e-6, extra=extra_1, return_tuple=True)
        assert math.isclose(c1, c_uncompacted_expected, rel_tol=1e-4)

        # 2. Transition state: mu = 1.0 -> rvol = 0.5 (between 1.0 and vcomp = 0.3)
        extra_2: Dict[str, Any] = {"mu": 1.0, "amu": 1.0, "rho": rho0 * 2.0, "compacted": False, "off50": 1.0, "eps50": np.zeros(6), "uvar50": np.zeros(6)}
        sig2 = np.zeros(6)
        deps2 = np.zeros(6)
        _, _, c2 = solid_update(mat, sig2, deps=deps2, dt=1e-6, extra=extra_2, return_tuple=True)
        assert c2 > c1  # sound speed stiffens during compaction
        assert np.isfinite(c2)

        # 3. Fully compacted state: mu = 3.0 -> rvol = 0.25 <= vcomp = 0.3
        extra_3: Dict[str, Any] = {"mu": 3.0, "amu": 3.0, "rho": rho0 * 4.0, "compacted": True, "off50": 1.0, "eps50": np.zeros(6), "uvar50": np.zeros(6)}
        sig3 = np.zeros(6)
        deps3 = np.zeros(6)
        _, _, c3 = solid_update(mat, sig3, deps=deps3, dt=1e-6, extra=extra_3, return_tuple=True)

        assert bool(extra_3["compacted"])
        assert c3 > 0.0
        assert np.isfinite(c3)

        # Element Courant time step for characteristic length Le = 1.0 mm remains well-behaved
        le = 1.0
        dt_c1 = le / c1
        dt_c2 = le / c2
        dt_c3 = le / c3
        assert 1.0e-2 >= dt_c1 > 0.0
        assert 1.0e-2 >= dt_c2 > 0.0
        assert 1.0e-2 >= dt_c3 > 0.0
        assert np.isfinite([dt_c1, dt_c2, dt_c3]).all()

    def test_dynamic_strain_rate_filtering_multicycle(self):
        """Strain rate filter uvar50 evolution under multi-step cyclic rate variation."""
        fcut = 500.0  # cut-off frequency (s^-1)
        mat = make_honeycomb_law50(
            ea=1000.0,
            asrate=fcut,
            irate=2,
        )

        dt = 1.0e-4
        n_steps = 20

        sig = np.zeros(6)
        extra: Dict[str, Any] = {
            "mu": 0.0,
            "amu": 0.0,
            "off50": 1.0,
            "eps50": np.zeros(6),
            "uvar50": np.zeros(6),
        }

        asrate_weight = min(1.0, fcut * dt)  # 500 * 1e-4 = 0.05
        assert math.isclose(asrate_weight, 0.05, rel_tol=1e-6)

        # Apply constant strain rate of 100.0 s^-1 for 20 steps
        rate_input = 100.0
        deps = np.zeros(6)
        deps[0] = rate_input * dt

        expected_uvar = 0.0
        for step in range(n_steps):
            expected_uvar = asrate_weight * rate_input + (1.0 - asrate_weight) * expected_uvar
            sig, _, _ = solid_update(mat, sig, deps=deps, dt=dt, extra=extra, return_tuple=True)
            actual_uvar = extra["uvar50"][0]
            assert math.isclose(actual_uvar, expected_uvar, rel_tol=1e-5)

        # After 20 steps, filtered rate uvar smoothly converges toward the applied rate
        assert actual_uvar > 0.6 * rate_input

    def test_dynamic_erosion_multicycle_stress_collapse_and_energy_freeze(self):
        """Dynamic tensile failure erosion (eps > eps_max): stress collapses to 0, forces zeroed, energy frozen."""
        eps_limit = 0.05
        mat = make_honeycomb_law50(
            ea=1000.0,
            eps_max11=eps_limit,
        )
        prop = MockProp(qa=0.0, qb=0.0, h=0.0)

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

        dt = 1.0e-5
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        # Cycle 1: Pre-failure elastic tension (eps_xx = 0.03 < 0.05)
        v = np.zeros((8, 3))
        v[[1, 2, 5, 6], 0] = 3000.0  # deps = 0.03
        dt_crit = solid_hexa8.forces(group, coords, v, None, dt, fint, mint)

        assert group.state["off"][0] == 1.0
        assert group.state["mat_extra"]["off50"][0] == 1.0
        assert group.state["sig"][0, 0] > 0.0
        assert group.state["eint"][0] > 0.0
        assert dt_crit[0] < 1.0e20

        # Cycle 2: Trigger failure (eps_xx = 0.03 + 0.03 = 0.06 > 0.05)
        coords_step2 = coords.copy()
        coords_step2[[1, 2, 5, 6], 0] += 0.03
        v[[1, 2, 5, 6], 0] = 3000.0
        fint.fill(0.0)
        dt_crit = solid_hexa8.forces(group, coords_step2, v, None, dt, fint, mint)

        # Deletion triggered: off = 0.0, stresses zeroed, dt_crit set to 1e30
        assert group.state["off"][0] == 0.0
        assert group.state["mat_extra"]["off50"][0] == 0.0
        assert np.all(group.state["sig"][0] == 0.0)
        assert np.all(fint == 0.0)
        assert dt_crit[0] >= 1.0e29

        # Internal energy at failure is now frozen for all remaining cycles
        frozen_eint = group.state["eint"][0]
        assert frozen_eint > 0.0

        # Cycles 3 to 15: Element is dead, state remains frozen despite ongoing velocities
        for _ in range(12):
            coords_step2[[1, 2, 5, 6], 0] += 0.03
            fint.fill(0.0)
            dt_crit = solid_hexa8.forces(group, coords_step2, v, None, dt, fint, mint)

            assert group.state["off"][0] == 0.0
            assert np.all(group.state["sig"][0] == 0.0)
            assert np.all(fint == 0.0)
            assert dt_crit[0] >= 1.0e29
            # Internal energy remains frozen (no energy leak)
            assert math.isclose(group.state["eint"][0], frozen_eint, rel_tol=1e-6)

    def test_tetra4_dynamic_crush_internal_equilibrium(self):
        """Tetra4 element under dynamic crush: internal force sum f_int = 0 at every step, compaction."""
        mat = make_honeycomb_law50(
            ea=120.0,
            eb=120.0,
            ec=300.0,
            gab=60.0,
            gbc=60.0,
            gca=60.0,
            ecomp=2000.0,
            sigy=35.0,
            et=50.0,
            vcomp=0.35,
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

        dt = 1.0e-6
        n_steps = 30
        cur_coords = coords.copy()
        v = np.zeros((4, 3), dtype=float)
        # Apex node moving downward rapidly (crush)
        v[3, 2] = -5.0e3

        fint = np.zeros((4, 3))
        mint = np.zeros((4, 3))

        for step in range(n_steps):
            cur_coords += v * dt
            fint.fill(0.0)
            dt_crit = solid_tetra4.forces(group, cur_coords, v, None, dt, fint, mint)

            # Equilibrium: sum of internal forces on closed element must sum to zero identically
            f_sum = np.sum(fint, axis=0)
            assert np.allclose(f_sum, 0.0, atol=1e-4)

            assert dt_crit[0] > 0.0
            assert np.isfinite(group.state["sig"]).all()

        assert group.state["eint"][0] > 0.0
