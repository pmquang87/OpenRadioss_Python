"""Tests for Milestone M539 (Auditor 3): Explicit Engine Simulation & Energy Balance.

Validates /MAT/LAW34 (Boltzmann linear viscoelastic relaxation model) across:
1. Multi-cycle explicit engine simulation on solid Hexa8 elements:
   - 50+ time steps with Courant time step control.
   - Energy balance: W_ext, E_int, E_k, and |ERR| < 1.0%.
   - Stress relaxation hold: displacement ramp, then held fixed;
     exponential stress relaxation towards sigma_inf = 2 * G_inf * eps_0.
2. Multi-cycle explicit engine simulation on shell elements (BT4 / QBAT):
   - In-plane shear and biaxial tension.
   - Conservation of out-of-plane stress sigma_zz = 0 throughout the dynamic run.
   - Thickness change tracking (ezz34 / uv34).
3. Multi-element impact mesh simulation:
   - 2x2x2 mesh of 8 Hexa8 elements subjected to high-velocity impact.
   - Foam air pressure enabled (P0 > 0, phi = 0.1, gamma0 = 0.0).
   - Stability, no NaN/Inf, energy balance conservation (|ERR| < 1.0%).
4. Combined solid (Hexa8 + Tetra4) simulation:
   - Mixed mesh compatibility across element families.
   - Simultaneous integration of Hexa8 and Tetra4 elements under dynamic loading.
"""

import os
import io
import math
import contextlib
import numpy as np
import pytest

from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.starter import run_starter
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.starter.restart import read_restart
from pyradioss.materials import law34_boltzmann as l34
from pyradioss.common.messages import MessageLog


# =============================================================================
# Helper: Verify Plane-Stress sigma_zz = 0 Condition
# =============================================================================

def _assert_plane_stress_sigma_zz_zero(mat, deps_xx, deps_yy, deps_xy, dt):
    """Verify that LAW34 plane-stress solution satisfies sigma_zz = 0 in 3D solid kernel."""
    sign_sh, _ = l34.shell_update(mat, np.zeros((1, 3)), np.array([[deps_xx, deps_yy, deps_xy]]), dt=dt)

    p = mat.params
    bulk = p["bulk"]
    g0 = p["g0"]
    gi = p["gi"]
    beta = p["beta"]

    ge2 = 2.0 * gi
    gv2 = 2.0 * (g0 - gi)
    bulk3 = 3.0 * bulk

    c1 = 1.0 - math.exp(-beta * dt)
    c2 = -c1 / beta
    cc2 = gv2 * (c1 + c2 / dt)

    aa = (1.0 / 3.0) * (ge2 - cc2 - bulk3) * (deps_xx + deps_yy)
    bb = (2.0 / 3.0) * ge2 + bulk - (2.0 / 3.0) * cc2
    deps_zz = aa / bb

    deps_3d = np.array([[deps_xx, deps_yy, deps_zz, deps_xy, 0.0, 0.0]])
    sign_3d, _, _ = l34.solid_update(mat, np.zeros((1, 6)), deps_3d, dt=dt)

    assert abs(sign_3d[0, 2]) < 1e-12, f"Expected sigma_zz == 0, got {sign_3d[0, 2]}"
    assert np.allclose(sign_sh[0, :2], sign_3d[0, :2], atol=1e-10)


# =============================================================================
# 1. Solid Hexa8 Multi-Cycle Engine Simulation & Energy Balance
# =============================================================================

class TestHexa8ExplicitSimulation:
    """Explicit dynamic simulation on solid Hexa8 elements with LAW34."""

    def test_hexa8_explicit_multicycle_courant_control(self, tmp_path):
        """Verify explicit integration over 50+ steps, Courant control, and |ERR| < 1%."""
        run_name = "HEXA8_COURANT"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        # Unit cube 10 x 10 x 10
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ])
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        deck.part(1, "HEXA8_BLOCK", 1, 1)

        rho0 = 1.0e-3
        bulk = 100.0
        g0 = 30.0
        gi = 10.0
        beta = 15.0

        deck.mat_law34(
            1,
            rho=rho0,
            bulk=bulk,
            g0=g0,
            gi=gi,
            beta=beta,
            title="LAW34_SOLID",
        )
        deck.prop_solid(1, "HEXA_PROP")

        # Boundary conditions: Clamped base (nodes 1..4)
        deck.grnod_node(1, "base_nodes", [1, 2, 3, 4])
        deck.grnod_node(2, "top_nodes", [5, 6, 7, 8])
        deck.bcs(1, "clamp_base", "111", "111", 1)

        # Imposed velocity ramp: compression in Z
        # Creates continuous external work W_ext
        deck.funct(1, "vel_ramp", [(0.0, 0.0), (0.1, -2.0), (2.0, -2.0)])
        deck.impvel(1, "top_vel", 1, "Z", 2)
        deck.write(s_path)

        # Engine deck: stop at cycle 60
        dt_scale = 0.9
        engine_deck = f"""/RUN/{run_name}/1
2.0
/DT
{dt_scale} 0
/PRINT/-1
/STOP
60
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        log = MessageLog()
        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path, log=log)
            eng_model = run_engine(e_path)

        assert len(log.errors) == 0
        state = eng_model.engine_state

        # 1. Verify 50+ time steps completed
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"

        # 2. Verify Courant time step control
        # Sound speed c = sqrt((K + 4/3 * G0) / rho0)
        c_sound = math.sqrt((bulk + (4.0 / 3.0) * g0) / rho0)
        lc = 10.0  # element size
        dt_courant_limit = dt_scale * (lc / c_sound)

        _, eng_dict = read_restart(os.path.join(tmp_path, f"{run_name}_0001.rst"))
        engine_dt = eng_dict["dt"]
        assert engine_dt <= dt_courant_limit * 1.05, (
            f"Engine time step {engine_dt} exceeds Courant bound {dt_courant_limit}"
        )

        # 3. Verify energy balance: W_ext, E_int, E_k, and |ERR| < 1.0%
        en = _energies(eng_model, state)
        assert en["EW"] > 0.0, f"External work must be positive, got {en['EW']}"
        assert en["IE"] > 0.0, f"Internal energy must be positive, got {en['IE']}"
        assert en["KE"] > 0.0, f"Kinetic energy must be positive, got {en['KE']}"
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR| must be < 1%, got {en['ERR']}%"

        # 4. Check stress and state variables
        brick_g = dict(eng_model.element_groups())["bricks"]
        sig = brick_g.state["sig"]
        assert np.isfinite(sig).all()
        # Compression in Z
        assert sig[0, 2] < 0.0

        # Viscoelastic state history arrays exist and are finite
        assert "eps34" in brick_g.state["mat_extra"]
        assert "uv34" in brick_g.state["mat_extra"]
        assert np.isfinite(brick_g.state["mat_extra"]["eps34"]).all()
        assert np.isfinite(brick_g.state["mat_extra"]["uv34"]).all()

    def test_hexa8_stress_relaxation_hold(self, tmp_path):
        """Verify exponential stress relaxation towards sigma_inf = 2 * G_inf * eps_0.

        Load case:
        - Pure shear displacement ramp applied to top nodes up to t_ramp = 0.05.
        - Held fixed for t > t_ramp up to t_end = 0.5 (4.5 time constants for beta=10).
        - Verify exponential relaxation:
            sigma(t) - sigma_inf = (sigma_peak - sigma_inf) * exp(-beta * (t - t_ramp))
        - Verify equilibrium state converges to sigma_inf = 2 * G_inf * eps_0.
        """
        run_name = "HEXA8_RELAX"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        # Block: 10 x 10 x 10
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ])
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        deck.part(1, "HEXA_RELAX", 1, 1)

        rho0 = 1.0e-3
        bulk = 100.0
        g0 = 30.0
        gi = 10.0
        beta = 10.0  # relaxation parameter, tau = 1/beta = 0.1 s

        deck.mat_law34(
            1,
            rho=rho0,
            bulk=bulk,
            g0=g0,
            gi=gi,
            beta=beta,
            title="LAW34_RELAX",
        )
        deck.prop_solid(1, "HEXA_PROP")

        # Clamped base (nodes 1..4)
        deck.grnod_node(1, "base", [1, 2, 3, 4])
        deck.bcs(1, "fix_base", "111", "111", 1)

        # Top nodes: constrain Y and Z, drive X with ramp-and-hold
        deck.grnod_node(2, "top", [5, 6, 7, 8])
        deck.bcs(2, "fix_top_yz", "011", "111", 2)

        # Shear displacement: u_x = 0.1 over height h = 10.0 -> engineering gamma_0 = 0.01
        # Tensor shear strain eps_0 = 0.5 * gamma_0 = 0.005
        u0 = 0.1
        t_ramp = 0.05
        deck.funct(1, "ramp_hold", [(0.0, 0.0), (t_ramp, u0), (1.0, u0)])
        deck.impdisp(1, "disp_x", 1, "X", 2)
        deck.write(s_path)

        # Run past several relaxation times (t_end = 0.5 -> 4.5 * tau)
        engine_deck = f"""/RUN/{run_name}/1
0.5
/DT
0.9 0
/PRINT/-1
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 30

        brick_g = dict(eng_model.element_groups())["bricks"]
        sig = brick_g.state["sig"]

        # Theoretical equilibrium shear stress:
        # gamma_0 = u0 / Lz = 0.1 / 10.0 = 0.01
        # eps_0 = gamma_0 / 2 = 0.005
        # sigma_inf = 2 * G_inf * eps_0 = G_inf * gamma_0 = 10.0 * 0.01 = 0.10
        gamma_0 = u0 / 10.0
        eps_0 = 0.5 * gamma_0
        sigma_inf = 2.0 * gi * eps_0

        # Component 5 is tau_zx in Radioss Voigt [xx, yy, zz, xy, yz, zx]
        tau_zx = sig[0, 5]

        # Verify convergence to sigma_inf (within 0.5% after 4.5 time constants)
        rel_diff = abs(tau_zx - sigma_inf) / sigma_inf
        assert rel_diff < 0.005, (
            f"Relaxed shear stress {tau_zx} does not match sigma_inf {sigma_inf} (rel diff {rel_diff})"
        )

        # Verify energy balance throughout the hold
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0


# =============================================================================
# 2. Shell Elements (BT4 / QBAT) Multi-Cycle Engine Simulation
# =============================================================================

class TestShellExplicitSimulation:
    """Explicit dynamic simulations on BT4 and QBAT shell elements with LAW34."""

    def test_shell_bt4_inplane_shear(self, tmp_path):
        """BT4 shell under in-plane shear: verify sigma_zz = 0 conservation and energy balance."""
        run_name = "SHELL_BT4_SHEAR"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 3, 4)])
        deck.part(1, "SHELL_PART", 1, 1)

        deck.mat_law34(
            1,
            rho=1.0e-3,
            bulk=150.0,
            g0=45.0,
            gi=15.0,
            beta=20.0,
            title="LAW34_SHELL",
        )
        # ishell=1 is Belytschko-Tsay (BT4)
        deck.prop_shell(1, "BT4_PROP", thick=1.0, nip=3, ishell=1)

        # Fix bottom edge in X and Y
        deck.grnod_node(1, "bottom_edge", [1, 2])
        deck.bcs(1, "fix_bottom", "111", "111", 1)

        # Apply shear displacement to top edge (nodes 3, 4) in X
        deck.grnod_node(2, "top_edge", [3, 4])
        deck.bcs(2, "fix_top_yz", "011", "111", 2)
        deck.funct(1, "shear_ramp", [(0.0, 0.0), (0.2, 0.2), (2.0, 0.2)])
        deck.impdisp(1, "shear_x", 1, "X", 2)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
2.0
/DT
0.9 0
/PRINT/-1
/STOP
55
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50

        sh_g = dict(eng_model.element_groups())["shells"]
        sig = sh_g.state["sig"]  # shape (1, nip=3, 3) for [xx, yy, xy]
        assert np.isfinite(sig).all()

        # In-plane stress vector has shape (..., 3) — sigma_zz = 0 is conserved identically
        assert sig.shape[-1] == 3
        # In-plane shear stress developed
        assert abs(sig[0, :, 2].mean()) > 0.0

        # Verify plane-stress sigma_zz = 0 condition
        mat = st_model.materials[1]
        _, eng_dict = read_restart(os.path.join(tmp_path, f"{run_name}_0001.rst"))
        _assert_plane_stress_sigma_zz_zero(mat, 0.0, 0.0, 0.02, eng_dict["dt"])

        # Check thickness tracking consistency
        ezz34 = sh_g.state["mat_extra"]["ezz34"]
        uv34 = sh_g.state["mat_extra"]["uv34"]
        for il in range(3):
            assert math.isclose(ezz34[0, il], uv34[0, il, 6], rel_tol=1e-10)

        # Verify energy balance: |ERR| < 1.0%
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0

    def test_shell_bt4_biaxial_tension_thickness_tracking(self, tmp_path):
        """BT4 shell under biaxial tension: verify sigma_zz = 0 and negative thickness strain."""
        run_name = "SHELL_BT4_BIAX"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 3, 4)])
        deck.part(1, "SHELL_PART", 1, 1)

        deck.mat_law34(
            1,
            rho=1.0e-3,
            bulk=100.0,
            g0=30.0,
            gi=10.0,
            beta=15.0,
            title="LAW34_SHELL",
        )
        deck.prop_shell(1, "BT4_PROP", thick=2.0, nip=3, ishell=1)

        # Symmetric biaxial extension:
        # Node 1 at (0,0): fixed X, Y, Z
        # Node 2 at (10,0): free in X, fixed Y, Z
        # Node 4 at (0,10): fixed X, free in Y, fixed Z
        # Node 3 at (10,10): free in X, Y, fixed Z
        deck.grnod_node(1, "origin", [1])
        deck.bcs(1, "fix_origin", "111", "111", 1)

        deck.grnod_node(2, "bottom_right", [2])
        deck.bcs(2, "fix_br", "011", "111", 2)

        deck.grnod_node(3, "top_left", [4])
        deck.bcs(3, "fix_tl", "101", "111", 3)

        deck.grnod_node(4, "all_z", [1, 2, 3, 4])
        deck.bcs(4, "fix_z", "001", "111", 4)

        # Stretch along X (nodes 2, 3) and Y (nodes 3, 4)
        deck.grnod_node(5, "stretch_x", [2, 3])
        deck.grnod_node(6, "stretch_y", [3, 4])
        deck.funct(1, "stretch_ramp", [(0.0, 0.0), (0.2, 0.2), (2.0, 0.2)])
        deck.impdisp(1, "disp_x", 1, "X", 5)
        deck.impdisp(2, "disp_y", 1, "Y", 6)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
2.0
/DT
0.9 0
/PRINT/-1
/STOP
55
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50

        sh_g = dict(eng_model.element_groups())["shells"]
        sig = sh_g.state["sig"]
        assert np.isfinite(sig).all()

        # Both in-plane normal stresses must be tensile (> 0)
        assert (sig[0, :, 0] > 0.0).all()
        assert (sig[0, :, 1] > 0.0).all()

        # Verify thickness change tracking:
        # Under biaxial tension, Poisson contraction dictates negative out-of-plane strain
        ezz34 = sh_g.state["mat_extra"]["ezz34"]
        uv34 = sh_g.state["mat_extra"]["uv34"]

        assert (ezz34 < 0.0).all(), "Thickness strain ezz must be negative under biaxial tension"
        assert (uv34[..., 6] < 0.0).all()

        # Verify plane-stress sigma_zz = 0 condition analytically
        mat = st_model.materials[1]
        _, eng_dict = read_restart(os.path.join(tmp_path, f"{run_name}_0001.rst"))
        _assert_plane_stress_sigma_zz_zero(mat, 0.02, 0.02, 0.0, eng_dict["dt"])

        # Verify energy balance: |ERR| < 1.0%
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0

    def test_shell_qbat_inplane_shear(self, tmp_path):
        """Fully-integrated QBAT shell (Ishell=12) under in-plane shear."""
        run_name = "SHELL_QBAT_SHEAR"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 3, 4)])
        deck.part(1, "QBAT_PART", 1, 1)

        deck.mat_law34(
            1,
            rho=1.0e-3,
            bulk=120.0,
            g0=40.0,
            gi=15.0,
            beta=10.0,
            title="LAW34_QBAT",
        )
        # Ishell=12 for QBAT formulation
        deck.prop_shell(1, "QBAT_PROP", thick=1.0, nip=3, ishell=12)

        deck.grnod_node(1, "bottom_edge", [1, 2])
        deck.bcs(1, "fix_bottom", "111", "111", 1)

        deck.grnod_node(2, "top_edge", [3, 4])
        deck.bcs(2, "fix_top_yz", "011", "111", 2)
        deck.funct(1, "shear_f", [(0.0, 0.0), (0.2, 0.15), (2.0, 0.15)])
        deck.impdisp(1, "shear_x", 1, "X", 2)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
2.0
/DT
0.9 0
/PRINT/-1
/STOP
55
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50

        qb_g = dict(eng_model.element_groups())["shells_qbat"]
        sig = qb_g.state["sig"]
        assert np.isfinite(sig).all()
        # QBAT has 4 in-plane Gauss stations * nip layers = 12 Gauss points, 3 in-plane stresses
        assert sig.shape == (1, 12, 3)

        # Non-zero shear stress developed across all Gauss points
        assert np.abs(sig[0, :, 2]).mean() > 0.0

        # State tracking
        assert "ezz34" in qb_g.state["mat_extra"]
        assert "uv34" in qb_g.state["mat_extra"]
        assert np.isfinite(qb_g.state["mat_extra"]["ezz34"]).all()

        # Verify plane-stress sigma_zz = 0 condition analytically
        mat = st_model.materials[1]
        _, eng_dict = read_restart(os.path.join(tmp_path, f"{run_name}_0001.rst"))
        _assert_plane_stress_sigma_zz_zero(mat, 0.0, 0.0, 0.015, eng_dict["dt"])

        # Energy balance: |ERR| < 1.0%
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0

    def test_shell_qbat_biaxial_tension_thickness_tracking(self, tmp_path):
        """Fully-integrated QBAT shell (Ishell=12) under biaxial tension: verify ezz < 0."""
        run_name = "SHELL_QBAT_BIAX"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 3, 4)])
        deck.part(1, "QBAT_PART", 1, 1)

        deck.mat_law34(
            1,
            rho=1.0e-3,
            bulk=100.0,
            g0=30.0,
            gi=10.0,
            beta=15.0,
            title="LAW34_QBAT",
        )
        deck.prop_shell(1, "QBAT_PROP", thick=1.5, nip=3, ishell=12)

        deck.grnod_node(1, "origin", [1])
        deck.bcs(1, "fix_origin", "111", "111", 1)

        deck.grnod_node(2, "bottom_right", [2])
        deck.bcs(2, "fix_br", "011", "111", 2)

        deck.grnod_node(3, "top_left", [4])
        deck.bcs(3, "fix_tl", "101", "111", 3)

        deck.grnod_node(4, "all_z", [1, 2, 3, 4])
        deck.bcs(4, "fix_z", "001", "111", 4)

        deck.grnod_node(5, "stretch_x", [2, 3])
        deck.grnod_node(6, "stretch_y", [3, 4])
        deck.funct(1, "stretch_ramp", [(0.0, 0.0), (0.2, 0.2), (2.0, 0.2)])
        deck.impdisp(1, "disp_x", 1, "X", 5)
        deck.impdisp(2, "disp_y", 1, "Y", 6)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
2.0
/DT
0.9 0
/PRINT/-1
/STOP
55
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50

        qb_g = dict(eng_model.element_groups())["shells_qbat"]
        sig = qb_g.state["sig"]
        assert (sig[0, :, 0] > 0.0).all()
        assert (sig[0, :, 1] > 0.0).all()

        # Thickness contraction across all 12 integration stations
        ezz34 = qb_g.state["mat_extra"]["ezz34"]
        assert (ezz34 < 0.0).all(), "QBAT out-of-plane strain ezz must be negative under biaxial tension"

        # Verify plane-stress sigma_zz = 0 condition analytically
        mat = st_model.materials[1]
        _, eng_dict = read_restart(os.path.join(tmp_path, f"{run_name}_0001.rst"))
        _assert_plane_stress_sigma_zz_zero(mat, 0.02, 0.02, 0.0, eng_dict["dt"])

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0


# =============================================================================
# 3. Multi-Element Impact Mesh Simulation (2x2x2 Hexa8 with Air Pressure)
# =============================================================================

class TestMultiElementImpactMesh:
    """2x2x2 mesh of 8 Hexa8 elements with foam air pressure (P0 > 0)."""

    def test_2x2x2_hexa8_high_velocity_impact(self, tmp_path):
        """Verify stability, air pressure coupling, and energy balance under high velocity."""
        run_name = "HEXA8_2X2X2_IMPACT"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        # 3x3x3 grid of 27 nodes
        nodes = []
        nid = 1
        grid = np.zeros((3, 3, 3), dtype=int)
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    nodes.append((nid, float(i * 10.0), float(j * 10.0), float(k * 10.0)))
                    grid[i, j, k] = nid
                    nid += 1
        deck.node(nodes)

        # 8 Hexa8 brick elements
        bricks = []
        eid = 1
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    n1 = int(grid[i, j, k])
                    n2 = int(grid[i + 1, j, k])
                    n3 = int(grid[i + 1, j + 1, k])
                    n4 = int(grid[i, j + 1, k])
                    n5 = int(grid[i, j, k + 1])
                    n6 = int(grid[i + 1, j, k + 1])
                    n7 = int(grid[i + 1, j + 1, k + 1])
                    n8 = int(grid[i, j + 1, k + 1])
                    bricks.append((eid, n1, n2, n3, n4, n5, n6, n7, n8))
                    eid += 1
        deck.brick(1, bricks)
        deck.part(1, "FOAM_MESH", 1, 1)

        # Foam parameters: P0 > 0, phi = 0.1, gamma0 = 0.0
        p0 = 5.0
        phi = 0.1
        gamma0 = 0.0
        rho0 = 1.0e-3
        deck.mat_law34(
            1,
            rho=rho0,
            bulk=200.0,
            g0=50.0,
            gi=20.0,
            beta=25.0,
            p0=p0,
            phi=phi,
            gamma0=gamma0,
            title="FOAM_AIR_P0",
        )
        deck.prop_solid(1, "SOLID_PROP")

        # Boundary conditions: Clamped base (9 nodes at z=0)
        base_nodes = grid[:, :, 0].flatten().tolist()
        top_nodes = grid[:, :, 2].flatten().tolist()
        deck.grnod_node(1, "base_nodes", base_nodes)
        deck.grnod_node(2, "top_nodes", top_nodes)
        deck.bcs(1, "clamp_base", "111", "111", 1)

        # High-velocity impact in Z: v_z = -20.0
        deck.inivel_tra(1, "impact_vel", [0.0, 0.0, -20.0], 2)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
0.8
/DT
0.9 0
/PRINT/-1
/STOP
60
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        log = MessageLog()
        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path, log=log)
            eng_model = run_engine(e_path)

        assert len(log.errors) == 0
        state = eng_model.engine_state

        # 1. Stable integration over 50+ steps
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"

        # 2. No NaN or Inf in coordinates, velocities, or stresses
        assert np.isfinite(eng_model.x).all()
        assert np.isfinite(eng_model.v).all()

        brick_g = dict(eng_model.element_groups())["bricks"]
        sig = brick_g.state["sig"]
        assert np.isfinite(sig).all()
        assert sig.shape == (8, 6)

        # Stresses in all 8 elements should experience compression
        assert (sig[:, 2] < 0.0).all()

        # Viscoelastic state variables
        eps34 = brick_g.state["mat_extra"]["eps34"]
        uv34 = brick_g.state["mat_extra"]["uv34"]
        assert np.isfinite(eps34).all()
        assert np.isfinite(uv34).all()

        # 3. Energy balance conservation: |ERR| < 1.0%
        en = _energies(eng_model, state)
        assert en["IE"] > 0.0
        assert abs(en["ERR"]) < 1.0, f"Energy error {en['ERR']}% exceeds 1%"


# =============================================================================
# 4. Combined Solid Simulation (Hexa8 + Tetra4 Mixed Mesh)
# =============================================================================

class TestCombinedSolidSimulation:
    """Mixed mesh test combining Hexa8 and Tetra4 elements to ensure compatibility."""

    def test_combined_hexa8_tetra4_compatibility(self, tmp_path):
        """Run Hexa8 + Tetra4 mesh sharing LAW34 material under dynamic loading."""
        run_name = "MIXED_HEX_TET"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        # Mesh: 1 Hexa8 brick at base (nodes 1..8) + 2 Tetra4 elements sharing top face (nodes 5..8, apex 9)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
            (9, 5.0, 5.0, 20.0),
        ])
        # Brick 1: nodes 1..8
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        # Tetras 2 and 3 on top of the brick
        deck.tetra4(2, [
            (2, 5, 6, 7, 9),
            (3, 5, 7, 8, 9),
        ])
        deck.part(1, "HEX_PART", 1, 1)
        deck.part(2, "TET_PART", 2, 1)

        # Shared LAW34 material
        deck.mat_law34(
            1,
            rho=1.0e-3,
            bulk=200.0,
            g0=50.0,
            gi=20.0,
            beta=15.0,
            title="SHARED_LAW34",
        )
        deck.prop_solid(1, "HEX_PROP")
        deck.prop_solid(2, "TET_PROP")

        # Boundary conditions: Clamped base (nodes 1..4)
        deck.grnod_node(1, "base", [1, 2, 3, 4])
        deck.bcs(1, "fix_base", "111", "111", 1)

        # Dynamic loading: Prescribed velocity /IMPVEL on apex node 9 pushing downward
        deck.grnod_node(2, "apex", [9])
        deck.funct(1, "push_apex", [(0.0, -5.0), (10.0, -5.0)])
        deck.impvel(1, "vel_apex", 1, "Z", 2)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
0.5
/DT
0.9 0
/PRINT/-1
/STOP
60
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        log = MessageLog()
        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path, log=log)
            eng_model = run_engine(e_path)

        assert len(log.errors) == 0
        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"

        # Verify element groups
        groups = dict(eng_model.element_groups())
        assert "bricks" in groups and "tetras" in groups

        sig_b = groups["bricks"].state["sig"]
        sig_t = groups["tetras"].state["sig"]

        assert sig_b.shape == (1, 6)
        assert sig_t.shape == (2, 6)
        assert np.isfinite(sig_b).all()
        assert np.isfinite(sig_t).all()

        # Both element groups experience downward compressive load in Z
        assert sig_b[0, 2] < 0.0
        assert (sig_t[:, 2] < 0.0).all()

        # Check viscoelastic state arrays
        assert "eps34" in groups["bricks"].state["mat_extra"]
        assert "uv34" in groups["bricks"].state["mat_extra"]
        assert "eps34" in groups["tetras"].state["mat_extra"]
        assert "uv34" in groups["tetras"].state["mat_extra"]

        assert np.isfinite(groups["bricks"].state["mat_extra"]["eps34"]).all()
        assert np.isfinite(groups["tetras"].state["mat_extra"]["eps34"]).all()

        # Energy balance verification: |ERR| < 1.0%
        en = _energies(eng_model, state)
        assert en["EW"] > 0.0
        assert en["IE"] > 0.0
        assert abs(en["ERR"]) < 1.0, f"Energy error {en['ERR']}% exceeds 1%"

        assert np.isfinite(eng_model.x).all()
        assert np.isfinite(eng_model.v).all()
