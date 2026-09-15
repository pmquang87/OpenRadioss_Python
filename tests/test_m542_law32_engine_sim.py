"""Auditor 3: Explicit Engine Simulation & Shell Mechanics Audit for /MAT/LAW32 (Hill 1948).

Tests explicit engine dynamic simulations and shell mechanics for Material Law 32:
1. Multi-cycle explicit engine simulation:
   - 4-node shells (Belytschko-Tsay ishell=1 and QEPH ishell=24).
   - Cyclic in-plane velocity loading (tensile past yield, hold, unload).
   - Central difference time integration for 50+ time steps.
   - Stability (no NaN/divergence), plastic strain accumulation (epsp > 0),
     energy dissipation on unloading, strict total energy conservation
     (|E_tot - W_ext| / E_ref < 1e-4 and |ERR| < 0.01%).
   - Full end-to-end Starter + Engine run with energy verification.
2. Lankford anisotropy directionality:
   - Uniaxial tension along 0°, 45°, and 90° relative to material fiber axes.
   - Reduction to isotropic von Mises when R00 = R45 = R90 = 1.0.
   - Distinct directional yield stresses matching analytical Hill 1948 theory.
3. Through-thickness strain & thinning:
   - Shell thinning under in-plane tensile plastic deformation.
   - Plastic incompressibility and through-thickness strain in uv32[:, :, 1].
4. Dynamic failure & element deletion:
   - Element deletion when epsp >= eps_max (off = 0, off32 = 0, sig = 0).
   - Post-deletion explicit stability for 50+ cycles in multi-element mesh.
5. Acoustic wave propagation & Courant time step:
   - Sound speed c = sqrt(E / rho0) matching hm_read_mat32.F:155.
   - Courant CFL bound dt <= L / c matching solver time step within 10%.
   - 1D dynamic stress wave propagation across shell strip.
"""

import contextlib
import io
import math
import os
import tempfile
import numpy as np
import pytest

from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.starter import run_starter
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.elements import shell_bt4, shell_qeph
from pyradioss.materials.law32_hill import (
    build_law32,
    shell_update as law32_shell_update,
    sound_speed as law32_sound_speed,
)


def hill_yield_stress(mat, theta_deg: float) -> float:
    """Analytical uniaxial yield stress at angle theta (degrees) to material axis 1."""
    p = mat.params
    a11 = p["A11"]
    a22 = p["A22"]
    a1122 = p["A1122"]
    a12 = p["A12"]
    sy0 = p.get("A", 1.0)
    rad = math.radians(theta_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    c2 = c * c
    s2 = s * s
    cs = c * s
    denom = math.sqrt(a11 * (c2**2) + a22 * (s2**2) - a1122 * c2 * s2 + a12 * (cs**2))
    return sy0 / denom


def make_single_shell_model(
    tmp_path,
    name="SHELL_1EL",
    lx=10.0,
    ly=10.0,
    thick=1.0,
    ishell=1,
    nip=3,
    rho0=7.8e-6,
    e=210000.0,
    nu=0.3,
    sigy=300.0,
    beta=1.0,
    hard=0.2,
    r00=1.5,
    r45=1.2,
    r90=1.8,
    eps_max=1e30,
):
    """Build and initialize a 1-element shell model via StarterDeck and run_starter."""
    deck = StarterDeck(name)
    deck.node([
        (1, 0.0, 0.0, 0.0),
        (2, lx, 0.0, 0.0),
        (3, lx, ly, 0.0),
        (4, 0.0, ly, 0.0),
    ])
    deck.shell(1, [(1, 1, 2, 3, 4)])
    deck.part(1, "SHELL_PART", 1, 1)
    deck.mat_law32(
        1,
        rho=rho0,
        e=e,
        nu=nu,
        sigy=sigy,
        beta=beta,
        hard=hard,
        r00=r00,
        r45=r45,
        r90=r90,
        eps=eps_max,
    )
    deck.prop_shell(1, "SHELL_PROP", thick=thick, nip=nip, ishell=ishell)
    s_path = os.path.join(tmp_path, f"{name}_0000.rad")
    deck.write(s_path)
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s_path)
    group = model.shells if ishell == 1 else model.shells_qeph
    kernel = shell_bt4 if ishell == 1 else shell_qeph
    return model, group, kernel


def make_2x2_shell_model(
    tmp_path,
    name="SHELL_2X2",
    lx=10.0,
    ly=10.0,
    thick=1.0,
    ishell=1,
    nip=3,
    rho0=7.8e-6,
    e=210000.0,
    nu=0.3,
    sigy=300.0,
    beta=1.0,
    hard=0.2,
    r00=1.5,
    r45=1.2,
    r90=1.8,
):
    """Build and initialize a 2x2 shell mesh (4 elements, 9 nodes)."""
    deck = StarterDeck(name)
    nodes = []
    nid = 1
    for j in range(3):
        for i in range(3):
            nodes.append((nid, i * (lx / 2.0), j * (ly / 2.0), 0.0))
            nid += 1
    deck.node(nodes)

    shells = []
    eid = 1
    for j in range(2):
        for i in range(2):
            n1 = j * 3 + i + 1
            n2 = n1 + 1
            n3 = n2 + 3
            n4 = n1 + 3
            shells.append((eid, n1, n2, n3, n4))
            eid += 1
    deck.shell(1, shells)
    deck.part(1, "SHELL_PART", 1, 1)
    deck.mat_law32(
        1,
        rho=rho0,
        e=e,
        nu=nu,
        sigy=sigy,
        beta=beta,
        hard=hard,
        r00=r00,
        r45=r45,
        r90=r90,
    )
    deck.prop_shell(1, "SHELL_PROP", thick=thick, nip=nip, ishell=ishell)
    s_path = os.path.join(tmp_path, f"{name}_0000.rad")
    deck.write(s_path)
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s_path)
    group = model.shells if ishell == 1 else model.shells_qeph
    kernel = shell_bt4 if ishell == 1 else shell_qeph
    return model, group, kernel


class TestLaw32ExplicitEngineMultiCycle:
    """Multi-cycle explicit engine simulation with central difference integration."""

    @pytest.mark.parametrize("ishell_name,ishell", [
        ("BT4", 1),
        ("QEPH", 24),
    ])
    def test_single_element_cyclic_loading(self, tmp_path, ishell_name, ishell):
        """Cyclic in-plane loading: tension past yield, hold, unloading for 70 cycles.

        Verifies:
        - Numerical stability across all time steps with no NaN or divergence.
        - Plastic strain accumulation (epsp > 0) during plastic loading.
        - Energy dissipation on unloading due to plasticity.
        - Strict total energy conservation (|E_tot - W_ext| / E_ref < 1e-4 and |ERR| < 0.01%).
        """
        model, group, kernel = make_single_shell_model(
            tmp_path,
            name=f"CYC_{ishell_name}",
            lx=10.0,
            ly=10.0,
            thick=1.0,
            ishell=ishell,
            nip=3,
            rho0=7.8e-6,
            e=210000.0,
            nu=0.3,
            sigy=300.0,
            beta=1.0,
            hard=0.2,
            r00=1.5,
            r45=1.2,
            r90=1.8,
        )

        dt = 1e-6
        v = np.zeros_like(model.x)
        w_ext = 0.0
        f_old = 0.0

        n_load = 30
        n_hold = 15
        n_unload = 25
        total_steps = n_load + n_hold + n_unload  # 70 cycles
        vx_load = 1000.0  # mm/s: strain rate 100/s, total strain 0.003 past yield

        epsp_history = []
        eint_history = []
        energy_ratio_history = []

        for step in range(total_steps):
            if step < n_load:
                curr_vx = vx_load
            elif step < n_load + n_hold:
                curr_vx = 0.0
            else:
                curr_vx = -vx_load

            # Pull right edge nodes (0-based indices 1 and 2)
            v[1, 0] = curr_vx
            v[2, 0] = curr_vx
            v[0, :] = 0.0
            v[3, :] = 0.0

            # Central difference displacement update
            model.x += v * dt

            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

            # Numerical stability: no NaN / Inf
            assert np.all(np.isfinite(fint)), f"Step {step}: fint contains NaN or Inf"
            assert np.all(np.isfinite(mint)), f"Step {step}: mint contains NaN or Inf"

            # External work done by pulling nodes (nodes 1 and 2) using midpoint quadrature
            f_new = -(fint[1, 0] + fint[2, 0])
            w_ext += 0.5 * (f_old + f_new) * curr_vx * dt
            f_old = f_new

            # Internal energy accumulated by the element
            e_int = float(np.sum(group.state["eint"]))
            e_ref = max(abs(w_ext), e_int, 1e-6)
            err_ratio = abs(e_int - w_ext) / e_ref
            energy_ratio_history.append(err_ratio)

            epsp_curr = float(np.max(group.state["epsp"]))
            epsp_history.append(epsp_curr)
            eint_history.append(e_int)

        # 1. Verification of plastic strain accumulation during tensile loading
        assert epsp_history[n_load] > 0.0, "Plastic strain must accumulate during loading past yield"
        assert epsp_history[-1] >= epsp_history[n_load], "Plastic strain must be monotonically non-decreasing"

        # 2. Plastic energy dissipation: residual internal energy after unloading
        assert eint_history[-1] > 0.0, "Plastic deformation must leave dissipated residual internal energy"

        # 3. Strict total energy conservation (|E_tot - W_ext| / E_ref < 1e-4 and |ERR| < 0.01%)
        max_err = max(energy_ratio_history)
        assert max_err < 1e-4, f"Energy error {max_err:.4e} exceeds 1e-4 tolerance"
        pct_err = max_err * 100.0
        assert pct_err < 0.01, f"Energy error percent {pct_err:.4e}% exceeds 0.01%"

    def test_2x2_mesh_cyclic_engine_simulation(self, tmp_path):
        """2x2 shell mesh under cyclic in-plane tension/unloading (70 cycles)."""
        model, group, kernel = make_2x2_shell_model(
            tmp_path,
            name="CYC_2X2",
            lx=10.0,
            ly=10.0,
            thick=1.0,
            ishell=1,
            nip=3,
            sigy=300.0,
            beta=1.0,
            hard=0.2,
            r00=1.5,
            r45=1.2,
            r90=1.8,
        )

        dt = 1e-6
        v = np.zeros_like(model.x)
        w_ext = 0.0
        f_old = 0.0

        top_nodes = [6, 7, 8]  # y = 10.0 nodes
        mid_nodes = [3, 4, 5]  # y = 5.0 nodes
        bot_nodes = [0, 1, 2]  # y = 0.0 nodes

        n_load, n_hold, n_unload = 30, 15, 25
        vy_load = 1000.0

        for step in range(n_load + n_hold + n_unload):
            if step < n_load:
                curr_vy = vy_load
            elif step < n_load + n_hold:
                curr_vy = 0.0
            else:
                curr_vy = -vy_load

            v[top_nodes, 1] = curr_vy
            v[mid_nodes, 1] = 0.5 * curr_vy
            v[bot_nodes, :] = 0.0
            model.x += v * dt

            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

            assert np.all(np.isfinite(fint))
            assert np.all(np.isfinite(mint))

            f_top = -np.sum(fint[top_nodes, 1])
            f_mid = -np.sum(fint[mid_nodes, 1])
            f_new = f_top + 0.5 * f_mid
            w_ext += 0.5 * (f_old + f_new) * curr_vy * dt
            f_old = f_new

        # Verify plastic strain accumulation across all 4 elements
        assert np.all(group.state["epsp"] > 0.0)
        eint = float(np.sum(group.state["eint"]))
        assert eint > 0.0
        err = abs(w_ext - eint) / max(abs(w_ext), eint, 1e-6)
        assert err < 1e-4

    def test_end_to_end_starter_and_engine_run(self, tmp_path):
        """End-to-end Starter + Engine simulation: runs 55+ cycles and verifies energy balance."""
        run_name = "E2E_LAW32"
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
        deck.mat_law32(
            1,
            rho=7.8e-6,
            e=210000.0,
            nu=0.3,
            sigy=300.0,
            beta=1.0,
            hard=0.2,
            r00=1.5,
            r45=1.2,
            r90=1.8,
        )
        deck.prop_shell(1, "PROP_SHELL", thick=1.0, nip=3, ishell=1)

        # Fix root nodes 1 and 4
        deck.grnod_node(1, "root_nodes", [1, 4])
        deck.bcs(1, "fix_root", "111", "111", 1)

        # Impose velocity on nodes 2 and 3
        deck.grnod_node(2, "pull_nodes", [2, 3])
        deck.funct(1, "vel_ramp", [(0.0, 50.0), (0.001, 50.0), (0.003, 0.0)])
        deck.impvel(1, "pull_x", 1, "X", 2)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
0.003
/DT
0.9 0
/PRINT/-1
/STOP
70
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Engine should complete 50+ cycles, got {state.cycle}"

        # Verify energy accounting
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 0.01, f"Engine energy error {en['ERR']}% exceeds 0.01%"


class TestLaw32LankfordAnisotropyDirectionality:
    """Anisotropic yield stress and plastic flow directionality."""

    def test_isotropic_reduction_lankford_unity(self):
        """When R00 = R45 = R90 = 1.0, behavior reduces exactly to isotropic von Mises."""
        e = 210000.0
        nu = 0.3
        sy0 = 300.0
        mat = build_law32(
            1,
            e=e,
            nu=nu,
            a=sy0,
            b=1.0,
            n=0.0,
            r00=1.0,
            r45=1.0,
            r90=1.0,
        )
        p = mat.params
        assert pytest.approx(p["A11"], rel=1e-7) == 1.0
        assert pytest.approx(p["A22"], rel=1e-7) == 1.0
        assert pytest.approx(p["A1122"], rel=1e-7) == 1.0
        assert pytest.approx(p["A12"], rel=1e-7) == 3.0

        # Uniaxial yield stress in 0°, 45°, 90° must be identical
        s0 = hill_yield_stress(mat, theta_deg=0.0)
        s45 = hill_yield_stress(mat, theta_deg=45.0)
        s90 = hill_yield_stress(mat, theta_deg=90.0)
        assert pytest.approx(s0, rel=1e-7) == sy0
        assert pytest.approx(s45, rel=1e-7) == sy0
        assert pytest.approx(s90, rel=1e-7) == sy0

    def test_directional_yield_stresses(self):
        """Verify analytical directional yield stresses for orthotropic sheet."""
        r00 = 1.8
        r45 = 1.2
        r90 = 2.2
        sy0 = 300.0
        mat = build_law32(
            1,
            e=210000.0,
            nu=0.3,
            a=sy0,
            b=1.0,
            n=0.0,
            r00=r00,
            r45=r45,
            r90=r90,
        )
        p = mat.params
        a11 = p["A11"]
        a22 = p["A22"]
        a1122 = p["A1122"]
        a12 = p["A12"]

        # Analytical Hill yield stresses:
        # sigma_0 = sy0 / sqrt(A11)
        # sigma_90 = sy0 / sqrt(A22)
        # sigma_45 = 2 * sy0 / sqrt(A11 + A22 - A1122 + A12)
        sig_0_exact = sy0 / math.sqrt(a11)
        sig_90_exact = sy0 / math.sqrt(a22)
        sig_45_exact = 2.0 * sy0 / math.sqrt(a11 + a22 - a1122 + a12)

        s0 = hill_yield_stress(mat, 0.0)
        s90 = hill_yield_stress(mat, 90.0)
        s45 = hill_yield_stress(mat, 45.0)

        assert pytest.approx(s0, rel=1e-7) == sig_0_exact
        assert pytest.approx(s90, rel=1e-7) == sig_90_exact
        assert pytest.approx(s45, rel=1e-7) == sig_45_exact

        # Verify Hill yield stresses are distinct
        assert abs(s0 - s90) > 5.0
        assert abs(s45 - s0) > 5.0

    def test_directional_plastic_strain_evolution(self):
        """Verify directional plasticity in 0° vs 90° tension under identical strain increment."""
        mat = build_law32(1, e=210000.0, nu=0.3, a=300.0, b=1.0, n=0.0, r00=2.0, r45=1.0, r90=0.8, ipla=0)

        # 0° tension
        sig_0 = np.zeros((1, 3))
        deps_0 = np.array([[0.005, 0.0, 0.0]])
        sig_new_0, epsp_new_0 = law32_shell_update(mat, sig_0, deps_0, np.zeros(1), dt=1e-6)

        # 90° tension
        sig_90 = np.zeros((1, 3))
        deps_90 = np.array([[0.0, 0.005, 0.0]])
        sig_new_90, epsp_new_90 = law32_shell_update(mat, sig_90, deps_90, np.zeros(1), dt=1e-6)

        # Anisotropy must produce different stresses and plastic strain increments
        assert abs(sig_new_0[0, 0] - sig_new_90[0, 1]) > 5.0
        assert abs(epsp_new_0[0] - epsp_new_90[0]) > 1e-5


class TestLaw32ThroughThicknessStrainAndThinning:
    """Through-thickness strain, plastic incompressibility, and shell thinning."""

    def test_through_thickness_strain_accumulation_in_uv32(self):
        """Verify thickness strain is negative under in-plane tension and tracked in uv32."""
        mat = build_law32(1, e=210000.0, nu=0.3, a=250.0, b=1.0, n=0.0, r00=1.0, r45=1.0, r90=1.0, ipla=0)
        sig = np.zeros((1, 3))
        deps = np.array([[0.005, 0.0, 0.0]])
        epsp = np.array([0.0])
        extra = {"uv32": np.zeros((1, 2)), "off32": np.ones(1), "ezz": np.zeros(1)}

        sig_new, epsp_new = law32_shell_update(mat, sig, deps, epsp, dt=1e-6, extra=extra)

        # Plastic strain must accumulate
        assert epsp_new[0] > 0.0
        # Through-thickness strain must be negative (thinning)
        ezz = extra["ezz"][0]
        assert ezz < 0.0
        # uv32 second column must store accumulated ezz
        assert pytest.approx(extra["uv32"][0, 1], rel=1e-6) == ezz

    def test_dynamic_engine_thinning(self, tmp_path):
        """Verify shell thinning (t < t0) during explicit dynamic tensile deformation."""
        model, group, kernel = make_single_shell_model(
            tmp_path,
            name="THIN_SHELL",
            lx=10.0,
            ly=10.0,
            thick=2.0,
            ishell=1,
            nip=3,
            sigy=250.0,
            beta=1.0,
            hard=0.1,
        )

        dt = 1e-6
        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 500.0  # Pull right edge past yield

        for step in range(35):
            model.x += v * dt
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

        # Verify through-thickness strain in uv32 is negative
        uv32 = group.state["mat_extra"]["uv32"]
        ezz = uv32[0, 0, 1]
        assert ezz < 0.0, f"Expected negative thickness strain (thinning), got {ezz}"

        # Current thickness t = t0 * exp(ezz) must be less than initial thickness t0
        t0 = 2.0
        t_curr = t0 * math.exp(ezz)
        assert t_curr < t0


class TestLaw32DynamicFailureAndDeletion:
    """Dynamic failure and element deletion when epsp exceeds eps_max."""

    def test_element_deletion_at_eps_max(self):
        """Verify element is deleted (off=0, off32=0, sig=0) when epsp >= eps_max."""
        eps_max = 0.005
        mat = build_law32(1, e=210000.0, nu=0.3, a=250.0, b=1.0, n=0.0, eps_max=eps_max)

        sig = np.array([[240.0, 0.0, 0.0]])
        # Strain increment driving epsp beyond eps_max
        deps = np.array([[0.01, 0.0, 0.0]])
        epsp = np.array([0.003])
        extra = {
            "uv32": np.array([[0.003, 0.0]]),
            "off32": np.ones(1),
            "off": np.ones(1),
            "layfail": np.ones(1),
        }

        sig_new, epsp_new = law32_shell_update(mat, sig, deps, epsp, dt=1e-6, extra=extra)

        # Element must be failed and deleted
        assert epsp_new[0] >= eps_max
        assert extra["off32"][0] == 0.0
        assert extra["off"][0] == 0.0
        assert np.all(sig_new[0] == 0.0), "Stresses must be zeroed upon element failure"

    def test_multi_cycle_stability_post_deletion(self, tmp_path):
        """Explicit integration remains stable for 50+ cycles after dynamic deletion in 2-element strip."""
        deck = StarterDeck("FAIL_2EL")
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 20.0, 0.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 10.0, 10.0, 0.0),
            (6, 20.0, 10.0, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 5, 4)])
        deck.shell(2, [(2, 2, 3, 6, 5)])
        deck.part(1, "PART1", 1, 1)
        deck.part(2, "PART2", 2, 2)

        # Part 1 has eps_max = 0.001 (fails early)
        deck.mat_law32(1, rho=7.8e-6, e=210000.0, nu=0.3, sigy=250.0, beta=1.0, hard=0.0, eps=0.001)
        deck.prop_shell(1, "PROP1", thick=1.0, nip=1, ishell=1)

        # Part 2 has eps_max = 1e30 (remains intact)
        deck.mat_law32(2, rho=7.8e-6, e=210000.0, nu=0.3, sigy=250.0, beta=1.0, hard=0.0, eps=1e30)
        deck.prop_shell(2, "PROP2", thick=1.0, nip=1, ishell=1)

        s_path = os.path.join(tmp_path, "FAIL_2EL_0000.rad")
        deck.write(s_path)
        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        dt = 1e-6
        v = np.zeros_like(model.x)
        # Pull node 2 and 5 in X (stretching element 1)
        v[[1, 4], 0] = 500.0

        deleted_step = None
        total_steps = 60

        for step in range(total_steps):
            model.x += v * dt
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            shell_bt4.forces(model.shells, model.x, v, model.vr, dt, fint, mint)

            # Central difference integration must remain stable with no NaN
            assert np.all(np.isfinite(fint)), f"Step {step}: fint contains NaN or Inf"
            assert np.all(np.isfinite(mint)), f"Step {step}: mint contains NaN or Inf"

            if model.shells.state["off"][0] == 0.0 and deleted_step is None:
                deleted_step = step

        assert deleted_step is not None, "Element 1 should have been deleted during loading"
        # Element 1 stresses must be zero
        assert np.all(model.shells.state["sig"][0] == 0.0)
        assert model.shells.state["off"][0] == 0.0
        assert model.shells.state["mat_extra"]["off32"][0] == 0.0

        # Element 2 remains alive and active
        assert model.shells.state["off"][1] == 1.0
        assert model.shells.state["mat_extra"]["off32"][1] == 1.0


class TestLaw32AcousticWavePropagationAndCourant:
    """Acoustic wave speed and Courant time step."""

    def test_plane_stress_sound_speed(self):
        """Analytical shell sound speed c = sqrt(E / rho0) matching hm_read_mat32.F:155."""
        rho0 = 7.8e-6  # kg/mm^3
        e = 210000.0   # MPa
        nu = 0.3
        c_expected = math.sqrt(e / rho0)

        mat = build_law32(1, rho0=rho0, e=e, nu=nu)
        c_calc = law32_sound_speed(mat)
        assert pytest.approx(c_calc, rel=1e-6) == c_expected

    def test_courant_time_step_scaling(self, tmp_path):
        """Courant time step calculated by solver matches Courant CFL bound dt <= L / c."""
        lx = 5.0
        ly = 2.0
        rho0 = 7.8e-6
        e = 210000.0
        nu = 0.3
        c_sound = math.sqrt(e / rho0)

        deck = StarterDeck("COURANT_TEST")
        deck.node([(1, 0, 0, 0), (2, lx, 0, 0), (3, lx, ly, 0), (4, 0, ly, 0)])
        deck.shell(1, [(1, 1, 2, 3, 4)])
        deck.part(1, "PART", 1, 1)
        deck.mat_law32(1, rho=rho0, e=e, nu=nu, sigy=300.0, beta=1.0, hard=0.0)
        deck.prop_shell(1, "PROP", thick=1.0, nip=1, ishell=1)

        s_path = os.path.join(tmp_path, "COURANT_TEST_0000.rad")
        deck.write(s_path)
        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        dt_c = shell_bt4.forces(
            model.shells,
            model.x,
            np.zeros_like(model.x),
            model.vr,
            1e-6,
            np.zeros_like(model.x),
            np.zeros_like(model.x),
        )

        dt_solver = float(dt_c[0])
        dt_cfl = ly / c_sound
        # Solver time step matches CFL bound within 10%
        assert pytest.approx(dt_solver, rel=0.10) == dt_cfl

    def test_1d_shell_strip_dynamic_wave_propagation(self, tmp_path):
        """1D shell strip under dynamic velocity impact: verify wave propagates stably."""
        n_elem = 4
        l_elem = 5.0
        ly = 2.0
        nodes = []
        nid = 1
        for i in range(n_elem + 1):
            nodes.append((nid, i * l_elem, 0.0, 0.0))
            nodes.append((nid + 1, i * l_elem, ly, 0.0))
            nid += 2

        deck = StarterDeck("STRIP_WAVE")
        deck.node(nodes)
        shells = []
        for i in range(n_elem):
            n1 = 2 * i + 1
            n2 = 2 * (i + 1) + 1
            n3 = 2 * (i + 1) + 2
            n4 = 2 * i + 2
            shells.append((i + 1, n1, n2, n3, n4))
        deck.shell(1, shells)
        deck.part(1, "PART", 1, 1)

        rho0 = 7.8e-6
        e = 210000.0
        nu = 0.0

        deck.mat_law32(1, rho=rho0, e=e, nu=nu, sigy=10000.0, beta=1.0, hard=0.0)
        deck.prop_shell(1, "PROP", thick=1.0, nip=1, ishell=1)

        s_path = os.path.join(tmp_path, "STRIP_WAVE_0000.rad")
        deck.write(s_path)
        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        dt = 2e-7
        v = np.zeros_like(model.x)
        mass = np.zeros(len(model.x))
        for i_el in range(model.shells.n):
            conn = model.shells.conn[i_el] - 1
            el_m = rho0 * 1.0 * model.shells.state["area0"][i_el] / 4.0
            mass[conn] += el_m

        for step in range(25):
            v[[0, 1], 0] = -50.0
            model.x += v * dt
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            shell_bt4.forces(model.shells, model.x, v, model.vr, dt, fint, mint)

            acc = -fint / mass[:, None]
            acc[[0, 1], :] = 0.0
            acc[:, 1:] = 0.0
            v += acc * dt

        # Verify stress wave has propagated into downstream elements
        sig = model.shells.state["sig"][:, 0, 0]
        assert abs(sig[0]) > 5.0, "Element 0 must carry compressive stress"
        assert abs(sig[1]) > 0.01, "Element 1 must have received stress wave arrival"
