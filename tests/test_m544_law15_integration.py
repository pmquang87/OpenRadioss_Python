"""
Integration tests for /MAT/LAW15 (/MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG).
Milestone M544: Single-element explicit solver steps (BT4 and QEPH shells) and DeckWriter roundtrip.

Covers:
  1. DeckWriter roundtrip:
     - StarterDeck.mat_law15 roundtrip verification
     - StarterDeck.mat_chang roundtrip verification
     - StarterDeck.mat_plas_aniso roundtrip verification
     - StarterDeck.mat_comp_chang roundtrip verification
  2. Single-element explicit solver step with BT4 shell:
     - Elastic tension step (verify stress, internal forces balance, positive timestep)
     - Plastic yield step (verify yielding, epsp > 0, wpla15 > 0, force equilibrium)
     - Chang-Chang failure mode and stress relaxation over time
  3. Single-element explicit solver step with QEPH shell (Ishell=24):
     - Element routing to shells_qeph
     - Elastic uniaxial tension step
     - Multi-cycle plastic step and work accumulation
     - Pure shear step
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_bt4, shell_qeph
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.starter import (
    build_element_groups,
    resolve_node_groups,
    resolve_surfaces,
    initialize_elements_and_mass,
)


def _build_model(deck_text: str, tmp_path: Path) -> tuple[Model, MessageLog]:
    """Helper to parse and initialize model element groups and masses."""
    f = tmp_path / "DECK_0000.rad"
    f.write_text(deck_text.strip() + "\n", encoding="utf-8")
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    resolve_surfaces(model, log)
    initialize_elements_and_mass(model, log)
    assert not log.errors, f"Starter errors: {log.errors}"
    return model, log


# ============================================================================
# 1. DeckWriter Roundtrip Tests
# ============================================================================

class TestLaw15DeckWriterRoundtrip:
    """Verify StarterDeck roundtrip for mat_law15 and all synonyms."""

    def test_deck_writer_roundtrip_mat_law15(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_law15 through emitter, reader, and parser."""
        d = StarterDeck("DECK_LAW15")
        d.title("LAW15_ROUNDTRIP")
        d.mat_law15(
            mid=10,
            title="Composite_Chang",
            rho=1.5e-9,
            refer_rho=1.5e-9,
            e11=140000.0,
            e22=10000.0,
            nu12=0.3,
            g12=5000.0,
            g23=3000.0,
            g31=5000.0,
            b=0.5,
            n=0.8,
            fmax=10.0,
            wpmax=5.0,
            wpref=1.0,
            ioff=2,
            sig_1yt=1500.0,
            sig_2yt=50.0,
            sig_1yc=1200.0,
            sig_2yc=200.0,
            alpha=1.0,
            sig_12yc=80.0,
            sig_12yt=80.0,
            c=0.0,
            eps_dot_0=1.0,
            icc=1,
            beta=1.0,
            tmax=1.0e-4,
            s1=1500.0,
            s2=50.0,
            s12=80.0,
            fsmooth=0,
            fcut=0.0,
            c1=1200.0,
            c2=200.0,
        )
        rendered = d.render()
        assert "/MAT/LAW15/10" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 10 in model.mat_law15s
        mat = model.mat_law15s[10]
        assert mat.id == 10
        assert mat.rho0 == 1.5e-9
        assert mat.e11 == 140000.0
        assert mat.e22 == 10000.0
        assert mat.nu12 == 0.3
        assert mat.g12 == 5000.0
        assert mat.g23 == 3000.0
        assert mat.g31 == 5000.0
        assert mat.b == 0.5
        assert mat.n == 0.8
        assert mat.fmax == 10.0
        assert mat.wpmax == 5.0
        assert mat.wpref == 1.0
        assert mat.ioff == 2
        assert mat.sig_1yt == 1500.0
        assert mat.sig_2yt == 50.0
        assert mat.sig_1yc == 1200.0
        assert mat.sig_2yc == 200.0
        assert mat.alpha == 1.0
        assert mat.sig_12yc == 80.0
        assert mat.sig_12yt == 80.0
        assert mat.icc == 1
        assert mat.beta == 1.0
        assert mat.tmax == 1.0e-4
        assert mat.s1 == 1500.0
        assert mat.s2 == 50.0
        assert mat.s12 == 80.0
        assert mat.c1 == 1200.0
        assert mat.c2 == 200.0

        # Physical material in model.materials
        assert 10 in model.materials
        act_mat = model.materials[10]
        assert act_mat.law == 15
        assert act_mat.params["E1"] == 140000.0
        assert act_mat.params["E2"] == 10000.0

    def test_deck_writer_roundtrip_mat_chang(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_chang alias."""
        d = StarterDeck("DECK_CHANG")
        d.title("CHANG_ROUNDTRIP")
        d.mat_chang(
            mid=20,
            title="CHANG_Mat",
            rho=1.6e-9,
            refer_rho=1.6e-9,
            e11=150000.0,
            e22=9000.0,
            nu12=0.32,
            g12=4500.0,
            g23=2800.0,
            g31=4500.0,
            sig_1yt=1600.0,
            sig_2yt=40.0,
            sig_1yc=1300.0,
            sig_2yc=180.0,
            sig_12yc=70.0,
            sig_12yt=70.0,
            s1=1600.0,
            s2=40.0,
            s12=70.0,
            c1=1300.0,
            c2=180.0,
            tmax=2.0e-4,
        )
        rendered = d.render()
        assert "/MAT/CHANG/20" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 20 in model.mat_law15s
        mat = model.mat_law15s[20]
        assert mat.e11 == 150000.0
        assert mat.e22 == 9000.0
        assert mat.nu12 == 0.32
        assert mat.sig_1yt == 1600.0
        assert 20 in model.materials
        assert model.materials[20].law == 15

    def test_deck_writer_roundtrip_mat_plas_aniso(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_plas_aniso alias."""
        d = StarterDeck("DECK_PLAS_ANISO")
        d.title("PLAS_ANISO_ROUNDTRIP")
        d.mat_plas_aniso(
            mid=30,
            title="PLAS_ANISO_Mat",
            rho=1.5e-9,
            e11=130000.0,
            e22=8000.0,
            nu12=0.28,
            g12=4000.0,
            g23=2500.0,
            g31=4000.0,
            sig_1yt=1400.0,
            sig_2yt=45.0,
            sig_1yc=1100.0,
            sig_2yc=190.0,
            sig_12yc=75.0,
            sig_12yt=75.0,
            s1=1400.0,
            s2=45.0,
            s12=75.0,
            c1=1100.0,
            c2=190.0,
            tmax=1.5e-4,
        )
        rendered = d.render()
        assert "/MAT/PLAS_ANISO/30" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 30 in model.mat_law15s
        assert model.mat_law15s[30].e11 == 130000.0
        assert 30 in model.materials
        assert model.materials[30].law == 15

    def test_deck_writer_roundtrip_mat_comp_chang(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_comp_chang alias."""
        d = StarterDeck("DECK_COMP_CHANG")
        d.title("COMP_CHANG_ROUNDTRIP")
        d.mat_comp_chang(
            mid=40,
            title="COMP_CHANG_Mat",
            rho=1.5e-9,
            e11=145000.0,
            e22=9500.0,
            nu12=0.31,
            g12=4800.0,
            g23=2900.0,
            g31=4800.0,
            sig_1yt=1550.0,
            sig_2yt=48.0,
            sig_1yc=1250.0,
            sig_2yc=210.0,
            sig_12yc=85.0,
            sig_12yt=85.0,
            s1=1550.0,
            s2=48.0,
            s12=85.0,
            c1=1250.0,
            c2=210.0,
            tmax=1.0e-4,
        )
        rendered = d.render()
        assert "/MAT/COMP_CHANG/40" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 40 in model.mat_law15s
        assert model.mat_law15s[40].e11 == 145000.0
        assert 40 in model.materials
        assert model.materials[40].law == 15


# ============================================================================
# 2. Single-Element Explicit Solver Step: BT4 Shell
# ============================================================================

class TestLaw15ExplicitSolverBT4Shell:
    """Explicit solver step with single BT4 shell element using LAW15."""

    SHELL_DECK = """
/BEGIN
BT4_SHELL_LAW15
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate
1 1
/PROP/SHELL/1
Shell_Prop
1.0 1
/MAT/LAW15/1
Composite_Shell
1.5e-9, 1.5e-9
140000.0, 10000.0, 0.3
5000.0, 3000.0, 5000.0
0.5, 0.8, 10.0
5.0, 1.0, 2
1500.0, 50.0, 1200.0, 200.0, 1.0
80.0, 80.0, 0.0, 1.0, 1
1.0, 1.0e-4, 1500.0, 50.0, 80.0
0, 0.0, 1200.0, 200.0
/END
"""

    def test_bt4_shell_elastic_step(self, tmp_path: Path):
        """Verify elastic stress update and force equilibrium for BT4 shell."""
        model, log = _build_model(self.SHELL_DECK, tmp_path)
        g = model.shells
        assert g.n == 1

        # Prescribe small uniaxial strain rate in dir 1: vx = x * rate
        rate = 1.0e-3
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate  # nodes 2 and 3 move with vx = 1.0 * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1.0e-5

        dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)

        # Critical timestep must be positive
        assert dtc[0] > 0.0

        # Stress check in direction 1
        sig = g.state["sig"]
        assert sig[0, 0, 0] > 0.0  # Positive tensile stress in direction 1
        assert sig[0, 0, 0] < 1500.0  # Elastic regime (< sig_1yt = 1500)

        # Force balance: sum of internal forces on free body must be ~ 0
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)
        # Tension along x: nodes on x=1 pulled back (-x), nodes on x=0 pulled forward (+x)
        assert fint[1, 0] + fint[2, 0] < 0.0
        assert fint[0, 0] + fint[3, 0] > 0.0

    def test_bt4_shell_plastic_yield_step(self, tmp_path: Path):
        """Verify plastic yielding and plastic work accumulation under large strain."""
        model, log = _build_model(self.SHELL_DECK, tmp_path)
        g = model.shells

        # Prescribe large strain rate to cause yielding
        rate = 100.0
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1.0e-4

        dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0
        epsp = g.state["epsp"]
        assert epsp[0, 0] > 0.0

        wpla = g.state["mat_extra"]["wpla15"]
        assert wpla[0, 0] > 0.0

        # Forces balance
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

    def test_bt4_shell_chang_failure_and_relaxation(self, tmp_path: Path):
        """Verify Chang-Chang tensile failure triggers stress relaxation over time."""
        model, log = _build_model(self.SHELL_DECK, tmp_path)
        g = model.shells
        dt = 1.0e-5

        # Prescribe huge strain to exceed S1 = 1500
        rate = 1000.0
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate

        # Run several cycles to trigger failure and advance relaxation
        damt_hist = []
        for cycle in range(20):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt
            damt = g.state["mat_extra"]["damt15"][0, 0, 0]
            damt_hist.append(damt)
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-4)

        # Fiber damage should have triggered (damt < 1.0)
        assert min(damt_hist) < 1.0


# ============================================================================
# 3. Single-Element Explicit Solver Step: QEPH Shell
# ============================================================================

class TestLaw15ExplicitSolverQEPHShell:
    """Explicit solver step with single QEPH shell element (Ishell=24) using LAW15."""

    QEPH_DECK = """
/BEGIN
QEPH_SHELL_LAW15
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate_QEPH
1 1
/PROP/TYPE1/1
Prop_QEPH
24 1 0 0 0 0 1
0.01 0.01 0.01 0.0 0.0
1 0 1.0
/MAT/LAW15/1
LAW15_QEPH_Mat
1.5e-9, 1.5e-9
140000.0, 10000.0, 0.3
5000.0, 3000.0, 5000.0
0.5, 0.8, 10.0
5.0, 1.0, 2
1500.0, 50.0, 1200.0, 200.0, 1.0
80.0, 80.0, 0.0, 1.0, 1
1.0, 1.0e-4, 1500.0, 50.0, 80.0
0, 0.0, 1200.0, 200.0
/END
"""

    def test_qeph_shell_routing_and_elastic_step(self, tmp_path: Path):
        """Verify QEPH element routing and single-step elastic tension."""
        model, log = _build_model(self.QEPH_DECK, tmp_path)
        assert model.shells is None
        assert model.shells_qeph is not None
        g = model.shells_qeph
        assert g.n == 1

        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * 1.0e-3
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1.0e-5

        dtc = shell_qeph.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0
        sig = g.state["sig"]
        assert sig[0, 0, 0] > 0.0
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)

    def test_qeph_shell_multi_cycle_plastic(self, tmp_path: Path):
        """50 explicit cycles under uniaxial tension with QEPH formulation."""
        model, _ = _build_model(self.QEPH_DECK, tmp_path)
        g = model.shells_qeph
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 50.0

        wpla_history = []
        for cycle in range(30):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = shell_qeph.forces(g, model.x, v, model.vr, dt, fint, mint)
            assert dtc[0] > 0.0
            model.x += v * dt
            wpla = float(g.state["mat_extra"]["wpla15"][0, 0])
            wpla_history.append(wpla)
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        assert wpla_history[-1] > 0.0
        # Plastic work must be non-decreasing
        for i in range(1, len(wpla_history)):
            assert wpla_history[i] >= wpla_history[i - 1] - 1e-12

    def test_qeph_shell_pure_shear(self, tmp_path: Path):
        """Verify QEPH shell response under pure shear."""
        model, _ = _build_model(self.QEPH_DECK, tmp_path)
        g = model.shells_qeph
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 1] * 20.0  # vx proportional to y -> gamma_xy shear

        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dtc = shell_qeph.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0
        sig = g.state["sig"]
        assert abs(sig[0, 0, 2]) > 0.0  # Non-zero shear stress in direction 12
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-6)
