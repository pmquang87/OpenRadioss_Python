"""
Integration tests for /MAT/LAW25 (/MAT/COMP_PLAS, /MAT/COMPSH, /MAT/TSAI_WU, /MAT/CRASURV).
Milestone M543: Single-element explicit solver step and DeckWriter roundtrip.

Covers:
  1. DeckWriter roundtrip:
     - StarterDeck.mat_law25 roundtrip verification
     - StarterDeck.mat_comp_plas roundtrip verification
     - CRASURV formulation roundtrip verification
  2. Single-element explicit solver step with BT4 shell:
     - Setup unit shell quad element (BT4) with LAW25
     - Prescribe in-plane uniaxial strain velocity field
     - Execute shell_bt4.forces step
     - Verify stress update, internal forces, force balance, and critical timestep
     - Plastic return and plastic work accumulation under large deformation
  3. Single-element explicit solver step with Hexa8 solid:
     - Setup unit cube brick element (Hexa8) with LAW25
     - Prescribe uniaxial strain velocity field
     - Execute solid_hexa8.forces step
     - Verify 3D stress update, internal forces, force balance, and critical timestep
     - Plastic return under high strain
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_bt4, solid_hexa8
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
    """Helper to write deck, parse, and initialize model element groups and masses."""
    f = tmp_path / "SOLVER_0000.rad"
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

class TestLaw25DeckWriterRoundtrip:
    """Verify StarterDeck roundtrip for mat_law25 and mat_comp_plas."""

    def test_deck_writer_roundtrip_mat_law25(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_law25 through emitter, reader and parser."""
        d = StarterDeck("DECK_LAW25")
        d.title("LAW25_ROUNDTRIP")
        d.mat_law25(
            mat_id=10,
            rho=1.5e-9,
            rho_ref=1.5e-9,
            e11=140000.0,
            e22=10000.0,
            e33=140000.0,
            nu12=0.3,
            g12=5000.0,
            g23=3000.0,
            g31=5000.0,
            eps_f1=0.02,
            eps_f2=0.01,
            eps_t1=0.005,
            eps_m1=0.02,
            eps_t2=0.003,
            eps_m2=0.01,
            dmax=0.95,
            wpmax=5.0,
            wpref=1.0,
            ioff=2,
            b=0.5,
            n=0.8,
            fmax=10.0,
            sig_1yt=1500.0,
            sig_2yt=50.0,
            sig_1yc=1200.0,
            sig_2yc=200.0,
            alpha=1.0,
            sig_12yc=80.0,
            sig_12yt=70.0,
            c=0.1,
            eps_rate_0=10.0,
            icc=1,
        )
        rendered = d.render()
        assert "/MAT/LAW25/10" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 10 in model.mat_law25s
        mat = model.mat_law25s[10]
        assert mat.id == 10
        assert mat.rho0 == 1.5e-9
        assert mat.e11 == 140000.0
        assert mat.e22 == 10000.0
        assert mat.nu12 == 0.3
        assert mat.g12 == 5000.0
        assert mat.sig_1yt == 1500.0
        assert mat.sig_2yt == 50.0
        assert mat.sig_1yc == 1200.0
        assert mat.sig_2yc == 200.0
        assert mat.sig_12yt == 70.0
        assert mat.sig_12yc == 80.0
        assert mat.ioff == 2
        assert mat.b == 0.5
        assert mat.n == 0.8

        # Active Material in model.materials
        assert 10 in model.materials
        act_mat = model.materials[10]
        assert act_mat.law == 25
        assert act_mat.params["e11"] == 140000.0

    def test_deck_writer_roundtrip_mat_comp_plas(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_comp_plas alias."""
        d = StarterDeck("DECK_COMP_PLAS")
        d.title("COMP_PLAS_ROUNDTRIP")
        d.mat_comp_plas(
            mat_id=20,
            rho=1.6e-9,
            rho_ref=1.6e-9,
            e11=150000.0,
            e22=9000.0,
            e33=150000.0,
            nu12=0.34,
            g12=4500.0,
            g23=3000.0,
            g31=4500.0,
            sig_1yt=1800.0,
            sig_2yt=40.0,
            sig_1yc=1200.0,
            sig_2yc=150.0,
            sig_12yt=60.0,
            sig_12yc=60.0,
        )
        rendered = d.render()
        assert "/MAT/COMP_PLAS/20" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        assert 20 in model.mat_law25s
        mat = model.mat_law25s[20]
        assert mat.e11 == 150000.0
        assert mat.e22 == 9000.0
        assert mat.nu12 == 0.34
        assert mat.sig_1yt == 1800.0

    def test_deck_writer_roundtrip_crasurv(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_crasurv formulation."""
        d = StarterDeck("DECK_CRASURV")
        d.title("CRASURV_ROUNDTRIP")
        d.mat_crasurv(
            mat_id=30,
            rho=1.5e-9,
            rho_ref=1.5e-9,
            e11=120000.0,
            e22=40000.0,
            e33=20000.0,
            nu12=0.25,
            g12=10000.0,
            g23=5000.0,
            g31=8000.0,
            b_1t=0.5,
            n_1t=0.8,
            sig_1maxt=1500.0,
            eps_1t1=0.01,
            eps_2t1=0.03,
            sig_rst1=200.0,
            sig_1yt=1000.0,
            sig_2yt=100.0,
            sig_1yc=800.0,
            sig_2yc=200.0,
            sig_12yt=80.0,
            sig_12yc=80.0,
        )
        rendered = d.render()
        assert "/MAT/CRASURV/30" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        assert 30 in model.mat_law25s
        mat = model.mat_law25s[30]
        assert mat.iform == 1
        assert mat.b_1t == 0.5
        assert mat.n_1t == 0.8
        assert mat.sig_1maxt == 1500.0
        assert mat.eps_1t1 == 0.01
        assert mat.eps_2t1 == 0.03
        assert mat.sig_rst1 == 200.0


# ============================================================================
# 2. Single-Element Explicit Solver Step: BT4 Shell
# ============================================================================

class TestLaw25ExplicitSolverBT4Shell:
    """Explicit solver step with single BT4 shell element using LAW25."""

    SHELL_DECK = """
/BEGIN
BT4_SHELL_LAW25
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
/MAT/LAW25/1
Composite_Shell
1.5e-9, 1.5e-9
100000.0, 50000.0, 0.2, 0, 100000.0
20000.0, 10000.0, 15000.0, 0.05, 0.05
0.01, 0.05, 0.01, 0.05, 0.95
10.0, 1.0, 0
0.5, 0.8, 10.0
200.0, 100.0, 200.0, 100.0, 1.0
80.0, 80.0, 0.0, 1.0, 0
/END
"""

    def test_bt4_shell_elastic_step(self, tmp_path: Path):
        """Verify elastic stress update and force equilibrium for BT4 shell."""
        model, log = _build_model(self.SHELL_DECK, tmp_path)
        g = model.shells
        assert g.n == 1

        # Prescribe uniaxial strain rate in dir 1: vx = x * rate
        rate = 1e-3
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate  # nodes 2 and 3 move with vx = 1.0 * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1e-4

        dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)

        # Critical timestep must be positive
        assert dtc[0] > 0.0

        # Stress check in direction 1
        sig = g.state["sig"]
        assert sig[0, 0, 0] > 0.0  # Positive tensile stress in direction 1
        assert sig[0, 0, 0] < 200.0  # Still in elastic regime (< sigyt1=200)

        # Force balance: sum of internal forces on free body must be ~ 0
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)
        # Tension along x: nodes on x=1 pulled back (-x), nodes on x=0 pulled forward (+x)
        assert fint[1, 0] + fint[2, 0] < 0.0
        assert fint[0, 0] + fint[3, 0] > 0.0

    def test_bt4_shell_plastic_yield_step(self, tmp_path: Path):
        """Verify plastic yielding and plastic work accumulation under large strain."""
        model, log = _build_model(self.SHELL_DECK, tmp_path)
        g = model.shells

        # Prescribe large strain rate to cause yielding (trial stress > 200)
        rate = 5.0
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1e-3

        dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0
        # Plastic strain and plastic work accumulated
        epsp = g.state["epsp"]
        assert epsp[0, 0] > 0.0

        wpla = g.state["mat_extra"]["wpla25"]
        assert wpla[0, 0] > 0.0

        # Forces balance
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-6)


# ============================================================================
# 3. Single-Element Explicit Solver Step: Hexa8 Solid
# ============================================================================

class TestLaw25ExplicitSolverHexa8Solid:
    """Explicit solver step with single Hexa8 solid element using LAW25."""

    HEXA8_DECK = """
/BEGIN
HEXA8_SOLID_LAW25
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
Cube
1 1
/PROP/SOLID/1
Solid_Prop
1.1 0.05 0.1
/MAT/LAW25/1
Composite_Solid
1.5e-9, 1.5e-9
100000.0, 50000.0, 0.2, 0, 30000.0
20000.0, 10000.0, 15000.0, 0.05, 0.05
0.01, 0.05, 0.01, 0.05, 0.95
10.0, 1.0, 0
0.5, 0.8, 10.0
200.0, 100.0, 200.0, 100.0, 1.0
80.0, 80.0, 0.0, 1.0, 0
/END
"""

    def test_hexa8_solid_elastic_step(self, tmp_path: Path):
        """Verify elastic 3D stress update and force equilibrium for Hexa8 solid."""
        model, log = _build_model(self.HEXA8_DECK, tmp_path)
        g = model.bricks
        assert g.n == 1

        # Uniaxial strain rate along x: vx = x * rate
        rate = 1e-3
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1e-4

        dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0

        # Normal stress in direction 1
        sig = g.state["sig"]
        assert sig[0, 0] > 0.0
        assert sig[0, 0] < 200.0  # Elastic

        # Force equilibrium on free body
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)
        # Tensile reaction: nodes with x=1 pulled in -x, nodes with x=0 pulled in +x
        assert fint[[1, 2, 5, 6], 0].sum() < 0.0
        assert fint[[0, 3, 4, 7], 0].sum() > 0.0

    def test_hexa8_solid_plastic_step(self, tmp_path: Path):
        """Verify 3D plastic yielding and work accumulation for Hexa8 solid."""
        model, log = _build_model(self.HEXA8_DECK, tmp_path)
        g = model.bricks

        # High strain rate to cause plastic deformation
        rate = 5.0
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1e-3

        dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0
        # Plastic strain accumulated
        assert g.state["epsp"][0] > 0.0
        assert g.state["mat_extra"]["wpla25"][0] > 0.0

        # Force balance
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-6)
