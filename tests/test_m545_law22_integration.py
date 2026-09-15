"""
Integration tests for /MAT/LAW22 (/MAT/DAMA, /MAT/PLAS_DAMA).
Milestone M545: Single-element explicit solver steps (BT4 shell and Hexa8 solid) and DeckWriter roundtrip.

Covers:
  1. DeckWriter roundtrip:
     - StarterDeck.mat_law22 roundtrip verification
     - StarterDeck.mat_dama roundtrip verification
     - StarterDeck.mat_plas_dama roundtrip verification
  2. Single-element explicit solver step with BT4 shell:
     - Elastic tension step (verify stress, internal forces balance, positive timestep)
     - Plastic yield step (verify yielding, epsp > 0, force equilibrium)
     - Damage softening step (verify epsp > eps_dam, modulus degradation alpe22 < 1.0)
  3. Single-element explicit solver step with Hexa8 solid:
     - Elastic tension step (verify 3D stress, internal forces balance, positive timestep)
     - Plastic yield step (verify yielding, epsp > 0, force equilibrium)
     - Damage softening step (verify epsp > eps_dam, modulus degradation alpe22 < 1.0)
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

class TestLaw22DeckWriterRoundtrip:
    """Verify StarterDeck roundtrip for mat_law22 and all synonyms."""

    def test_deck_writer_roundtrip_mat_law22(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_law22 through emitter, reader, and parser."""
        d = StarterDeck("DECK_LAW22")
        d.title("LAW22_ROUNDTRIP")
        d.mat_law22(
            mid=1,
            title="Steel_DP600",
            rho=7.85e-9,
            refer_rho=7.85e-9,
            e=210000.0,
            nu=0.3,
            a=350.0,
            b=450.0,
            n=0.5,
            eps_max=0.25,
            sig_max=900.0,
            c=0.03,
            eps_dot_0=1.0,
            icc=1,
            eps_dam=0.05,
            e_tan=-10000.0,
        )
        rendered = d.render()
        assert "/MAT/LAW22/1" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 1 in model.mat_law22s
        mat = model.mat_law22s[1]
        assert mat.id == 1
        assert mat.rho0 == pytest.approx(7.85e-9)
        assert mat.rhor == pytest.approx(7.85e-9)
        assert mat.e == pytest.approx(210000.0)
        assert mat.nu == pytest.approx(0.3)
        assert mat.a == pytest.approx(350.0)
        assert mat.b == pytest.approx(450.0)
        assert mat.n == pytest.approx(0.5)
        assert mat.eps_max == pytest.approx(0.25)
        assert mat.sig_max == pytest.approx(900.0)
        assert mat.c == pytest.approx(0.03)
        assert mat.eps_dot_0 == pytest.approx(1.0)
        assert mat.icc == 1
        assert mat.eps_dam == pytest.approx(0.05)
        assert mat.e_tan == pytest.approx(-10000.0)

        # Physical material in model.materials
        assert 1 in model.materials
        m = model.materials[1]
        assert m.law == 22
        assert m.params["E"] == pytest.approx(210000.0)
        assert m.params["E_tan"] == pytest.approx(-10000.0)

    def test_deck_writer_roundtrip_mat_dama(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_dama alias."""
        d = StarterDeck("DECK_DAMA")
        d.title("DAMA_ROUNDTRIP")
        d.mat_dama(
            mid=2,
            title="Alu_DAMA",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            a=180.0,
            b=220.0,
            n=0.4,
            eps_max=0.15,
            sig_max=400.0,
            c=0.0,
            eps_dot_0=1.0,
            icc=1,
            eps_dam=0.08,
            e_tan=-5000.0,
        )
        rendered = d.render()
        assert "/MAT/DAMA/2" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 2 in model.mat_damas
        mat = model.mat_damas[2]
        assert mat.id == 2
        assert mat.rho0 == pytest.approx(2.7e-9)
        assert mat.e == pytest.approx(70000.0)
        assert mat.nu == pytest.approx(0.33)
        assert mat.a == pytest.approx(180.0)
        assert mat.b == pytest.approx(220.0)
        assert mat.n == pytest.approx(0.4)
        assert mat.eps_dam == pytest.approx(0.08)
        assert mat.e_tan == pytest.approx(-5000.0)
        assert 2 in model.materials
        assert model.materials[2].law == 22

    def test_deck_writer_roundtrip_mat_plas_dama(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_plas_dama alias."""
        d = StarterDeck("DECK_PLAS_DAMA")
        d.title("PLAS_DAMA_ROUNDTRIP")
        d.mat_plas_dama(
            mid=3,
            title="Polymer_DAMA",
            rho=1.2e-9,
            e=3000.0,
            nu=0.38,
            a=50.0,
            b=30.0,
            n=0.6,
            eps_max=0.5,
            sig_max=100.0,
            c=0.05,
            eps_dot_0=0.5,
            icc=2,
            eps_dam=0.10,
            e_tan=-1000.0,
        )
        rendered = d.render()
        assert "/MAT/PLAS_DAMA/3" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 3 in model.mat_plas_damas
        mat = model.mat_plas_damas[3]
        assert mat.id == 3
        assert mat.rho0 == pytest.approx(1.2e-9)
        assert mat.e == pytest.approx(3000.0)
        assert mat.nu == pytest.approx(0.38)
        assert mat.a == pytest.approx(50.0)
        assert mat.b == pytest.approx(30.0)
        assert mat.n == pytest.approx(0.6)
        assert mat.eps_dam == pytest.approx(0.10)
        assert mat.e_tan == pytest.approx(-1000.0)
        assert 3 in model.materials
        assert model.materials[3].law == 22


# ============================================================================
# 2. Single-Element Explicit Solver Step: BT4 Shell
# ============================================================================

class TestLaw22ExplicitSolverBT4Shell:
    """Explicit solver step with single BT4 shell element using LAW22."""

    SHELL_DECK = """
/BEGIN
BT4_SHELL_LAW22
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
/MAT/LAW22/1
Steel_DP600_DAMA
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.25, 900.0
0.03, 1.0, 1
0.05, -10000.0
/END
"""

    def test_bt4_shell_elastic_step(self, tmp_path: Path):
        """Verify elastic stress update and force equilibrium for BT4 shell."""
        model, log = _build_model(self.SHELL_DECK, tmp_path)
        g = model.shells
        assert g.n == 1

        # Prescribe small uniaxial strain rate along x: vx = x * rate
        rate = 1.0e-3
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate  # nodes 2 and 3 move with vx = 1.0 * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1.0e-5

        dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)

        # Critical timestep must be positive
        assert dtc[0] > 0.0

        # In-plane stress in direction 1
        sig = g.state["sig"]
        assert sig[0, 0, 0] > 0.0  # Positive tensile stress along x
        assert sig[0, 0, 0] < 350.0  # Elastic regime (< a = 350 MPa)

        # Force balance: sum of internal forces on free body must be ~ 0
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)
        # Tension along x: nodes on x=1 pulled back (-x), nodes on x=0 pulled forward (+x)
        assert fint[1, 0] + fint[2, 0] < 0.0
        assert fint[0, 0] + fint[3, 0] > 0.0

        # Alpha is 1.0 (no damage) and epsp is 0.0
        assert g.state["mat_extra"]["alpe22"][0, 0] == pytest.approx(1.0)
        assert g.state["epsp"][0] == pytest.approx(0.0)

    def test_bt4_shell_plastic_yield_step(self, tmp_path: Path):
        """Verify plastic yielding and plastic strain accumulation under large strain."""
        model, log = _build_model(self.SHELL_DECK, tmp_path)
        g = model.shells

        # Large strain rate causing plastic flow
        rate = 5.0
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1.0e-3

        dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0
        # Plastic strain accumulated
        assert g.state["epsp"][0] > 0.0
        # Force balance on free body
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-6)

    def test_bt4_shell_damage_softening_step(self, tmp_path: Path):
        """Verify damage initiation and modulus degradation when epsp exceeds eps_dam."""
        model, log = _build_model(self.SHELL_DECK, tmp_path)
        g = model.shells

        # Preset plastic strain beyond eps_dam = 0.05
        g.state["mat_extra"]["epsp22"][0, 0] = 0.06
        g.state["epsp"][0] = 0.06

        rate = 2.0
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1.0e-3

        dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0
        # Degradation factor alpha must be strictly < 1.0
        alpe = g.state["mat_extra"]["alpe22"][0, 0]
        assert alpe < 1.0
        assert alpe > 0.0
        # Element remains active
        assert g.state["mat_extra"]["off22"][0, 0] == pytest.approx(1.0)


# ============================================================================
# 3. Single-Element Explicit Solver Step: Hexa8 Solid
# ============================================================================

class TestLaw22ExplicitSolverHexa8Solid:
    """Explicit solver step with single Hexa8 solid element using LAW22."""

    HEXA8_DECK = """
/BEGIN
HEXA8_SOLID_LAW22
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
/MAT/LAW22/1
Steel_DP600_DAMA
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.25, 900.0
0.03, 1.0, 1
0.05, -10000.0
/END
"""

    def test_hexa8_solid_elastic_step(self, tmp_path: Path):
        """Verify elastic 3D stress update and force equilibrium for Hexa8 solid."""
        model, log = _build_model(self.HEXA8_DECK, tmp_path)
        g = model.bricks
        assert g.n == 1

        # Uniaxial strain rate along x: vx = x * rate
        rate = 1.0e-3
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1.0e-5

        dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0

        # Normal stress in direction 1
        sig = g.state["sig"]
        assert sig[0, 0] > 0.0
        assert sig[0, 0] < 350.0  # Elastic (< a = 350 MPa)

        # Force equilibrium on free body
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)
        # Tensile reaction: nodes with x=1 pulled in -x, nodes with x=0 pulled in +x
        assert fint[[1, 2, 5, 6], 0].sum() < 0.0
        assert fint[[0, 3, 4, 7], 0].sum() > 0.0

        # Alpha is 1.0 and epsp is 0.0
        assert g.state["mat_extra"]["alpe22"][0] == pytest.approx(1.0)
        assert g.state["epsp"][0] == pytest.approx(0.0)

    def test_hexa8_solid_plastic_step(self, tmp_path: Path):
        """Verify 3D plastic yielding and plastic strain accumulation for Hexa8 solid."""
        model, log = _build_model(self.HEXA8_DECK, tmp_path)
        g = model.bricks

        # High strain rate to cause plastic deformation
        rate = 5.0
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1.0e-3

        dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0
        # Plastic strain accumulated
        assert g.state["epsp"][0] > 0.0
        # Force balance
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-6)

    def test_hexa8_solid_damage_softening_step(self, tmp_path: Path):
        """Verify 3D damage initiation and modulus degradation when epsp exceeds eps_dam."""
        model, log = _build_model(self.HEXA8_DECK, tmp_path)
        g = model.bricks

        # Preset plastic strain beyond eps_dam = 0.05
        g.state["mat_extra"]["epsp22"][0] = 0.06
        g.state["epsp"][0] = 0.06

        rate = 2.0
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 0] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        dt = 1.0e-3

        dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc[0] > 0.0
        # Degradation factor alpha must be strictly < 1.0
        alpe = g.state["mat_extra"]["alpe22"][0]
        assert alpe < 1.0
        assert alpe > 0.0
        # Element remains active
        assert g.state["mat_extra"]["off22"][0] == pytest.approx(1.0)
