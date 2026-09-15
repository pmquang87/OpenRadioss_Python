"""Integration tests for /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL).

Milestone M547:
Covers:
  1. DeckWriter roundtrip:
     - StarterDeck.mat_law14 roundtrip verification
     - StarterDeck.mat_compso synonym roundtrip verification
     - StarterDeck.mat_comp_sol synonym roundtrip verification
  2. pyradioss.materials module dispatch:
     - materials.solid_update dispatches to law14_compso
     - materials.sound_speed dispatches to law14_compso
     - materials.consistent_solid_tangent dispatches to law14_compso
  3. extra_shapes verification:
     - returns required state array shapes for solid elements
  4. Consistent solid tangent:
     - returns exact 6x6 orthotropic D matrix in elastic regime
  5. Single-element explicit solver step with Hexa8 solid:
     - Elastic step (3D stress update, internal forces balance, positive timestep)
     - Tensile cracking damage step (dam14 increment, stress relaxation, force equilibrium)
     - Tsai-Wu plasticity step (plastic dissipation, epsp increment, force equilibrium)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials import law14_compso
from pyradioss.materials.law14_compso import (
    build_law14,
    consistent_solid_tangent,
    extra_shapes,
    solid_update,
    sound_speed,
)
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

class TestLaw14DeckWriterRoundtrip:
    """Verify StarterDeck roundtrip for mat_law14 and all synonyms."""

    def test_deck_writer_roundtrip_mat_law14(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_law14 through emitter, reader, and parser."""
        d = StarterDeck("DECK_LAW14")
        d.title("LAW14_ROUNDTRIP")
        d.mat_law14(
            mid=14,
            title="Composite_Solid",
            rho=1.55e-9,
            refer_rho=1.55e-9,
            ea=140000.0,
            eb=10000.0,
            ec=10000.0,
            prab=0.30,
            prbc=0.45,
            prca=0.02,
            gab=5000.0,
            gbc=3500.0,
            gca=5000.0,
            sigt1=1800.0,
            sigt2=40.0,
            sigt3=40.0,
            delta=0.08,
            cb=150.0,
            cn=0.5,
            fmax=1200.0,
            wplaref=1.0,
            sigyt1=2000.0,
            sigyt2=50.0,
            sigyc1=1500.0,
            sigyc2=200.0,
            sigyt12=80.0,
            sigyc12=80.0,
            sigyt23=50.0,
            sigyc23=50.0,
            alpha=0.25,
            efib=180000.0,
            cc=0.04,
            eps0=1.0,
            strflag=1,
        )
        rendered = d.render()
        assert "/MAT/LAW14/14" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 14 in model.mat_law14s
        mat = model.mat_law14s[14]
        assert mat.id == 14
        assert mat.rho0 == pytest.approx(1.55e-9)
        assert mat.ea == pytest.approx(140000.0)
        assert mat.eb == pytest.approx(10000.0)
        assert mat.ec == pytest.approx(10000.0)
        assert mat.prab == pytest.approx(0.30)
        assert mat.prbc == pytest.approx(0.45)
        assert mat.prca == pytest.approx(0.02)
        assert mat.gab == pytest.approx(5000.0)
        assert mat.gbc == pytest.approx(3500.0)
        assert mat.gca == pytest.approx(5000.0)
        assert mat.sigt1 == pytest.approx(1800.0)
        assert mat.sigt2 == pytest.approx(40.0)
        assert mat.sigt3 == pytest.approx(40.0)
        assert mat.delta == pytest.approx(0.08)
        assert mat.cb == pytest.approx(150.0)
        assert mat.cn == pytest.approx(0.5)
        assert mat.fmax == pytest.approx(1200.0)
        assert mat.wplaref == pytest.approx(1.0)
        assert mat.sigyt1 == pytest.approx(2000.0)
        assert mat.sigyt2 == pytest.approx(50.0)
        assert mat.sigyc1 == pytest.approx(1500.0)
        assert mat.sigyc2 == pytest.approx(200.0)
        assert mat.sigt12 == pytest.approx(80.0)
        assert mat.sigc12 == pytest.approx(80.0)
        assert mat.sigt23 == pytest.approx(50.0)
        assert mat.sigc23 == pytest.approx(50.0)
        assert mat.sigyt12 == pytest.approx(80.0)
        assert mat.sigyc12 == pytest.approx(80.0)
        assert mat.sigyt23 == pytest.approx(50.0)
        assert mat.sigyc23 == pytest.approx(50.0)
        assert mat.alpha == pytest.approx(0.25)
        assert mat.efib == pytest.approx(180000.0)
        assert mat.cc == pytest.approx(0.04)
        assert mat.eps0 == pytest.approx(1.0)
        assert mat.strflag == 1

        # Model materials dispatch
        assert 14 in model.materials
        m = model.materials[14]
        assert m.law == 14
        assert m.params["D11"] > 0.0

    def test_deck_writer_roundtrip_synonyms(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_compso and mat_comp_sol synonyms."""
        d = StarterDeck("DECK_SYNONYMS")
        d.title("SYNONYMS_ROUNDTRIP")
        d.mat_compso(
            mid=101,
            title="Mat_Compso",
            rho=1.5e-9,
            ea=100000.0,
            eb=20000.0,
            ec=20000.0,
            prab=0.25,
            prbc=0.20,
            prca=0.15,
            gab=8000.0,
            gbc=6000.0,
            gca=8000.0,
        )
        d.mat_comp_sol(
            mid=102,
            title="Mat_Comp_Sol",
            rho=1.5e-9,
            ea=100000.0,
            eb=20000.0,
            ec=20000.0,
            prab=0.25,
            prbc=0.20,
            prca=0.15,
            gab=8000.0,
            gbc=6000.0,
            gca=8000.0,
        )
        rendered = d.render()
        assert "/MAT/COMPSO/101" in rendered
        assert "/MAT/COMP_SOL/102" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 101 in model.mat_law14s
        assert 102 in model.mat_law14s
        assert 101 in model.materials
        assert 102 in model.materials
        assert model.materials[101].law == 14
        assert model.materials[102].law == 14


# ============================================================================
# 2. Materials Dispatch and Consistency Tests
# ============================================================================

class TestLaw14MaterialsDispatch:
    """Verify pyradioss.materials dispatch functions for LAW14."""

    def test_materials_dispatch(self):
        """pyradioss.materials.solid_update and sound_speed dispatch LAW14."""
        mat = build_law14(
            id=14,
            rho0=1.5e-9,
            E11=100000.0,
            E22=20000.0,
            E33=20000.0,
            nu12=0.25,
            nu23=0.20,
            nu31=0.15,
            G12=8000.0,
            G23=6000.0,
            G31=8000.0,
        )

        sig1 = np.zeros((1, 6), dtype=float)
        sig2 = np.zeros((1, 6), dtype=float)
        deps = np.array([[1.0e-5, 2.0e-5, -1.0e-5, 5.0e-6, -5.0e-6, 3.0e-6]], dtype=float)
        extra1: Dict[str, Any] = {}
        extra2: Dict[str, Any] = {}

        # Top-level dispatch through pyradioss.materials.solid_update
        s_disp, ep_disp, c_disp = materials.solid_update(mat, sig1, deps, dt=1.0e-6, extra=extra1)

        # Direct call to law14_compso.solid_update
        s_dir, ep_dir, c_dir = solid_update(mat, sig2, deps, dt=1.0e-6, extra=extra2)

        np.testing.assert_allclose(s_disp, s_dir)
        np.testing.assert_allclose(ep_disp, ep_dir)
        np.testing.assert_allclose(c_disp, c_dir)

        # sound_speed dispatch
        c_snd_disp = materials.sound_speed(mat)
        c_snd_dir = sound_speed(mat)
        assert c_snd_disp == pytest.approx(c_snd_dir)

    def test_extra_shapes(self):
        """extra_shapes returns required state array shapes for solid elements."""
        mat = build_law14(
            id=14,
            rho0=1.5e-9,
            E11=100000.0,
            E22=20000.0,
            E33=20000.0,
            nu12=0.25,
            nu23=0.20,
            nu31=0.15,
            G12=8000.0,
            G23=6000.0,
            G31=8000.0,
        )

        nip = 8
        shapes_disp = materials.extra_shapes(mat, nip=nip)
        shapes_dir = extra_shapes(mat, nip=nip)

        assert shapes_disp["dam14"] == (nip, 5)
        assert shapes_disp["epe14"] == (nip, 3)
        assert shapes_disp["epc14"] == (nip, 3)
        assert shapes_disp["wpla14"] == (nip,)
        assert shapes_disp["off14"] == (nip,)
        assert shapes_disp["epsf14"] == (nip,)
        assert shapes_disp["sigf14"] == (nip,)
        assert shapes_disp["tsaiwu14"] == (nip,)
        assert shapes_disp == shapes_dir

    def test_consistent_solid_tangent_elastic(self):
        """consistent_solid_tangent returns exact orthotropic D matrix in elastic regime."""
        mat = build_law14(
            id=14,
            rho0=1.5e-9,
            E11=100000.0,
            E22=20000.0,
            E33=20000.0,
            nu12=0.25,
            nu23=0.20,
            nu31=0.15,
            G12=8000.0,
            G23=6000.0,
            G31=8000.0,
        )

        p = mat.params
        d11, d12, d13 = p["D11"], p["D12"], p["D13"]
        d22, d23, d33 = p["D22"], p["D23"], p["D33"]
        g12, g23, g31 = p["G12"], p["G23"], p["G31"]

        expected_d = np.array([
            [d11, d12, d13, 0.0, 0.0, 0.0],
            [d12, d22, d23, 0.0, 0.0, 0.0],
            [d13, d23, d33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, g12, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, g23, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, g31],
        ], dtype=float)

        sig = np.zeros(6, dtype=float)
        d_tangent_disp = materials.consistent_solid_tangent(mat, sig)
        d_tangent_dir = consistent_solid_tangent(mat, sig)

        np.testing.assert_allclose(d_tangent_disp, expected_d, rtol=1e-12)
        np.testing.assert_allclose(d_tangent_dir, expected_d, rtol=1e-12)


# ============================================================================
# 3. Explicit Solver Element Integration (Hexa8 Solid)
# ============================================================================

class TestLaw14Hexa8Integration:
    """Hexa8 solid single-element explicit solver step with LAW14."""

    HEXA8_DECK = """
/BEGIN
HEXA8_LAW14_TEST
-1
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
Cube_LAW14
1 1
/PROP/SOLID/1
Solid_Prop
1.1 0.05 0.1
/MAT/LAW14/1
Composite_Solid_Hexa
1.55e-9, 1.55e-9
140000.0, 10000.0, 10000.0
0.30, 0.45, 0.02
5000.0, 3500.0, 5000.0
1800.0, 40.0, 40.0, 0.08
150.0, 0.5, 1200.0, 1.0
2000.0, 50.0, 1500.0, 200.0
80.0, 80.0, 50.0, 50.0
0.25, 180000.0, 0.04, 1.0, 1
/END
"""

    def test_hexa8_solid_elastic_step(self, tmp_path: Path):
        """Verify elastic 3D stress update and force equilibrium for Hexa8 solid with LAW14."""
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
        assert sig[0, 0] < 1800.0  # Elastic (< sigt1 = 1800 MPa)

        # Force equilibrium on free body: sum of all internal nodal forces is zero
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)
        # Tensile reaction: nodes with x=1 pulled in -x, nodes with x=0 pulled in +x
        assert fint[[1, 2, 5, 6], 0].sum() < 0.0
        assert fint[[0, 3, 4, 7], 0].sum() > 0.0

        # State checks
        assert g.state["epsp"][0] == pytest.approx(0.0)
        assert g.state["mat_extra"]["off14"][0] == pytest.approx(1.0)
        assert g.state["mat_extra"]["dam14"][0, 0] == pytest.approx(0.0)

    HEXA8_DAMAGE_DECK = """
/BEGIN
HEXA8_LAW14_DAMAGE_TEST
-1
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
Cube_LAW14_Dam
1 1
/PROP/SOLID/1
Solid_Prop
1.1 0.05 0.1
/MAT/LAW14/1
Composite_Cracking
1.55e-9, 1.55e-9
140000.0, 10000.0, 10000.0
0.30, 0.45, 0.02
5000.0, 3500.0, 5000.0
1800.0, 40.0, 40.0, 0.08
150.0, 0.5, 1200.0, 1.0
2000.0, 500.0, 1500.0, 500.0
80.0, 80.0, 50.0, 50.0
0.0, 0.0, 0.0, 0.0, 1
/END
"""

    def test_hexa8_solid_damage_step(self, tmp_path: Path):
        """Verify tensile cracking damage initiation, softening, and force equilibrium for Hexa8."""
        model, log = _build_model(self.HEXA8_DAMAGE_DECK, tmp_path)
        g = model.bricks
        assert g.n == 1

        # Impose large transverse strain rate along y: vy = y * rate
        # Transverse tensile limit sigt2 = 40.0 MPa; E22 ~ 10000 MPa, strain ~ 0.004 reaches 40 MPa.
        # Imposing strain delta = 0.01 exceeds sigt2.
        dt = 1.0e-4
        rate = 100.0  # rate * dt = 0.01 strain
        v = np.zeros_like(model.x)
        v[:, 1] = model.x[:, 1] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        # Step 1: Initiates damage
        dtc1 = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
        assert dtc1[0] > 0.0

        # Step 1 damage & stress: clipped to sigt2 = 40.0, dam14 updated to delta = 0.08
        dam14 = g.state["mat_extra"]["dam14"]
        assert dam14[0, 1] == pytest.approx(0.08)
        assert g.state["sig"][0, 1] == pytest.approx(40.0, rel=1e-3)
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)

        # Step 2: Softened stress clipped to (1 - delta) * sigt2 = 0.92 * 40.0 = 36.8 MPa
        fint.fill(0.0)
        mint.fill(0.0)
        dtc2 = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
        assert dtc2[0] > 0.0

        assert dam14[0, 1] == pytest.approx(0.16)
        assert g.state["sig"][0, 1] == pytest.approx(36.8, rel=1e-3)
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)

    def test_hexa8_solid_plastic_step(self, tmp_path: Path):
        """Verify Tsai-Wu plastic deformation and internal energy dissipation for Hexa8 solid."""
        model, log = _build_model(self.HEXA8_DECK, tmp_path)
        g = model.bricks
        assert g.n == 1

        # Apply pure shear deformation in xy: vx = y * rate, vy = 0
        # Shear limit sigyt12 = 80 MPa; G12 = 5000 MPa; yield strain gamma_12 ~ 80/5000 = 0.016
        dt = 1.0e-4
        rate = 500.0  # Engineering shear rate -> large plastic flow
        v = np.zeros_like(model.x)
        v[:, 0] = model.x[:, 1] * rate
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        # Run two steps so trial stress exists and Tsai-Wu plastic return activates
        dtc1 = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
        fint.fill(0.0)
        mint.fill(0.0)
        dtc2 = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)

        assert dtc2[0] > 0.0
        # Plastic work accumulated
        wpla14 = g.state["mat_extra"]["wpla14"]
        assert wpla14[0] > 0.0

        # Equilibrium holds in plasticity
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)
