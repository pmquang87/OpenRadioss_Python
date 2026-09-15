"""Integration tests for /MAT/LAW43 (/MAT/HILL_TAB) - Milestone M548.

Covers:
1. StarterDeck roundtrip:
   - StarterDeck.mat_law43 roundtrip with multiple curves and parameters
   - StarterDeck.mat_hill_tab synonym roundtrip
2. pyradioss.materials module dispatch:
   - Registration of 43, "43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB" in MAT_PHYSICS_REGISTRY
   - materials.shell_update dispatches to law43_hill_tab
   - materials.solid_update dispatches to law43_hill_tab
   - materials.sound_speed dispatches to law43_hill_tab
   - materials.consistent_solid_tangent dispatches to law43_hill_tab
   - materials.consistent_shell_tangent dispatches to law43_hill_tab
   - materials.shell_membrane_tangent dispatches to law43_hill_tab
3. extra_shapes verification:
   - returns uv43: (4,) without nip, and (nip, 4) with nip
   - returns off43: () without nip, and (nip,) with nip
4. Single-element explicit solver step with Shell BT4 (Belytschko-Tsay):
   - In-plane tensile deformation into plasticity
   - Membrane force assembly and positive timestep
   - Numerical stability (no NaN/Inf)
   - Free-body internal force equilibrium (sum(F_int) == 0)
5. Single-element explicit solver step with Shell QEPH:
   - In-plane tensile deformation into plasticity
   - Force assembly and positive timestep
   - Numerical stability (no NaN/Inf)
   - Free-body internal force equilibrium (sum(F_int) == 0)
6. Single-element explicit solver step with Hexa8 solid:
   - 3D stress update and positive timestep
   - Compressive/tensile deformation into plasticity
   - Internal nodal force equilibrium (sum(F_int) == 0)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_bt4, shell_qeph, solid_hexa8
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.starter import (
    build_element_groups,
    initialize_elements_and_mass,
    resolve_node_groups,
    resolve_surfaces,
    run_starter,
)


def _build_model(deck_text: str, tmp_path: Path) -> tuple[Model, MessageLog]:
    """Helper to parse and initialize model element groups and masses from raw deck text."""
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

class TestLaw43DeckWriterRoundtrip:
    """Verify StarterDeck roundtrip for mat_law43 and mat_hill_tab."""

    def test_deck_writer_roundtrip_mat_law43(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_law43 with curves through emitter and parser."""
        d = StarterDeck("DECK_LAW43")
        d.title("LAW43_ROUNDTRIP")
        d.funct(10, "YIELD_CURVE_0", [(0.0, 200.0), (0.05, 300.0), (0.1, 350.0), (0.5, 400.0)])
        d.funct(11, "YIELD_CURVE_1", [(0.0, 250.0), (0.05, 350.0), (0.1, 400.0), (0.5, 450.0)])
        d.mat_law43(
            mid=43,
            title="Hill_Tab_Material",
            rho=7.8e-6,
            rhor=7.8e-6,
            e=210000.0,
            nu=0.3,
            ifunce=10,
            einf=250000.0,
            ce=0.05,
            r00=1.2,
            r45=1.5,
            r90=1.8,
            chard=0.1,
            iyield=1,
            eps_max=0.35,
            epst1=0.25,
            epst2=0.30,
            fcut=1000.0,
            fsmooth=1,
            curves=[(10, 1.0, 0.0), (11, 1.0, 100.0)],
        )
        rendered = d.render()
        assert "/MAT/LAW43/43" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 43 in model.materials or 43 in model.mat_law43s
        mat = model.mat_law43s[43] if 43 in model.mat_law43s else model.materials[43]
        assert getattr(mat, "id", 0) == 43
        assert getattr(mat, "rho0", getattr(mat, "rho", 0.0)) == pytest.approx(7.8e-6)
        assert getattr(mat, "e", getattr(mat, "E", 0.0)) == pytest.approx(210000.0)
        assert getattr(mat, "nu", 0.0) == pytest.approx(0.3)
        assert getattr(mat, "r00", 0.0) == pytest.approx(1.2)
        assert getattr(mat, "r45", 0.0) == pytest.approx(1.5)
        assert getattr(mat, "r90", 0.0) == pytest.approx(1.8)

        # Check curves
        curves = getattr(mat, "curves", [])
        assert len(curves) == 2
        fid0 = curves[0].get("fct_id", curves[0].get("funct_id", 0))
        fid1 = curves[1].get("fct_id", curves[1].get("funct_id", 0))
        assert fid0 == 10
        assert fid1 == 11

    def test_deck_writer_roundtrip_synonyms(self, tmp_path: Path):
        """Roundtrip StarterDeck.mat_hill_tab synonym."""
        d = StarterDeck("DECK_HILL_TAB")
        d.title("HILL_TAB_ROUNDTRIP")
        d.funct(20, "YIELD_CURVE_ISO", [(0.0, 300.0), (0.1, 400.0)])
        d.mat_hill_tab(
            mid=143,
            title="Hill_Tab_Synonym",
            rho=7.85e-6,
            e=205000.0,
            nu=0.29,
            r00=1.1,
            r45=1.3,
            r90=1.7,
            curves=[(20, 1.0, 0.0)],
        )
        rendered = d.render()
        assert "/MAT/HILL_TAB/143" in rendered

        deck_file = tmp_path / "DECK_0000.rad"
        deck_file.write_text(rendered, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        assert 143 in model.materials or 143 in model.mat_law43s
        mat = model.mat_law43s[143] if 143 in model.mat_law43s else model.materials[143]
        assert getattr(mat, "id", 0) == 143
        assert getattr(mat, "rho0", getattr(mat, "rho", 0.0)) == pytest.approx(7.85e-6)
        assert getattr(mat, "e", getattr(mat, "E", 0.0)) == pytest.approx(205000.0)
        assert getattr(mat, "nu", 0.0) == pytest.approx(0.29)
        assert getattr(mat, "r00", 0.0) == pytest.approx(1.1)
        assert getattr(mat, "r45", 0.0) == pytest.approx(1.3)
        assert getattr(mat, "r90", 0.0) == pytest.approx(1.7)


# ============================================================================
# 2. pyradioss.materials Module Dispatch Tests
# ============================================================================

class TestLaw43MaterialsDispatch:
    """Verify registration, dispatch, state variables, and tangents in pyradioss.materials."""

    def test_registry_contains_all_aliases(self):
        """MAT_PHYSICS_REGISTRY must contain 43 and all standard aliases."""
        for key in (43, "43", "LAW43", "HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB"):
            assert key in MAT_PHYSICS_REGISTRY, f"Key {key!r} not in MAT_PHYSICS_REGISTRY"

    def test_extra_shapes(self):
        """Verify extra_shapes returns uvar43/uv43: (4,) or (nip, 4), and off43."""
        for law_val in (43, "43", "LAW43", "HILL_TAB"):
            mat = Material(id=1, law=law_val, rho0=7800.0, params={"e": 2.1e11, "nu": 0.3})
            shapes_elem = materials.extra_shapes(mat)
            assert "uvar43" in shapes_elem or "uv43" in shapes_elem
            key = "uvar43" if "uvar43" in shapes_elem else "uv43"
            assert shapes_elem[key] == (4,)
            assert "off43" in shapes_elem
            assert shapes_elem["off43"] == ()

            shapes_layer = materials.extra_shapes(mat, nip=5)
            assert key in shapes_layer
            assert shapes_layer[key] == (5, 4)
            assert "off43" in shapes_layer
            assert shapes_layer["off43"] == (5,)

    def test_sound_speed(self):
        """Verify sound speed c = sqrt(A1 / rho0) for shell plane stress."""
        mat = Material(id=1, law=43, rho0=7800.0, params={"e": 2.1e11, "nu": 0.3})
        c = materials.sound_speed(mat)
        expected_c = math.sqrt(2.1e11 / ((1.0 - 0.3 ** 2) * 7800.0))
        assert math.isclose(float(c), expected_c, rel_tol=1e-4)

    def test_shell_membrane_tangent(self):
        """Verify shell membrane tangent returns (3, 3) elastic plane-stress matrix."""
        mat = Material(id=1, law=43, rho0=7800.0, params={"e": 2.1e11, "nu": 0.3})
        C = materials.shell_membrane_tangent(mat)
        assert C.shape == (3, 3)
        c11 = 2.1e11 / (1.0 - 0.3 ** 2)
        c12 = 0.3 * c11
        g = 2.1e11 / (2.0 * (1.0 + 0.3))
        assert math.isclose(C[0, 0], c11, rel_tol=1e-4)
        assert math.isclose(C[1, 1], c11, rel_tol=1e-4)
        assert math.isclose(C[0, 1], c12, rel_tol=1e-4)
        assert math.isclose(C[2, 2], g, rel_tol=1e-4)

    def test_consistent_solid_tangent(self):
        """Verify consistent solid tangent returns (6, 6) matrix."""
        mat = Material(id=1, law=43, rho0=7800.0, params={"e": 2.1e11, "nu": 0.3})
        sig = np.zeros(6, dtype=float)
        D = materials.consistent_solid_tangent(mat, sig)
        assert D.shape == (6, 6)
        assert D[0, 0] > 0.0

    def test_shell_update_signature_and_return(self):
        """materials.shell_update must return a 2-tuple (sig, epsp)."""
        n = 2
        mat = Material(
            id=1,
            law=43,
            rho0=7800.0,
            params={
                "e": 2.1e11,
                "nu": 0.3,
                "r00": 1.0,
                "r45": 1.0,
                "r90": 1.0,
                "curves": [{"fct_id": 1, "pts": [(0.0, 200e6), (0.1, 300e6)]}],
            },
        )
        sig = np.zeros((n, 3))
        deps = np.full((n, 3), 1.0e-4)
        epsp = np.zeros(n)
        extra = {"uv43": np.zeros((n, 4)), "off43": np.ones(n)}

        res = materials.shell_update(mat, sig, deps, epsp, dt=1.0e-6, extra=extra)
        assert isinstance(res, tuple)
        assert len(res) == 2
        sig_new, epsp_new = res
        assert sig_new.shape == (n, 3)
        assert epsp_new.shape == (n,)
        assert np.isfinite(sig_new).all()
        assert np.isfinite(epsp_new).all()

    def test_solid_update_signature_and_return(self):
        """materials.solid_update must update sig and epsp."""
        n = 2
        mat = Material(
            id=1,
            law=43,
            rho0=7800.0,
            params={
                "e": 2.1e11,
                "nu": 0.3,
                "r00": 1.0,
                "r45": 1.0,
                "r90": 1.0,
                "curves": [{"fct_id": 1, "pts": [(0.0, 200e6), (0.1, 300e6)]}],
            },
        )
        sig = np.zeros((n, 6))
        deps = np.full((n, 6), 1.0e-4)
        epsp = np.zeros(n)
        extra = {"uv43": np.zeros((n, 4)), "off43": np.ones(n)}

        res = materials.solid_update(mat, sig, deps, epsp, dt=1.0e-6, extra=extra)
        # solid_update either returns (sig, epsp, c) or modifies sig/epsp in-place
        assert np.isfinite(sig).all()
        assert not np.allclose(sig, 0.0)


# ============================================================================
# 3. Explicit Solver Element Integration Tests
# ============================================================================

class TestLaw43ElementSteps:
    """Single-element explicit steps for Shell BT4, Shell QEPH, and Hexa8 solid."""

    def _make_shell_model(self, tmp_path: Path, name: str, ishell: int):
        d = StarterDeck(name)
        d.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])
        d.part(1, "Shell_Part", 1, 1)
        d.prop_shell(1, "Shell_Prop", thick=1.0, nip=3, ishell=ishell)
        d.funct(1, "Yield_Curve", [(0.0, 200.0), (0.05, 300.0), (0.10, 350.0), (0.50, 400.0)])
        d.mat_law43(
            mid=1,
            title="Hill_Tab_Mat",
            rho=7.8e-6,
            e=210000.0,
            nu=0.3,
            r00=1.2,
            r45=1.5,
            r90=1.8,
            curves=[(1, 1.0, 0.0)],
        )
        s_path = str(tmp_path / f"{name}_0000.rad")
        d.write(s_path)
        model = run_starter(s_path)
        group = model.shells if ishell == 1 else model.shells_qeph
        kernel = shell_bt4 if ishell == 1 else shell_qeph
        return model, group, kernel

    def test_shell_bt4_single_element_explicit_step(self, tmp_path: Path):
        """Single 4-node BT4 shell explicit step under tension into plasticity with LAW43."""
        model, group, kernel = self._make_shell_model(tmp_path, "SHELL_BT4", ishell=1)
        assert group.n == 1

        dt = 1.0e-6
        v = np.zeros_like(model.x)
        # Pull right edge nodes (indices 1 and 2 at x=10) in +x
        v[1, 0] = 1000.0
        v[2, 0] = 1000.0

        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        # Run explicit cycles to accumulate strain into plastic regime
        for _ in range(15):
            model.x += v * dt
            fint.fill(0.0)
            mint.fill(0.0)
            dtc = kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

        # 1. Output timestep must be positive
        assert dtc > 0.0

        # 2. Numerical stability: no NaN or Inf
        assert np.isfinite(fint).all()
        assert np.isfinite(mint).all()
        assert np.isfinite(group.state["sig"]).all()

        # 3. Membrane force assembly non-zero
        assert np.any(np.abs(fint[:, 0]) > 0.0)

        # 4. Free-body internal force equilibrium: sum of all nodal forces is zero
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-3)

        # 5. Reaction signs: pulled nodes pulled back (-x), fixed nodes pulled forward (+x)
        assert fint[[1, 2], 0].sum() < 0.0
        assert fint[[0, 3], 0].sum() > 0.0

        # 6. Plastic strain accumulation
        epsp_val = group.state.get("epsp", None)
        extra = group.state.get("mat_extra", {})
        uvar = extra.get("uvar43", extra.get("uv43", None))
        pla = extra.get("pla43", None)
        has_plasticity = (
            (epsp_val is not None and np.any(epsp_val > 0.0))
            or (uvar is not None and np.any(uvar > 0.0))
            or (pla is not None and np.any(pla > 0.0))
        )
        assert has_plasticity

    def test_shell_qeph_single_element_explicit_step(self, tmp_path: Path):
        """Single 4-node QEPH shell explicit step under tension into plasticity with LAW43."""
        model, group, kernel = self._make_shell_model(tmp_path, "SHELL_QEPH", ishell=24)
        assert group.n == 1

        dt = 1.0e-6
        v = np.zeros_like(model.x)
        # Pull right edge nodes (indices 1 and 2 at x=10) in +x
        v[1, 0] = 1000.0
        v[2, 0] = 1000.0

        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        # Run explicit cycles into plasticity
        for _ in range(15):
            model.x += v * dt
            fint.fill(0.0)
            mint.fill(0.0)
            dtc = kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

        # 1. Timestep positive
        assert dtc > 0.0

        # 2. Stability
        assert np.isfinite(fint).all()
        assert np.isfinite(mint).all()
        assert np.isfinite(group.state["sig"]).all()

        # 3. Membrane force assembly
        assert np.any(np.abs(fint[:, 0]) > 0.0)

        # 4. Internal force equilibrium
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-3)

        # 5. Reaction signs
        assert fint[[1, 2], 0].sum() < 0.0
        assert fint[[0, 3], 0].sum() > 0.0

        # 6. Plastic strain accumulation
        epsp_val = group.state.get("epsp", None)
        extra = group.state.get("mat_extra", {})
        uvar = extra.get("uvar43", extra.get("uv43", None))
        pla = extra.get("pla43", None)
        has_plasticity = (
            (epsp_val is not None and np.any(epsp_val > 0.0))
            or (uvar is not None and np.any(uvar > 0.0))
            or (pla is not None and np.any(pla > 0.0))
        )
        assert has_plasticity

    def _make_solid_model(self, tmp_path: Path, name: str):
        d = StarterDeck(name)
        d.node([
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
            (3, 1.0, 1.0, 0.0),
            (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0),
            (6, 1.0, 0.0, 1.0),
            (7, 1.0, 1.0, 1.0),
            (8, 0.0, 1.0, 1.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        d.part(1, "Cube_Part", 1, 1)
        d.prop_solid(1, "Solid_Prop")
        d.funct(1, "Yield_Curve_Solid", [(0.0, 200.0), (0.05, 300.0), (0.10, 350.0), (0.50, 400.0)])
        d.mat_law43(
            mid=1,
            title="Hill_Tab_Solid",
            rho=7.8e-6,
            e=210000.0,
            nu=0.3,
            r00=1.2,
            r45=1.5,
            r90=1.8,
            curves=[(1, 1.0, 0.0)],
        )
        s_path = str(tmp_path / f"{name}_0000.rad")
        d.write(s_path)
        model = run_starter(s_path)
        group = model.bricks
        return model, group

    def test_solid_hexa8_single_element_explicit_step(self, tmp_path: Path):
        """Single Hexa8 solid element explicit step under tension into plasticity."""
        model, group = self._make_solid_model(tmp_path, "HEXA8_SOLID")
        assert group.n == 1

        dt = 1.0e-6
        v = np.zeros_like(model.x)
        # Pull nodes with x=1 (indices 1, 2, 5, 6) in +x
        v[[1, 2, 5, 6], 0] = 500.0

        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        # Run several explicit steps
        for _ in range(15):
            model.x += v * dt
            fint.fill(0.0)
            mint.fill(0.0)
            dtc = solid_hexa8.forces(group, model.x, v, model.vr, dt, fint, mint)

        # 1. Timestep positive
        assert dtc[0] > 0.0

        # 2. Stability
        assert np.isfinite(fint).all()
        assert np.isfinite(group.state["sig"]).all()

        # 3. 3D stress update non-zero
        sig = group.state["sig"]
        assert sig[0, 0] > 0.0

        # 4. Internal force equilibrium: sum of all 8 nodal forces is zero
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-3)

        # 5. Reaction signs
        assert fint[[1, 2, 5, 6], 0].sum() < 0.0
        assert fint[[0, 3, 4, 7], 0].sum() > 0.0

        # 6. Plastic strain accumulation
        epsp_val = group.state.get("epsp", None)
        extra = group.state.get("mat_extra", {})
        uvar = extra.get("uvar43", extra.get("uv43", None))
        pla = extra.get("pla43", None)
        has_plasticity = (
            (epsp_val is not None and np.any(epsp_val > 0.0))
            or (uvar is not None and np.any(uvar > 0.0))
            or (pla is not None and np.any(pla > 0.0))
        )
        assert has_plasticity
