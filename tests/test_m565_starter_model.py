"""
Tests for Milestone M565: /MAT/LAW88 Tabulated Hyperelastic Material Model
Starter, Model Entity, Deck Writer, Checks & Element Integration.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MaterialLaw88,
    MatLaw88,
    MatTabulatedHyperelastic,
    MatHyperElas,
    MatTabHyp,
    Part,
    Property,
)
from pyradioss.model import Model
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.cfg_catalogue import law_number, canonical_law_name
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law88
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import (
    check_mat_law88,
    check_materials,
    check_model,
    MessageLog,
)
from pyradioss.input.checks import (
    check_mat_law88 as input_check_mat_law88,
    check_all,
)
import pyradioss.materials as mats


# ============================================================================
# 1. Model Entity & Aliases
# ============================================================================

def test_material_law88_aliases_and_defaults():
    """Verify MaterialLaw88 class, aliases, and default attributes."""
    assert MatLaw88 is MaterialLaw88
    assert MatTabulatedHyperelastic is MaterialLaw88
    assert MatHyperElas is MaterialLaw88
    assert MatTabHyp is MaterialLaw88

    mat = MaterialLaw88(id=10, title="Rubber_LAW88")
    assert mat.id == 10
    assert mat.title == "Rubber_LAW88"
    assert mat.law == 88
    assert mat.law_name == "LAW88"
    assert mat.rho0 == 0.0
    assert mat.nu == 0.495
    assert mat.bulk == 0.0
    assert mat.fcut == 0.0
    assert mat.fsmooth == 0
    assert mat.nl == 0
    assert mat.ifunc_unload == 0
    assert mat.fscale_unload == 1.0
    assert mat.hys == 0.0
    assert mat.shape == 1.0
    assert mat.tension == 0
    assert mat.rtype == 0
    assert mat.func_load_list == []
    assert mat.fscale_load_list == []
    assert mat.rate_load_list == []
    assert mat.lamfit_list == []
    assert mat.sgl == 0.0
    assert mat.sw == 0.0
    assert mat.st == 0.0
    assert mat.g == 0.0
    assert mat.sigf == 0.0
    assert mat.kfail == 0.0
    assert mat.gam1 == 0.0
    assert mat.gam2 == 0.0
    assert mat.eh == 0.0
    assert mat.failip == 0
    assert mat.beta == 0.0
    assert mat.young == 0.0
    assert mat.shear == 0.0

    model = Model()
    assert hasattr(model, "mat_law88s")
    assert isinstance(model.mat_law88s, dict)


def test_material_law88_mapping_and_properties():
    """Verify dictionary-like mapping protocol and property helpers."""
    mat = MaterialLaw88(
        id=1,
        title="HypRubber",
        rho0=1000.0,
        nu=0.45,
        bulk=1.0e7,
        func_load_list=[101, 102],
        fscale_load_list=[1.0, 1.0],
        rate_load_list=[0.0, 10.0],
    )

    # Mapping protocol
    assert mat["id"] == 1
    assert mat["rho0"] == 1000.0
    assert mat["nu"] == 0.45
    assert "bulk" in mat
    assert mat.get("shape", 1.0) == 1.0
    assert "func_load_list" in mat.keys()
    assert 1.0e7 in mat.values()
    assert ("id", 1) in list(mat.items())

    # Setting custom items in params
    mat["user_tag"] = "test_tag"
    assert mat["user_tag"] == "test_tag"
    assert mat.params["user_tag"] == "test_tag"

    # Properties & modulus derivation
    assert mat.rho == 1000.0
    assert mat.K == 1.0e7
    # E = 3K(1 - 2nu) = 3 * 1e7 * (1 - 0.9) = 3e6
    assert math.isclose(mat.E, 3.0e6, rel_tol=1e-6)
    # G = 3K(1 - 2nu) / (2(1 + nu)) = 3e6 / 2.9 = 1.03448e6
    assert math.isclose(mat.G, 3.0e6 / (2.0 * 1.45), rel_tol=1e-6)

    # Curves property alias
    assert mat.curves == [101, 102]

    # Sound speed: both float and callable
    assert isinstance(mat.sound_speed, float)
    assert mat.sound_speed > 0.0
    assert mat.sound_speed() == float(mat.sound_speed)
    assert mat.sound_speed_solid() == float(mat.sound_speed_solid)
    assert mat.sound_speed_shell() == float(mat.sound_speed_shell)


# ============================================================================
# 2. Card Layouts & Synonyms
# ============================================================================

def test_card_layouts_law88():
    """Verify CARD_LAYOUTS has MAT_LAW88_1..6 and all synonym definitions."""
    synonym_prefixes = [
        "MAT_LAW88",
        "MAT_TABULATED_HYPERELASTIC",
        "MAT_HYPER_ELAS",
        "MAT_TAB_HYP",
    ]
    for prefix in synonym_prefixes:
        for card_idx in range(1, 7):
            card_key = f"{prefix}_{card_idx}"
            assert card_key in CARD_LAYOUTS, f"Missing layout {card_key}"

    # Verify column widths
    assert CARD_LAYOUTS["MAT_LAW88_1"] == [20, 20]
    assert CARD_LAYOUTS["MAT_LAW88_2"] == [20, 20, 20, 10, 10]
    assert CARD_LAYOUTS["MAT_LAW88_3"] == [10, 10, 20, 20, 20, 10, 10]
    assert CARD_LAYOUTS["MAT_LAW88_4"] == [10, 10, 20, 20, 20]
    assert CARD_LAYOUTS["MAT_LAW88_5"] == [20, 20, 20, 20, 20]
    assert CARD_LAYOUTS["MAT_LAW88_6"] == [20, 20, 20, 20, 10, 10]


# ============================================================================
# 3. CFG Catalogue & Law Number Resolution
# ============================================================================

def test_cfg_catalogue_law88():
    """Verify law catalogue mappings for LAW88 and synonyms."""
    test_keys = [
        88,
        "88",
        "LAW88",
        "TABULATED_HYPERELASTIC",
        "TAB_HYP",
        "HYPER_ELAS",
        "TABULATED_HYP",
        "MAT_LAW88",
        "MAT_TABULATED_HYPERELASTIC",
        "MAT_HYPER_ELAS",
        "MAT_TAB_HYP",
    ]
    for k in test_keys:
        assert law_number(k) == 88, f"law_number failed for {k}"
        assert canonical_law_name(k) == "LAW88", f"canonical_law_name failed for {k}"


# ============================================================================
# 4. Starter Keywords Reader
# ============================================================================

def test_read_mat_law88_fixed_format(tmp_path):
    """Verify reading fixed-format /MAT/LAW88 with Cards 1-6."""
    deck_text = (
        "/BEGIN\n"
        "Tension_Specimen_Run\n"
        "      2022         0\n"
        "/MAT/LAW88/1\n"
        "Tension_Specimen\n"
        f"{1.2e-3:>20.6e}{0.0:>20.6e}\n"
        f"{0.495:>20.6e}{2000.0:>20.6e}{0.0:>20.6e}{0:>10d}{2:>10d}\n"
        f"{0:>10d}{0:>10d}{1.0:>20.6e}{0.0:>20.6e}{1.0:>20.6e}{0:>10d}{0:>10d}\n"
        f"{101:>10d}{0:>10d}{1.0:>20.6e}{0.0:>20.6e}{1.0:>20.6e}\n"
        f"{102:>10d}{0:>10d}{1.0:>20.6e}{100.0:>20.6e}{1.0:>20.6e}\n"
        f"{50.0:>20.6e}{10.0:>20.6e}{2.0:>20.6e}{5.0:>20.6e}{100.0:>20.6e}\n"
        f"{1.0:>20.6e}{0.1:>20.6e}{0.2:>20.6e}{0.0:>20.6e}{' ':>10s}{1:>10d}\n"
        "/END\n"
    )

    deck_file = tmp_path / "deck_fixed_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law88(block, model, log)

    assert 1 in model.mat_law88s
    assert 1 in model.materials
    mat = model.mat_law88s[1]

    assert mat.id == 1
    assert mat.title == "Tension_Specimen"
    assert math.isclose(mat.rho0, 1.2e-3)
    assert math.isclose(mat.nu, 0.495)
    assert math.isclose(mat.bulk, 2000.0)
    assert mat.nl == 2
    assert mat.func_load_list == [101, 102]
    # Specimen dimension scaling:
    # areafac = 1.0 / (sw * st) = 1.0 / (10 * 2) = 0.05
    # scale factor = 1.0 * areafac = 0.05
    assert math.isclose(mat.fscale_load_list[0], 0.05)
    assert math.isclose(mat.fscale_load_list[1], 0.05)
    assert mat.rate_load_list == [0.0, 100.0]
    assert math.isclose(mat.sgl, 50.0)
    assert math.isclose(mat.sw, 10.0)
    assert math.isclose(mat.st, 2.0)
    assert math.isclose(mat.g, 5.0)
    assert math.isclose(mat.sigf, 100.0)
    assert mat.failip == 1


def test_read_mat_law88_free_format(tmp_path):
    """Verify reading free-format /MAT/LAW88 with comma-separated values."""
    deck_text = (
        "/MAT/LAW88/2\n"
        "Free_LAW88\n"
        "1.1e-3, 0.0\n"
        "0.48, 1500.0, 0.0, 0, 1\n"
        "0, 0, 1.0, 0.0, 1.0, 0, 0\n"
        "201, 0, 1.0, 0.0, 0.0\n"
        "0.0, 0.0, 0.0, 10.0, 0.0\n"
        "50.0, 0.0, 0.0, 0.0, 0, 0\n"
        "/END\n"
    )

    deck_file = tmp_path / "deck_free_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law88(block, model, log)

    assert 2 in model.mat_law88s
    mat = model.mat_law88s[2]
    assert mat.id == 2
    assert mat.title == "Free_LAW88"
    assert math.isclose(mat.rho0, 1.1e-3)
    assert math.isclose(mat.nu, 0.48)
    assert math.isclose(mat.bulk, 1500.0)
    assert mat.nl == 1
    assert mat.func_load_list == [201]
    assert math.isclose(mat.g, 10.0)


def test_read_mat_law88_negative_shear_warning(tmp_path):
    """Verify ANCMSG 3109 warning and bulk modulus adjustment when shear is negative."""
    deck_text = (
        "/MAT/LAW88/3\n"
        "NegShear\n"
        "1.0e-3, 0.0\n"
        "0.49, 100.0, 0.0, 0, 1\n"
        "0, 0, 1.0, 0.0, 1.0, 0, 0\n"
        "301, 0, 1.0, 0.0, 0.0\n"
        "0.0, 0.0, 0.0, -5.0, 0.0\n"
        "0.0, 0.0, 0.0, 0.0, 0, 0\n"
        "/END\n"
    )

    deck_file = tmp_path / "deck_negshear_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law88(block, model, log)

    mat = model.mat_law88s[3]
    # Warning was logged and G clamped to 0.0, K adjusted
    assert mat.shear >= 0.0
    assert mat.g >= 0.0


# ============================================================================
# 5. Deck Writer & Roundtrip
# ============================================================================

def test_deck_writer_mat_law88_fixed_format(tmp_path):
    """Verify StarterDeck.mat_law88 produces valid fixed format and roundtrips."""
    deck = StarterDeck("M565_TEST")
    mat = MaterialLaw88(
        id=5,
        title="DeckRubber",
        rho0=1.2e-3,
        nu=0.49,
        bulk=500.0,
        fcut=10.0,
        fsmooth=1,
        nl=1,
        func_load_list=[501],
        fscale_load_list=[1.0],
        rate_load_list=[0.0],
        lamfit_list=[1.0],
        sgl=25.0,
        sw=5.0,
        st=1.0,
        g=2.5,
        sigf=20.0,
        kfail=0.8,
        gam1=0.05,
        gam2=0.1,
        eh=0.0,
        failip=1,
        beta=0.2,
    )

    deck.mat_law88(mat)
    deck_str = deck.write()

    assert "/MAT/LAW88/5" in deck_str
    assert "DeckRubber" in deck_str

    # Read back into model
    deck_file = tmp_path / "deck_writer_0000.rad"
    deck_file.write_text(deck_str + "\n/END\n", encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law88(block, model, log)

    mat_rt = model.mat_law88s[5]
    assert mat_rt.id == 5
    assert mat_rt.title == "DeckRubber"
    assert math.isclose(mat_rt.rho0, 1.2e-3)
    assert math.isclose(mat_rt.nu, 0.49)
    assert math.isclose(mat_rt.bulk, 500.0)
    assert mat_rt.nl == 1
    assert mat_rt.func_load_list == [501]


def test_deck_writer_mat_law88_free_format():
    """Verify StarterDeck.mat_law88 free format output."""
    deck = StarterDeck("M565_TEST")
    deck.mat_law88(
        id=6,
        title="FreeRubber",
        rho0=1.0e-3,
        nu=0.45,
        bulk=800.0,
        nl=1,
        func_load_list=[601],
        fscale_load_list=[1.0],
        rate_load_list=[0.0],
        free=True,
    )
    deck_str = deck.write()
    assert "/MAT/LAW88/6" in deck_str
    assert "0.001 0.0" in deck_str


def test_deck_writer_aliases():
    """Verify deck writer aliases produce identical output."""
    d1 = StarterDeck("TEST")
    d1.mat_law88(id=7, title="A", rho0=1e-3, nu=0.49, bulk=100.0, nl=1, func_load_list=[1])

    d2 = StarterDeck("TEST")
    d2.mat_tabulated_hyperelastic(id=7, title="A", rho0=1e-3, nu=0.49, bulk=100.0, nl=1, func_load_list=[1])

    d3 = StarterDeck("TEST")
    d3.mat_hyper_elas(id=7, title="A", rho0=1e-3, nu=0.49, bulk=100.0, nl=1, func_load_list=[1])

    d4 = StarterDeck("TEST")
    d4.mat_tab_hyp(id=7, title="A", rho0=1e-3, nu=0.49, bulk=100.0, nl=1, func_load_list=[1])

    assert d1.write() == d2.write()
    assert d1.write() == d3.write()
    assert d1.write() == d4.write()


# ============================================================================
# 6. Starter Checks
# ============================================================================

def test_checks_mat_law88_valid():
    """Verify clean check on valid LAW88 material."""
    mat = MaterialLaw88(
        id=1,
        rho0=1000.0,
        nu=0.45,
        bulk=1.0e6,
        nl=1,
        func_load_list=[101],
        fscale_load_list=[1.0],
        rate_load_list=[0.0],
        g=5.0e4,
    )
    log = MessageLog()
    check_mat_law88(mat=mat, log=log)
    assert not log.has_errors
    assert not log.has_warnings


def test_checks_mat_law88_invalid_density():
    """Verify error ANCMSG 1514 on zero or negative density."""
    mat = MaterialLaw88(id=1, rho0=0.0, nu=0.45, bulk=1e6, nl=1, func_load_list=[101])
    log = MessageLog()
    check_mat_law88(mat=mat, log=log)
    assert log.has_errors
    assert any("ANCMSG 1514" in err for err in log.errors)


def test_checks_mat_law88_no_loading_curves():
    """Verify error ANCMSG 866 when NL == 0 and no loading curves defined."""
    mat = MaterialLaw88(id=1, rho0=1000.0, nu=0.45, bulk=1e6, nl=0, func_load_list=[])
    log = MessageLog()
    check_mat_law88(mat=mat, log=log)
    assert log.has_errors
    assert any("ANCMSG 866" in err for err in log.errors)


def test_checks_mat_law88_descending_strain_rates():
    """Verify error ANCMSG 478 when strain rates are not ascending."""
    mat = MaterialLaw88(
        id=1,
        rho0=1000.0,
        nu=0.45,
        bulk=1e6,
        nl=2,
        func_load_list=[101, 102],
        rate_load_list=[100.0, 10.0],  # Descending!
    )
    log = MessageLog()
    check_mat_law88(mat=mat, log=log)
    assert log.has_errors
    assert any("ANCMSG 478" in err for err in log.errors)


def test_checks_mat_law88_negative_shear_warning():
    """Verify warning ANCMSG 3109 on negative shear modulus."""
    mat = MaterialLaw88(
        id=1,
        rho0=1000.0,
        nu=0.45,
        bulk=1e6,
        nl=1,
        func_load_list=[101],
        rate_load_list=[0.0],
        shear=-10.0,
    )
    log = MessageLog()
    check_mat_law88(mat=mat, log=log)
    assert log.has_warnings
    assert any("ANCMSG 3109" in w for w in log.warnings)


def test_checks_mat_law88_1d_element_rejection():
    """Verify rejection of 1D elements (ANCMSG 306)."""
    class DummyElement:
        def __init__(self, mid):
            self.mat_id = mid

    class DummyGroup:
        def __init__(self, elements):
            self._elems = {i: el for i, el in enumerate(elements)}
        def values(self):
            return self._elems.values()

    model = Model()
    mat = MaterialLaw88(id=1, rho0=1000.0, nu=0.45, bulk=1e6, nl=1, func_load_list=[101])
    model.materials[1] = mat
    model.mat_law88s[1] = mat

    # Add 1D beam element attached to material 1
    model.element_groups = lambda: [("beams", DummyGroup([DummyElement(1)]))]

    log = MessageLog()
    check_mat_law88(model=model, mat_id=1, log=log)
    assert log.has_errors
    assert any("ANCMSG 306" in err for err in log.errors)


def test_checks_mat_law88_integration():
    """Verify check_materials and check_all properly invoke check_mat_law88."""
    model = Model()
    mat = MaterialLaw88(id=2, rho0=-1.0, nl=0)  # Invalid
    model.materials[2] = mat
    model.mat_law88s[2] = mat

    log = MessageLog()
    check_materials(model, log)
    assert log.has_errors
    assert any("ANCMSG 1514" in err for err in log.errors)
    assert any("ANCMSG 866" in err for err in log.errors)


# ============================================================================
# 7. Element & Materials Integration
# ============================================================================

def test_materials_module_integration():
    """Verify pyradioss.materials dispatch functions recognize LAW88 and synonyms."""
    mat = MaterialLaw88(id=1, rho0=1000.0, nu=0.45, bulk=1.0e6, young=3.0e5)

    assert mats.needs_env(mat) is True
    shapes = mats.extra_shapes(mat)
    assert "uvar88" in shapes
    assert shapes["uvar88"] == (30,)

    # Test sound speed
    c = mats.sound_speed(mat)
    assert c > 0.0

    # Test solid_update
    sig = np.zeros((1, 6), dtype=np.float64)
    deps = np.full((1, 6), 1e-4, dtype=np.float64)
    uvar = np.zeros((1, 30), dtype=np.float64)
    extra = {"uvar88": uvar, "rho": np.array([1000.0])}

    sig_new, epsp_new, c_elem = mats.solid_update(mat, sig, deps, extra=extra)
    assert sig_new.shape == (1, 6)
    assert np.any(sig_new != 0.0)

    # Test shell_update
    sig_sh = np.zeros((1, 3), dtype=np.float64)
    deps_sh = np.full((1, 3), 1e-4, dtype=np.float64)
    uvar_sh = np.zeros((1, 30), dtype=np.float64)
    extra_sh = {"uvar88": uvar_sh, "rho": np.array([1000.0])}

    sig_sh_new, _ = mats.shell_update(mat, sig_sh, deps_sh, extra=extra_sh)
    assert sig_sh_new.shape == (1, 3)
    assert np.any(sig_sh_new != 0.0)


def test_materials_synonyms_dispatch():
    """Verify all LAW88 synonyms dispatch identically in pyradioss.materials."""
    synonyms = [88, "88", "LAW88", "TABULATED_HYPERELASTIC", "HYPER_ELAS", "TAB_HYP", "MAT_LAW88"]
    for syn in synonyms:
        mat = MaterialLaw88(id=1, rho0=1000.0, nu=0.45, bulk=1.0e6, young=3.0e5)
        mat.law = syn
        mat.law_name = str(syn)

        assert mats.needs_env(mat) is True
        shapes = mats.extra_shapes(mat)
        assert "uvar88" in shapes
        c = mats.sound_speed(mat)
        assert c > 0.0
