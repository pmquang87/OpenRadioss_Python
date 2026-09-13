"""Tests for /MAT/LAW109 starter diagnostics and model entities (Milestone M578).

Validates:
  1. ANCMSG 1514: Density rho > 0 and Young's modulus E > 0.
  2. ANCMSG 3068: Poisson's ratio bounds 0 <= Nu < 0.5.
  3. Taylor-Quinney coefficient bounds: 0 <= ETA <= 1.0.
  4. Yield stress scale factor bounds: Yscale > 0.
  5. ANCMSG 306: Element compatibility rejection for 1D elements (beam, truss, spring).
  6. Model entity classes and aliases: MaterialLaw109, MatLaw109, MatTabPlas,
     MaterialTabPlas, MatElastoPlasTab, MaterialElastoPlasTab, MatLaw109TabPlas.
  7. Model collections: model.mat_law109s, model.mat_tab_plass, model.mat_elasto_plas_tabs.
"""

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import (
    Material,
    MaterialLaw109,
    MatLaw109,
    MatTabPlas,
    MaterialTabPlas,
    MatElastoPlasTab,
    MaterialElastoPlasTab,
    MatLaw109TabPlas,
    Part,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_mat_law109


def test_starter_check_valid_material():
    """Verify clean starter check for fully valid /MAT/LAW109 card."""
    mat = MaterialLaw109(
        id=1,
        title="Valid Steel 109",
        rho0=7.8e-6,
        young=210000.0,
        nu=0.3,
        cp=450.0,
        eta=0.9,
        tref=293.0,
        tini=293.0,
        tab_yld=10,
        tab_temp=20,
        xscale_h=1.0,
        yscale_h=1.0,
        ismooth=1,
        tab_eta=30,
        xscale_eta=1.0,
    )
    log = MessageLog()
    check_mat_law109(mat, log)
    assert len(log.errors) == 0
    assert len(log.warnings) == 0


def test_starter_check_ancmsg_1514_negative_or_zero_modulus_or_density():
    """Verify ANCMSG 1514 when rho <= 0 or young <= 0."""
    # Zero density
    mat1 = MaterialLaw109(id=1, rho0=0.0, young=210000.0, nu=0.3)
    log1 = MessageLog()
    check_mat_law109(mat1, log1)
    assert len(log1.errors) > 0
    assert any("1514" in msg for msg in log1.errors)

    # Negative density
    mat2 = MaterialLaw109(id=2, rho0=-1.0e-6, young=210000.0, nu=0.3)
    log2 = MessageLog()
    check_mat_law109(mat2, log2)
    assert len(log2.errors) > 0
    assert any("1514" in msg for msg in log2.errors)

    # Zero Young's modulus
    mat3 = MaterialLaw109(id=3, rho0=7.8e-6, young=0.0, nu=0.3)
    log3 = MessageLog()
    check_mat_law109(mat3, log3)
    assert len(log3.errors) > 0
    assert any("1514" in msg for msg in log3.errors)

    # Negative Young's modulus
    mat4 = MaterialLaw109(id=4, rho0=7.8e-6, young=-100000.0, nu=0.3)
    log4 = MessageLog()
    check_mat_law109(mat4, log4)
    assert len(log4.errors) > 0
    assert any("1514" in msg for msg in log4.errors)


def test_starter_check_poisson_ratio_bounds():
    """Verify Poisson's ratio bounds: 0 <= Nu < 0.5."""
    # Negative Poisson ratio
    mat1 = MaterialLaw109(id=1, rho0=7.8e-6, young=210000.0, nu=-0.1)
    log1 = MessageLog()
    check_mat_law109(mat1, log1)
    assert len(log1.errors) > 0

    # Incompressible limit Nu = 0.5
    mat2 = MaterialLaw109(id=2, rho0=7.8e-6, young=210000.0, nu=0.5)
    log2 = MessageLog()
    check_mat_law109(mat2, log2)
    assert len(log2.errors) > 0


def test_starter_check_eta_and_yscale_bounds():
    """Verify ETA in [0, 1] and Yscale > 0."""
    # ETA < 0
    mat1 = MaterialLaw109(id=1, rho0=7.8e-6, young=210000.0, nu=0.3, eta=-0.1)
    log1 = MessageLog()
    check_mat_law109(mat1, log1)
    assert len(log1.errors) > 0

    # ETA > 1
    mat2 = MaterialLaw109(id=2, rho0=7.8e-6, young=210000.0, nu=0.3, eta=1.5)
    log2 = MessageLog()
    check_mat_law109(mat2, log2)
    assert len(log2.errors) > 0

    # Yscale <= 0
    mat3 = MaterialLaw109(id=3, rho0=7.8e-6, young=210000.0, nu=0.3, yscale_h=0.0)
    log3 = MessageLog()
    check_mat_law109(mat3, log3)
    assert len(log3.errors) > 0


def test_starter_check_ancmsg_306_element_type_compatibility():
    """Verify ANCMSG 306 rejects 1D elements (beam, truss, spring) assigned to LAW109."""
    model = Model()
    mat = MaterialLaw109(id=1, rho0=7.8e-6, young=210000.0, nu=0.3)
    model.materials[1] = mat

    part = Part(id=1, mat_id=1, prop_id=1)
    model.parts[1] = part

    # Add 1D beam element assigned to part 1
    class DummyBeam:
        part_id = 1
    model.beams = {1: DummyBeam()}

    log = MessageLog()
    check_mat_law109(mat, log, model=model)
    assert len(log.errors) > 0
    assert any("306" in msg for msg in log.errors)

    # Remove beam: passes cleanly
    model.beams = {}
    log2 = MessageLog()
    check_mat_law109(mat, log2, model=model)
    assert not any("306" in msg for msg in log2.errors)


def test_model_entities_and_collection_access():
    """Verify MaterialLaw109 synonyms, properties, and model collection access."""
    # Test class synonyms
    assert issubclass(MatLaw109, MaterialLaw109)
    assert issubclass(MatTabPlas, MaterialLaw109)
    assert issubclass(MaterialTabPlas, MaterialLaw109)
    assert issubclass(MatElastoPlasTab, MaterialLaw109)
    assert issubclass(MaterialElastoPlasTab, MaterialLaw109)
    assert issubclass(MatLaw109TabPlas, MaterialLaw109)

    mat = MaterialLaw109(
        id=55,
        title="High Strength Steel 109",
        rho0=7.85e-6,
        refer_rho=7.85e-6,
        young=205000.0,
        nu=0.29,
        cp=460.0,
        eta=0.95,
        tref=293.0,
        tini=300.0,
        tab_yld=101,
        tab_temp=102,
        xscale_h=1.0,
        yscale_h=1.0,
        ismooth=2,
        tab_eta=103,
        xscale_eta=1.0,
    )

    # Property aliases
    assert mat.rho == pytest.approx(7.85e-6)
    assert mat.rho0 == pytest.approx(7.85e-6)
    assert mat.young == pytest.approx(205000.0)
    assert mat.e == pytest.approx(205000.0)
    assert mat.E == pytest.approx(205000.0)
    assert mat.nu == pytest.approx(0.29)
    assert mat.Nu == pytest.approx(0.29)
    assert mat.cp == pytest.approx(460.0)
    assert mat.tref == pytest.approx(293.0)
    assert mat.tini == pytest.approx(300.0)
    assert mat.tab_yld == 101
    assert mat.tab_temp == 102
    assert mat.ismooth == 2
    assert mat.xrate == pytest.approx(1.0)

    # Elastic derived constants
    expected_g = 205000.0 / (2.0 * (1.0 + 0.29))
    expected_bulk = 205000.0 / (3.0 * (1.0 - 2.0 * 0.29))
    assert np.isclose(mat.G, expected_g)
    assert np.isclose(mat.bulk, expected_bulk)

    # Sound speeds
    expected_ssp_solid = math.sqrt((expected_bulk + 4.0 / 3.0 * expected_g) / 7.85e-6)
    a11 = 205000.0 / (1.0 - 0.29**2)
    expected_ssp_shell = math.sqrt(a11 / 7.85e-6)
    assert np.isclose(mat.sound_speed_solid(), expected_ssp_solid)
    assert np.isclose(mat.sound_speed_shell(), expected_ssp_shell)
    assert np.isclose(mat.sound_speed(), expected_ssp_solid)

    # Model collection access
    model = Model()
    model.mat_law109s[55] = mat

    assert 55 in model.mat_tab_plass
    assert model.mat_tab_plass[55] is mat
    assert 55 in model.mat_elasto_plas_tabs
    assert model.mat_elasto_plas_tabs[55] is mat
