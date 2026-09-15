"""
Tests for /MAT/LAW107 starter diagnostics and model entities (Milestone M577).

Validates:
  1. ANCMSG 1514: Density rho > 0 and Young's moduli E1, E2, E3 > 0.
  2. ANCMSG 3068: Poisson's ratio product nu12 * nu21 < 1.0.
  3. ANCMSG 1815-1822: Plastic potential parameter inequalities:
     - K1 >= 0 (ANCMSG 1815)
     - K2 <= 0 (ANCMSG 1816)
     - K3 >= 0 (ANCMSG 1817)
     - K4 >= 0 (ANCMSG 1818)
     - K5 >= 0 (ANCMSG 1819)
     - K6 >= 0 (ANCMSG 1820)
     - |K1| > |K2| (ANCMSG 1821)
     - |K2| > |K4| (ANCMSG 1822)
  4. ANCMSG 306: Element compatibility rejection for 1D elements (truss, beam, spring).
  5. Model entity classes and aliases: MaterialLaw107, MatPaperLight, MatPlasPaperLight, MatPfeiffer.
  6. Model collections: model.mat_paper_lights, model.mat_plas_paper_lights, model.mat_pfeiffers.
"""

import pytest
import numpy as np

from pyradioss.model.entities import (
    Material,
    MaterialLaw107,
    MatLaw107,
    MatPaperLight,
    MatPlasPaperLight,
    MatPfeiffer,
    Part,
)
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.checks import check_mat_law107


def test_starter_check_valid_material():
    """Verify clean starter check for fully valid /MAT/LAW107 card."""
    mat = MaterialLaw107(
        id=1,
        rho=1.0e-6,
        young1=12000.0,
        young2=6000.0,
        young3=1000.0,
        nu21=0.15,
        g12=3000.0,
        g23=1500.0,
        g31=2000.0,
        k1=1.0,
        k2=-0.6,
        k3=0.8,
        k4=0.4,
        k5=0.5,
        k6=0.3,
    )
    log = MessageLog()
    check_mat_law107(mat, log)
    assert len(log.errors) == 0
    assert len(log.warnings) == 0


def test_starter_check_ancmsg_1514_negative_or_zero_moduli_or_density():
    """Verify ANCMSG 1514 when rho <= 0 or any Young's modulus <= 0."""
    # Zero density
    mat1 = MaterialLaw107(id=1, rho=0.0, young1=10000.0, young2=5000.0, young3=1000.0)
    log1 = MessageLog()
    check_mat_law107(mat1, log1)
    assert len(log1.errors) > 0
    assert any("1514" in msg for msg in log1.errors)

    # Negative Young modulus E1
    mat2 = MaterialLaw107(id=2, rho=1.0e-6, young1=-10000.0, young2=5000.0, young3=1000.0)
    log2 = MessageLog()
    check_mat_law107(mat2, log2)
    assert len(log2.errors) > 0
    assert any("1514" in msg for msg in log2.errors)

    # Zero Young modulus E2
    mat3 = MaterialLaw107(id=3, rho=1.0e-6, young1=10000.0, young2=0.0, young3=1000.0)
    log3 = MessageLog()
    check_mat_law107(mat3, log3)
    assert len(log3.errors) > 0
    assert any("1514" in msg for msg in log3.errors)


def test_starter_check_ancmsg_3068_poisson_product():
    """Verify ANCMSG 3068 when nu12 * nu21 >= 1.0."""
    # nu21 = 0.8, E1 = 12000, E2 = 6000 -> nu12 = 0.8 * 12000 / 6000 = 1.6
    # nu12 * nu21 = 1.6 * 0.8 = 1.28 >= 1.0 -> ERROR
    mat = MaterialLaw107(
        id=1,
        rho=1.0e-6,
        young1=12000.0,
        young2=6000.0,
        young3=1000.0,
        nu21=0.8,
        k1=1.0,
        k2=-0.5,
        k3=0.8,
        k4=0.4,
    )
    log = MessageLog()
    check_mat_law107(mat, log)
    assert len(log.errors) > 0
    assert any("3068" in msg for msg in log.errors)


def test_starter_check_ancmsg_1815_to_1822_potential_parameters():
    """Verify ANCMSG 1815..1822 for invalid non-associated potential parameters."""
    base_kwargs = dict(
        id=1,
        rho=1.0e-6,
        young1=10000.0,
        young2=5000.0,
        young3=1000.0,
        nu21=0.2,
        k1=1.0,
        k2=-0.5,
        k3=0.8,
        k4=0.4,
        k5=0.5,
        k6=0.3,
    )

    # 1815: K1 < 0
    m = MaterialLaw107(**{**base_kwargs, "k1": -0.1})
    log = MessageLog()
    check_mat_law107(m, log)
    assert any("1815" in msg for msg in log.errors)

    # 1816: K2 > 0
    m = MaterialLaw107(**{**base_kwargs, "k2": 0.5})
    log = MessageLog()
    check_mat_law107(m, log)
    assert any("1816" in msg for msg in log.errors)

    # 1817: K3 < 0
    m = MaterialLaw107(**{**base_kwargs, "k3": -0.2})
    log = MessageLog()
    check_mat_law107(m, log)
    assert any("1817" in msg for msg in log.errors)

    # 1818: K4 < 0
    m = MaterialLaw107(**{**base_kwargs, "k4": -0.1})
    log = MessageLog()
    check_mat_law107(m, log)
    assert any("1818" in msg for msg in log.errors)

    # 1819: K5 < 0
    m = MaterialLaw107(**{**base_kwargs, "k5": -0.5})
    log = MessageLog()
    check_mat_law107(m, log)
    assert any("1819" in msg for msg in log.errors)

    # 1820: K6 < 0
    m = MaterialLaw107(**{**base_kwargs, "k6": -0.2})
    log = MessageLog()
    check_mat_law107(m, log)
    assert any("1820" in msg for msg in log.errors)

    # 1821: |K1| <= |K2| (e.g. K1=0.5, K2=-0.8 -> |0.5| < |0.8|)
    m = MaterialLaw107(**{**base_kwargs, "k1": 0.5, "k2": -0.8})
    log = MessageLog()
    check_mat_law107(m, log)
    assert any("1821" in msg for msg in log.errors)

    # 1822: |K2| <= |K4| (e.g. K2=-0.3, K4=0.6 -> |0.3| < |0.6|)
    m = MaterialLaw107(**{**base_kwargs, "k2": -0.3, "k4": 0.6})
    log = MessageLog()
    check_mat_law107(m, log)
    assert any("1822" in msg for msg in log.errors)


def test_starter_check_ancmsg_306_element_type_compatibility():
    """Verify ANCMSG 306 rejects 1D elements (beam, truss, spring) assigned to LAW107."""
    model = Model()
    mat = MaterialLaw107(
        id=1,
        rho=1.0e-6,
        young1=10000.0,
        young2=5000.0,
        young3=1000.0,
        nu21=0.2,
        k1=1.0,
        k2=-0.5,
        k3=0.8,
        k4=0.4,
    )
    model.materials[1] = mat

    part = Part(id=1, mat_id=1, prop_id=1)
    model.parts[1] = part

    # Add 1D beam element assigned to part 1
    class DummyBeam:
        part_id = 1
    model.beams = {1: DummyBeam()}

    log = MessageLog()
    check_mat_law107(mat, log, model=model)
    assert len(log.errors) > 0
    assert any("306" in msg for msg in log.errors)

    # Now remove beam: should pass without 306
    model.beams = {}

    log2 = MessageLog()
    check_mat_law107(mat, log2, model=model)
    assert not any("306" in msg for msg in log2.errors)


def test_model_entities_and_collection_access():
    """Verify MaterialLaw107 synonyms, properties, and model collection access."""
    # Test class synonyms
    assert issubclass(MatPaperLight, MaterialLaw107)
    assert issubclass(MatPlasPaperLight, MaterialLaw107)
    assert issubclass(MatPfeiffer, MaterialLaw107)

    mat = MaterialLaw107(
        id=10,
        title="Paperboard Grade A",
        rho=8.5e-7,
        young1=14000.0,
        young2=7000.0,
        young3=1200.0,
        nu21=0.18,
        g12=3500.0,
        g23=1600.0,
        g31=2200.0,
    )

    # Verify properties
    assert mat.E1 == 14000.0
    assert mat.E2 == 7000.0
    assert mat.E3 == 1200.0
    assert mat.Young1 == 14000.0
    assert mat.Young2 == 7000.0
    assert mat.Young3 == 1200.0
    assert np.isclose(mat.nu12, 0.18 * 14000.0 / 7000.0)
    assert mat.sound_speed > 0.0
    assert mat.sound_speed_shell > 0.0

    # Model collection access
    model = Model()
    model.mat_law107s[10] = mat

    assert 10 in model.mat_paper_lights
    assert model.mat_paper_lights[10] is mat
    assert 10 in model.mat_plas_paper_lights
    assert 10 in model.mat_pfeiffers
