"""Tests for Milestone M141: Extended Material Sub-Object Modifiers, Thermal-Stress & Viscoelastic Relaxation Suite
(/MAT/PLAS_ZERIL, /MAT/PLAS_BODNE, /MAT/VISC_PRONY, /VISC/LPRONY, /MAT/THERM_STRESS, /THERM_STRESS, /DAMP/STIFF, /DAMP/MASS).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m141_mat_plasticity_modifiers(tmp_path):
    deck = """/BEGIN
Test Material Plasticity Modifiers
/MAT/LAW1/1
Base Steel Material
7.85e-6
210.0 0.3
/MAT/PLAS_ZERIL/1
Zerilli-Armstrong Modifier
650.0 200.0 0.005 0.0001 50.0
0.002 0.35 1.0e6
/MAT/LAW1/2
Superalloy Material
8.2e-6
190.0 0.31
/MAT/PLAS_BODNE/2
Bodner-Partom Modifier
1200.0 3000.0 2.5 1.8 1.0e4
0.15 0.25
/PLAS_ZERIL/3
Direct Zerilli Modifier
700.0 250.0 0.004 0.0002 45.0
0.003 0.38 2.0e6
/PLAS_BODNE/4
Direct Bodner Modifier
1100.0 2800.0 2.2 1.6 1.2e4
0.12 0.22
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    # Zerilli-Armstrong
    assert 1 in model.mat_plas_zerils
    pz = model.mat_plas_zerils[1]
    assert pz.c0 == pytest.approx(650.0)
    assert pz.c1 == pytest.approx(200.0)
    assert pz.c2 == pytest.approx(0.005)
    assert pz.c3 == pytest.approx(0.0001)
    assert pz.c4 == pytest.approx(50.0)
    assert pz.c5 == pytest.approx(0.002)
    assert pz.n == pytest.approx(0.35)
    assert pz.fcut == pytest.approx(1.0e6)

    assert 3 in model.mat_plas_zerils
    pz3 = model.mat_plas_zerils[3]
    assert pz3.c0 == pytest.approx(700.0)
    assert pz3.c1 == pytest.approx(250.0)

    # Bodner-Partom
    assert 2 in model.mat_plas_bodnes
    pb = model.mat_plas_bodnes[2]
    assert pb.z0 == pytest.approx(1200.0)
    assert pb.z1 == pytest.approx(3000.0)
    assert pb.m == pytest.approx(2.5)
    assert pb.n == pytest.approx(1.8)
    assert pb.d0 == pytest.approx(1.0e4)
    assert pb.a1 == pytest.approx(0.15)
    assert pb.a2 == pytest.approx(0.25)

    assert 4 in model.mat_plas_bodnes
    pb4 = model.mat_plas_bodnes[4]
    assert pb4.z0 == pytest.approx(1100.0)
    assert pb4.z1 == pytest.approx(2800.0)


def test_m141_viscoelastic_and_thermal_stress(tmp_path):
    deck = """/BEGIN
Test Viscoelasticity and Thermal Stress
/MAT/LAW1/10
Polymer Matrix
1.2e-6
3.5 0.4
/VISC/LPRONY/10
Multi-term Maxwell Prony
3 1 0
0.35 1.5e-3
0.25 1.2e-2
0.15 0.5
/MAT/LAW1/20
Metal Matrix
4.5e-6
110.0 0.32
/MAT/THERM_STRESS/20
Anisotropic Thermal Expansion
1.2e-5 295.15 1.4e-5 1.6e-5
/THERM_STRESS/30
Direct Thermal Stress
2.0e-5 298.15
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    # Viscoelastic Prony
    assert 10 in model.mat_visc_pronys
    vp = model.mat_visc_pronys[10]
    assert vp.order == 3
    assert vp.form == 1
    assert vp.flag_visc == 0
    assert np.allclose(vp.gammas, [0.35, 0.25, 0.15])
    assert np.allclose(vp.taus, [1.5e-3, 1.2e-2, 0.5])

    # Thermal Stress
    assert 20 in model.mat_therm_stresses
    ts = model.mat_therm_stresses[20]
    assert ts.alpha == pytest.approx(1.2e-5)
    assert ts.t0 == pytest.approx(295.15)
    assert ts.alpha_y == pytest.approx(1.4e-5)
    assert ts.alpha_z == pytest.approx(1.6e-5)

    assert 30 in model.mat_therm_stresses
    ts30 = model.mat_therm_stresses[30]
    assert ts30.alpha == pytest.approx(2.0e-5)
    assert ts30.t0 == pytest.approx(298.15)


def test_m141_extended_damping(tmp_path):
    deck = """/BEGIN
Test Extended Damping Cards
/DAMP/STIFF/1
Stiffness Damping
100 0.005 0.0 0.05
/DAMP/MASS/2
Mass Damping
200 0.02 0.0 0.10
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    # DAMP/STIFF
    assert 1 in model.damp_stiffs
    ds = model.damp_stiffs[1]
    assert ds.grnod_id == 100
    assert ds.beta == pytest.approx(0.005)
    assert ds.tstart == pytest.approx(0.0)
    assert ds.tstop == pytest.approx(0.05)

    # DAMP/MASS
    assert len(model.damps) == 1
    dm = model.damps[0]
    assert dm.id == 2
    assert dm.grnod_id == 200
    assert dm.alpha == pytest.approx(0.02)
    assert dm.tstart == pytest.approx(0.0)
    assert dm.tstop == pytest.approx(0.10)
