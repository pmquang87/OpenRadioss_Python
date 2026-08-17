"""Tests for Milestone M186: Hydrodynamic Fluid with Turbulence, Boundary Layer / k-Epsilon Fluid,
Foam with Gas & Hysteresis, Multi-Material Mixture, Barlat 2000 3D Plasticity,
Kinematic Joint, Muscle Spring, and Stitch Seam Fastener Properties Suite.
"""
from __future__ import annotations
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, deck_text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_mat_law6_hydro_fluid(tmp_path: Path):
    """Test /MAT/LAW6 (/MAT/VISC_FLUID, /MAT/HYDRO, /MAT/K-EPS) in fixed and free format."""
    # Fixed format with k-epsilon turbulence parameters
    c1 = f"{1.0e-3:20.6e}{1.0e-3:20.6e}"
    c2 = f"{1.5e-4:20.6e}"
    c3 = f"{1000.0:20.6e}{2000.0:20.6e}{3000.0:20.6e}{4000.0:20.6e}"
    c4 = f"{-1.0e5:20.6e}{0.0:20.6e}"
    c5 = f"{0.4:20.6e}{0.2:20.6e}{1.0e6:20.6e}"
    c6 = f"{0.05:20.6e}{0.1:20.6e}"
    c7 = f"{0.09:20.6e}{1.0:20.6e}{1.3:20.6e}{1.0:20.6e}"
    c8 = f"{1.44:20.6e}{1.92:20.6e}{0.0:20.6e}"
    c9 = f"{0.41:20.6e}{9.0:20.6e}{1.0:20.6e}{1.0:20.6e}"

    deck_fixed = f"""/BEGIN
Test MAT LAW6 Fixed
       2026         0
/MAT/LAW6/1
Hydrodynamic Fluid with Turbulence
{c1}
{c2}
{c3}
{c4}
{c5}
{c6}
{c7}
{c8}
{c9}
/END
"""
    model1, log1 = _parse_starter(tmp_path, deck_fixed)
    assert len(log1.errors) == 0, f"Errors: {log1.errors}"
    assert 1 in model1.mat_law6s
    m1 = model1.mat_law6s[1]
    assert m1.rho == pytest.approx(1.0e-3)
    assert m1.nu == pytest.approx(1.5e-4)
    assert m1.c0 == pytest.approx(1000.0)
    assert m1.c1 == pytest.approx(2000.0)
    assert m1.c2 == pytest.approx(3000.0)
    assert m1.c3 == pytest.approx(4000.0)
    assert m1.pmin == pytest.approx(-1.0e5)
    assert m1.c4 == pytest.approx(0.4)
    assert m1.c5 == pytest.approx(0.2)
    assert m1.e0 == pytest.approx(1.0e6)
    assert m1.r0k0 == pytest.approx(0.05)
    assert m1.ssl == pytest.approx(0.1)
    assert m1.c_mu == pytest.approx(0.09)
    assert m1.sig_k == pytest.approx(1.0)
    assert m1.sig_eps == pytest.approx(1.3)
    assert m1.kappa == pytest.approx(0.41)

    # Free format with alias /MAT/VISC_FLUID
    deck_free = """/BEGIN
Test MAT LAW6 Free
/MAT/VISC_FLUID/2
Water Fluid Model
1.0e-3
1.0e-4
1500.0 0.0 0.0 0.0
-1.0e6 0.0
0.0 0.0 0.0
/END
"""
    model2, log2 = _parse_starter(tmp_path, deck_free)
    assert len(log2.errors) == 0, f"Errors: {log2.errors}"
    assert 2 in model2.mat_law6s
    m2 = model2.mat_law6s[2]
    assert m2.rho == pytest.approx(1.0e-3)
    assert m2.nu == pytest.approx(1.0e-4)
    assert m2.c0 == pytest.approx(1500.0)
    assert m2.pmin == pytest.approx(-1.0e6)
    assert 2 in model2.materials
    assert model2.materials[2].law == 6


def test_mat_law11_bound_bkeps(tmp_path: Path):
    """Test /MAT/LAW11 (/MAT/BOUND, /MAT/B-K-EPS) in fixed and free format."""
    # Fixed format /MAT/BOUND
    c1 = f"{1.2e-3:20.6e}"
    c2 = f"{1:10d}{0.0:20.6e}{1.0:20.6e}"
    c3 = f"{101:10d}{1.4:20.6e}{0.05:20.6e}"
    deck_bound = f"""/BEGIN
Test MAT LAW11 Bound
       2026         0
/MAT/BOUND/11
Boundary Fluid Law
{c1}
{c2}
{c3}
/END
"""
    model1, log1 = _parse_starter(tmp_path, deck_bound)
    assert len(log1.errors) == 0, f"Errors: {log1.errors}"
    assert 11 in model1.mat_law11s
    m1 = model1.mat_law11s[11]
    assert m1.rho == pytest.approx(1.2e-3)
    assert m1.itype == 1
    assert m1.node1 == 101
    assert m1.gamma == pytest.approx(1.4)
    assert 11 in model1.materials
    assert model1.materials[11].law == 11

    # Free format with /MAT/B-K-EPS
    deck_bkeps = """/BEGIN
Test MAT LAW11 Free
/MAT/B-K-EPS/12
Turbulent Boundary Layer
1.25e-3
2 0.0 1.0
201 1.4 0.02
/END
"""
    model2, log2 = _parse_starter(tmp_path, deck_bkeps)
    assert len(log2.errors) == 0, f"Errors: {log2.errors}"
    assert 12 in model2.mat_law11s
    m2 = model2.mat_law11s[12]
    assert m2.rho == pytest.approx(1.25e-3)
    assert m2.itype == 2
    assert m2.node1 == 201


def test_mat_law77_foam_air(tmp_path: Path):
    """Test /MAT/LAW77 (/MAT/FOAM_AIR, /MAT/FOAM_HYST) with loading/unloading curves."""
    # Fixed format
    c1 = f"{5.0e-5:20.6e}"
    c2 = f"{10.0:20.6e}{0.3:20.6e}{50.0:20.6e}{2.5:20.6e}"
    # FCUT, FSMOOTH, NLOAD, NUNLOAD, IFLAG, SHAPE, HYS
    c3 = f"{1000.0:20.6e}{1:10d}{2:10d}{1:10d}{1:10d}{0.5:20.6e}{0.2:20.6e}"
    load1 = f"{101:10d}{0.0:20.6e}{1.0:20.6e}"
    load2 = f"{102:10d}{100.0:20.6e}{1.2:20.6e}"
    unload1 = f"{201:10d}{0.0:20.6e}{1.0:20.6e}"
    c4 = f"{1.2e-3:20.6e}{1.0e5:20.6e}{1.4:20.6e}{' ':20s}{0.8:20.6e}"
    c5 = f"{1.2e-3:20.6e}{1.0e5:20.6e}{1:10d}{1:10d}"

    deck_fixed = f"""/BEGIN
Test MAT LAW77 Fixed
       2026         0
/MAT/LAW77/77
Foam with Air Cavity
{c1}
{c2}
{c3}
{load1}
{load2}
{unload1}
{c4}
{c5}
/END
"""
    model1, log1 = _parse_starter(tmp_path, deck_fixed)
    assert len(log1.errors) == 0, f"Errors: {log1.errors}"
    assert 77 in model1.mat_law77s
    m1 = model1.mat_law77s[77]
    assert m1.rho == pytest.approx(5.0e-5)
    assert m1.e0 == pytest.approx(10.0)
    assert m1.nu == pytest.approx(0.3)
    assert m1.emax == pytest.approx(50.0)
    assert m1.epsmax == pytest.approx(2.5)
    assert m1.fcut == pytest.approx(1000.0)
    assert m1.fsmooth == 1
    assert m1.nload == 2
    assert m1.nunload == 1
    assert len(m1.load_curves) == 2
    assert m1.load_curves[0].fct_id == 101
    assert m1.load_curves[1].strain_rate == pytest.approx(100.0)
    assert len(m1.unload_curves) == 1
    assert m1.unload_curves[0].fct_id == 201
    assert m1.p0 == pytest.approx(1.0e5)
    assert m1.gamma == pytest.approx(1.4)
    assert m1.poros == pytest.approx(0.8)
    assert 77 in model1.materials
    assert model1.materials[77].law == 77

    # Free format with alias /MAT/FOAM_AIR
    deck_free = """/BEGIN
Test MAT LAW77 Free
/MAT/FOAM_AIR/78
Foam Air Free Format
6.0e-5
15.0 0.35 60.0 3.0
500.0 0 1 0 0 0.0 0.0
301 0.0 1.0
1.2e-3 1.0e5 1.4 0.9
1.2e-3 1.0e5 0 0
/END
"""
    model2, log2 = _parse_starter(tmp_path, deck_free)
    assert len(log2.errors) == 0, f"Errors: {log2.errors}"
    assert 78 in model2.mat_law77s
    m2 = model2.mat_law77s[78]
    assert m2.rho == pytest.approx(6.0e-5)
    assert m2.nload == 1
    assert m2.nunload == 0
    assert len(m2.load_curves) == 1
    assert m2.load_curves[0].fct_id == 301


def test_mat_law151_multimat(tmp_path: Path):
    """Test /MAT/LAW151 (/MAT/MULTIFLUID, /MAT/MULTI_MAT) mixture law."""
    # Fixed format
    entry1 = f"{1:10d}{0.6:20.6e}"
    entry2 = f"{2:10d}{0.4:20.6e}"
    deck_fixed = f"""/BEGIN
Test MAT LAW151 Fixed
       2026         0
/MAT/LAW151/151
Multi-Material Mixture
{entry1}
{entry2}
/END
"""
    model1, log1 = _parse_starter(tmp_path, deck_fixed)
    assert len(log1.errors) == 0, f"Errors: {log1.errors}"
    assert 151 in model1.mat_law151s
    m1 = model1.mat_law151s[151]
    assert len(m1.fractions) == 2
    assert m1.fractions[0].mat_id == 1
    assert m1.fractions[0].vol_frac == pytest.approx(0.6)
    assert m1.fractions[1].mat_id == 2
    assert m1.fractions[1].vol_frac == pytest.approx(0.4)
    assert 151 in model1.materials
    assert model1.materials[151].law == 151

    # Free format with alias /MAT/MULTIFLUID
    deck_free = """/BEGIN
Test MAT LAW151 Free
/MAT/MULTIFLUID/152
Two-Fluid Mixture
10 0.75
20 0.25
/END
"""
    model2, log2 = _parse_starter(tmp_path, deck_free)
    assert len(log2.errors) == 0, f"Errors: {log2.errors}"
    assert 152 in model2.mat_law151s
    m2 = model2.mat_law151s[152]
    assert len(m2.fractions) == 2
    assert m2.fractions[0].mat_id == 10
    assert m2.fractions[0].vol_frac == pytest.approx(0.75)


def test_mat_law187_barlat3d(tmp_path: Path):
    """Test /MAT/LAW187 (/MAT/BARLAT20003D, /MAT/BARLAT_3D) in fixed and free format."""
    # Fixed format
    c1 = f"{2.7e-3:20.6e}"
    c2 = f"{70000.0:20.6e}{0.33:20.6e}{0:10d}{0:10d}{1.0e-3:20.6e}{0.0:20.6e}"
    c3 = f"{1.0:20.6e}{1.05:20.6e}{0.98:20.6e}{1.02:20.6e}"
    c4 = f"{1.01:20.6e}{0.99:20.6e}{1.03:20.6e}{0.97:20.6e}"
    c5 = f"{1.04:20.6e}{0.96:20.6e}{1.0:20.6e}{1.0:20.6e}"
    c6 = f"{' ':10s}{8:10d}{1.0:20.6e}{0.2:20.6e}{1000.0:20.6e}{1:10d}{2:10d}"
    c7 = f"{300.0:20.6e}{0.002:20.6e}{0.0:20.6e}{0.0:20.6e}{0.0:20.6e}"
    r1 = f"{501:10d}{' ':10s}{1.0:20.6e}{0.0:20.6e}"
    r2 = f"{502:10d}{' ':10s}{1.1:20.6e}{100.0:20.6e}"

    deck_fixed = f"""/BEGIN
Test MAT LAW187 Fixed
       2026         0
/MAT/LAW187/187
Barlat 2000 3D Aluminum Alloy
{c1}
{c2}
{c3}
{c4}
{c5}
{c6}
{c7}
{r1}
{r2}
/END
"""
    model1, log1 = _parse_starter(tmp_path, deck_fixed)
    assert len(log1.errors) == 0, f"Errors: {log1.errors}"
    assert 187 in model1.mat_law187s
    m1 = model1.mat_law187s[187]
    assert m1.rho == pytest.approx(2.7e-3)
    assert m1.e == pytest.approx(70000.0)
    assert m1.nu == pytest.approx(0.33)
    assert m1.alpha1 == pytest.approx(1.0)
    assert m1.alpha2 == pytest.approx(1.05)
    assert m1.alpha3 == pytest.approx(0.98)
    assert m1.alpha12 == pytest.approx(1.0)
    assert m1.a_exp == 8
    assert m1.alpha_xy == pytest.approx(1.0)
    assert m1.n_exp == pytest.approx(0.2)
    assert m1.nrate == 2
    assert len(m1.rates) == 2
    assert m1.rates[0].fct_id == 501
    assert m1.rates[1].fct_id == 502
    assert m1.rates[1].strain_rate == pytest.approx(100.0)
    assert 187 in model1.materials
    assert model1.materials[187].law == 187

    # Free format with alias /MAT/BARLAT20003D
    deck_free = """/BEGIN
Test MAT LAW187 Free
/MAT/BARLAT20003D/188
Barlat 3D Free Format
2.8e-3
72000.0 0.33 0 0 1.0e-3 0.0
1.0 1.0 1.0 1.0
1.0 1.0 1.0 1.0
1.0 1.0 1.0 1.0
8 1.0 0.25 500.0 0 1
350.0 0.005 0.0 0.0 0.0
601 1.0 0.0
/END
"""
    model2, log2 = _parse_starter(tmp_path, deck_free)
    assert len(log2.errors) == 0, f"Errors: {log2.errors}"
    assert 188 in model2.mat_law187s
    m2 = model2.mat_law187s[188]
    assert m2.rho == pytest.approx(2.8e-3)
    assert m2.e == pytest.approx(72000.0)
    assert m2.a_exp == 8
    assert len(m2.rates) == 1
    assert m2.rates[0].fct_id == 601


def test_prop_type33_kjoint(tmp_path: Path):
    """Test /PROP/TYPE33 (/PROP/KJOINT, /PROP/KINEMATIC_JOINT) in fixed and free format."""
    # Fixed format
    c1 = f"{1:10d}{1:10d}"
    c2 = f"{10:10d}{20:10d}{1.0e6:20.6e}{0.05:20.6e}"
    c3 = f"{1.0e7:20.6e}{5.0e4:20.6e}{6.0e4:20.6e}{7.0e4:20.6e}"
    c4 = f"{101:10d}{102:10d}{103:10d}"
    c5 = f"{500.0:20.6e}{600.0:20.6e}{700.0:20.6e}"
    c6 = f"{201:10d}{202:10d}{203:10d}"

    deck_fixed = f"""/BEGIN
Test PROP TYPE33 Fixed
       2026         0
/PROP/TYPE33/33
Kinematic Revolute Joint
{c1}
{c2}
{c3}
{c4}
{c5}
{c6}
/END
"""
    model1, log1 = _parse_starter(tmp_path, deck_fixed)
    assert len(log1.errors) == 0, f"Errors: {log1.errors}"
    assert 33 in model1.prop_type33s
    p1 = model1.prop_type33s[33]
    assert p1.joint_type == 1
    assert p1.skew_flag == 1
    assert p1.id_sk1 == 10
    assert p1.id_sk2 == 20
    assert p1.xk == pytest.approx(1.0e6)
    assert p1.cr == pytest.approx(0.05)
    assert p1.kn == pytest.approx(1.0e7)
    assert p1.krx == pytest.approx(5.0e4)
    assert p1.kry == pytest.approx(6.0e4)
    assert p1.krz == pytest.approx(7.0e4)
    assert p1.xr_fun == 101
    assert p1.yr_fun == 102
    assert p1.zr_fun == 103
    assert p1.crx == pytest.approx(500.0)
    assert 33 in model1.properties
    assert model1.properties[33].type == 33

    # Free format with alias /PROP/KJOINT
    deck_free = """/BEGIN
Test PROP TYPE33 Free
/PROP/KJOINT/34
Ball Joint Model
8 0
5 6 2.0e6 0.1
5.0e6
/END
"""
    model2, log2 = _parse_starter(tmp_path, deck_free)
    assert len(log2.errors) == 0, f"Errors: {log2.errors}"
    assert 34 in model2.prop_type33s
    p2 = model2.prop_type33s[34]
    assert p2.joint_type == 8
    assert p2.kn == pytest.approx(5.0e6)


def test_prop_type46_spr_muscle(tmp_path: Path):
    """Test /PROP/TYPE46 (/PROP/SPR_MUSCLE, /PROP/MUSCLE) in fixed and free format."""
    # Fixed format
    c1 = f"{0.05:20.6e}{100.0:20.6e}{10.0:20.6e}{500.0:20.6e}{1000.0:20.6e}"
    c2 = f"{11:10d}{12:10d}{13:10d}{14:10d}{' ':10s}{1:10d}"
    c3 = f"{5.0:20.6e}{0:10d}{1.0:20.6e}{1.0:20.6e}{1.0:20.6e}{1.0:20.6e}"

    deck_fixed = f"""/BEGIN
Test PROP TYPE46 Fixed
       2026         0
/PROP/TYPE46/46
Active Skeletal Muscle Model
{c1}
{c2}
{c3}
/END
"""
    model1, log1 = _parse_starter(tmp_path, deck_fixed)
    assert len(log1.errors) == 0, f"Errors: {log1.errors}"
    assert 46 in model1.prop_type46s
    p1 = model1.prop_type46s[46]
    assert p1.mass == pytest.approx(0.05)
    assert p1.stiff0 == pytest.approx(100.0)
    assert p1.vel_max == pytest.approx(10.0)
    assert p1.nforce == pytest.approx(500.0)
    assert p1.stiff1 == pytest.approx(1000.0)
    assert p1.fun_a1 == 11
    assert p1.fun_b1 == 12
    assert p1.fun_c1 == 13
    assert p1.fun_d1 == 14
    assert p1.mat_imass == 1
    assert p1.damp1 == pytest.approx(5.0)
    assert p1.epsi == 0
    assert 46 in model1.properties
    assert model1.properties[46].type == 46

    # Free format with alias /PROP/MUSCLE
    deck_free = """/BEGIN
Test PROP TYPE46 Free
/PROP/MUSCLE/47
Passive Biceps Element
0.08 120.0 12.0 600.0 1200.0
21 22 23 24 0
6.0 1 1.0 1.0 1.0 1.0
/END
"""
    model2, log2 = _parse_starter(tmp_path, deck_free)
    assert len(log2.errors) == 0, f"Errors: {log2.errors}"
    assert 47 in model2.prop_type46s
    p2 = model2.prop_type46s[47]
    assert p2.mass == pytest.approx(0.08)
    assert p2.fun_a1 == 21
    assert p2.epsi == 1


def test_prop_type35_stitch(tmp_path: Path):
    """Test /PROP/TYPE35 (/PROP/STITCH, /PROP/SEW) in fixed and free format."""
    # Fixed format
    c1 = f"{0.02:20.6e}{5000.0:20.6e}{0.05:20.6e}{1.0e5:20.6e}"
    c2 = f"{31:10d}{32:10d}{33:10d}{34:10d}{0.8:20.6e}{0.001:20.6e}"

    deck_fixed = f"""/BEGIN
Test PROP TYPE35 Fixed
       2026         0
/PROP/TYPE35/35
Seam Stitching Property
{c1}
{c2}
/END
"""
    model1, log1 = _parse_starter(tmp_path, deck_fixed)
    assert len(log1.errors) == 0, f"Errors: {log1.errors}"
    assert 35 in model1.prop_type35s
    p1 = model1.prop_type35s[35]
    assert p1.amas == pytest.approx(0.02)
    assert p1.elastif == pytest.approx(5000.0)
    assert p1.xlim1 == pytest.approx(0.05)
    assert p1.xk == pytest.approx(1.0e5)
    assert p1.fun_a1 == 31
    assert p1.fun_b1 == 32
    assert p1.fun_c1 == 33
    assert p1.fun_d1 == 34
    assert p1.damg == pytest.approx(0.8)
    assert p1.fdelay == pytest.approx(0.001)
    assert 35 in model1.properties
    assert model1.properties[35].type == 35

    # Free format with alias /PROP/STITCH
    deck_free = """/BEGIN
Test PROP TYPE35 Free
/PROP/STITCH/36
Airbag Seam Stitch
0.03 6000.0 0.08 2.0e5
41 42 43 44 0.9 0.002
/END
"""
    model2, log2 = _parse_starter(tmp_path, deck_free)
    assert len(log2.errors) == 0, f"Errors: {log2.errors}"
    assert 36 in model2.prop_type35s
    p2 = model2.prop_type35s[36]
    assert p2.amas == pytest.approx(0.03)
    assert p2.fun_a1 == 41
    assert p2.damg == pytest.approx(0.9)


def test_m186_error_guards(tmp_path: Path):
    """Test error handling and empty card guards for M186 models."""
    deck = """/BEGIN
Test Guards
       2026         0
/MAT/LAW6/100
/MAT/LAW11/101
/MAT/LAW77/102
/MAT/LAW151/103
/MAT/LAW187/104
/PROP/TYPE33/105
/PROP/TYPE46/106
/PROP/TYPE35/107
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) >= 8
