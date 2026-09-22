# Ported tests for Failure Model Extensions
# FAILWAVE, beam failure, integrated beam failure, thick shell FLD, and XFEM failure
"""Unit tests for failure model extensions: FAILWAVE, beam failure, integrated beam failure, thick shell FLD, XFEM."""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.failure import (
    beam_step,
    failwave,
    fld,
    integrated_beam_step,
    johnson,
    thick_shell_step,
    xfem_step,
)
from pyradioss.failure.failwave import Failwave, seg_intersect


class _FailStub:
    """Stub failure model configuration for testing."""

    def __init__(self, ftype: str, **params):
        self.type = ftype
        self.params = params
        self.ifail_sh = params.get("ifail_sh", 1)


class _CurveStub:
    """Stub function/curve for FLD and tabulated models."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)


# ---------------------------------------------------------------------------
# 1. FAILWAVE Tests
# ---------------------------------------------------------------------------


def test_seg_intersect():
    """Verify 2D line segment intersection calculations."""
    # Classic X intersection at (1, 1)
    ok, xint, yint = seg_intersect(0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0)
    assert ok is True
    assert xint == pytest.approx(1.0)
    assert yint == pytest.approx(1.0)

    # Parallel lines: no intersection
    ok, _, _ = seg_intersect(0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 1.0)
    assert ok is False

    # Lines intersect outside segment bounds
    ok, _, _ = seg_intersect(0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 2.0)
    assert ok is False


def test_failwave_mode1_isotropic_tri_and_quad():
    """Verify Mode 1 isotropic failure wave propagation for tri and quad shells."""
    fw = Failwave(wave_mod=1, num_nodes=6)

    # Element 0: Tri shell with nodes [1, 2, 3] cracked (fwave_el = -1)
    fwave_el_tri = np.array([-1])
    elem_nodes_tri = np.array([[1, 2, 3]])
    fw.set_failwave_nod3(fwave_el_tri, elem_nodes_tri)

    # Before update(), stack has data but active table is 0
    assert np.all(fw.fwave_nod == 0)
    assert fw.maxlev_stack[0] == 1  # Node 1
    assert fw.maxlev_stack[1] == 1  # Node 2
    assert fw.maxlev_stack[2] == 1  # Node 3

    # Commit update
    fw.update()
    assert fw.maxlev[0] == 1
    assert fw.maxlev_stack[0] == 0
    assert fw.fwave_nod[0, 0, 0] == 1

    # Neighbor tri element sharing node 2 and 3: [2, 3, 4]
    fwave_el_adj = np.array([0])
    offly = np.array([1])
    dadv = np.array([1.0])
    elem_nodes_adj = np.array([[2, 3, 4]])
    fw.upd_failwave_sh3n(fwave_el_adj, offly, dadv, elem_nodes_adj)
    assert fwave_el_adj[0] == 1  # Failure wave reached adjacent element!

    # Also test quad shell in Mode 1
    fw2 = Failwave(wave_mod=1, num_nodes=8)
    fwave_el_q = np.array([-1])
    nodes_q1 = np.array([[1, 2, 3, 4]])
    fw2.set_failwave_nod4(fwave_el_q, nodes_q1)
    fw2.update()

    fwave_el_q_adj = np.array([0])
    nodes_q2 = np.array([[3, 4, 5, 6]])
    fw2.upd_failwave_sh4n(fwave_el_q_adj, np.array([1]), np.array([1.0]), nodes_q2)
    assert fwave_el_q_adj[0] == 1


def test_failwave_mode2_directional_edges():
    """Verify Mode 2 directional failure wave propagation through quad edges."""
    fw = Failwave(wave_mod=2, num_nodes=8)

    # Cracked element with nodes 1,2,3,4, horizontal crack direction (cos=1, sin=0)
    # Direction 1 crack (-1) uses normal to crack (-crkdir[1], crkdir[0]) = (0, 1) -> vertical ray
    fwave_el = np.array([-1])
    nodes = np.array([[1, 2, 3, 4]])
    crkdir = np.array([[1.0, 0.0]])
    xl2 = np.array([1.0])
    xl3 = np.array([1.0])
    xl4 = np.array([0.0])
    yl2 = np.array([0.0])
    yl3 = np.array([1.0])
    yl4 = np.array([1.0])

    fw.set_failwave_nod4(fwave_el, nodes, crkdir=crkdir, xl2=xl2, xl3=xl3, xl4=xl4, yl2=yl2, yl3=yl3, yl4=yl4)
    fw.update()

    # Adjacent element sharing top edge N3-N4
    adj_nodes = np.array([[4, 3, 5, 6]])
    adj_el = np.array([0])
    fw.upd_failwave_sh4n(adj_el, np.array([1]), np.array([1.0]), adj_nodes)
    assert adj_el[0] == 1


# ---------------------------------------------------------------------------
# 2. Beam Failure Tests (Standard Beams TYPE 3)
# ---------------------------------------------------------------------------


def test_johnson_cook_beam_step():
    """Verify Johnson-Cook beam failure step."""
    fail = _FailStub("JOHNSON", D1=0.1, D2=0.5, D3=-1.5, eps_dot_0=1.0, eps_f_min=0.0)
    svm = np.array([100.0])
    pressure = np.array([33.3333])  # Uniaxial tension triaxiality ~ 1/3
    d_epsp = np.array([0.05])
    deps = np.array([0.05])
    dt = 1e-3
    dama = np.array([0.0])

    broken = beam_step(fail, svm, pressure, d_epsp, deps, dt, dama)
    assert broken[0] is False or broken[0] == (dama[0] >= 1.0)
    assert dama[0] > 0.0

    # Large plastic strain exceeding failure strain -> break
    d_epsp_large = np.array([1.0])
    broken2 = beam_step(fail, svm, pressure, d_epsp_large, deps, dt, dama)
    assert broken2[0] is True
    assert dama[0] == pytest.approx(1.0)


def test_biquad_beam_step():
    """Verify Bi-quadratic beam failure step."""
    fail = _FailStub("BIQUAD", c3=0.2, m_flag=1, s_flag=2)
    svm = np.array([200.0])
    pressure = np.array([66.6666])  # Uniaxial tension: triax = 1/3
    d_epsp = np.array([0.1])
    deps = np.array([0.1])
    dt = 1e-4
    dama = np.array([0.0])

    broken = beam_step(fail, svm, pressure, d_epsp, deps, dt, dama)
    assert broken[0] is False
    assert dama[0] == pytest.approx(0.1 / 0.2)  # eps_f at triax=1/3 is c3 = 0.2

    # Add remaining plastic strain -> D >= 1.0
    beam_step(fail, svm, pressure, np.array([0.15]), deps, dt, dama)
    assert dama[0] == pytest.approx(1.0)


def test_energy_beam_step():
    """Verify Specific Energy failure step for standard beams."""
    fail = _FailStub("ENERGY", e1=100.0, e2=200.0)
    svm = np.array([50.0])
    pressure = np.array([0.0])
    d_epsp = np.array([0.0])
    deps = np.array([0.0])
    dt = 1e-3
    dama = np.array([0.0])

    # Beam internal work per area = F1*eps_xx / area
    forces = np.array([[1000.0, 0.0, 0.0]])
    strains = np.array([[0.15, 0.0, 0.0]])  # work = 150.0
    area = 1.0

    broken = beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, forces=forces, strains=strains, area=area)
    assert broken[0] is False
    # (150 - 100) / (200 - 100) = 0.5
    assert dama[0] == pytest.approx(0.5)

    # Exceed failure energy 200.0
    strains_high = np.array([[0.25, 0.0, 0.0]])  # work = 250.0
    broken2 = beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, forces=forces, strains=strains_high, area=area)
    assert broken2[0] is True
    assert dama[0] == pytest.approx(1.0)


def test_tensstrain_beam_step():
    """Verify Tensile Strain failure step for standard beams."""
    fail = _FailStub("TENSSTRAIN", eps_t1=0.05, eps_t2=0.15)
    dama = np.array([0.0])
    strains = np.array([[0.10]])

    broken = beam_step(fail, None, None, np.array([0.10]), None, 1e-3, dama, strains=strains)
    assert broken[0] is False
    assert dama[0] == pytest.approx(0.5)

    strains_rupt = np.array([[0.20]])
    broken2 = beam_step(fail, None, None, np.array([0.20]), None, 1e-3, dama, strains=strains_rupt)
    assert broken2[0] is True
    assert dama[0] == pytest.approx(1.0)


def test_gene1_and_visual_beam_step():
    """Verify GENE1 and VISUAL beam failure steps."""
    # Gene1
    fail_g = _FailStub("GENE1", mat_sigvm=300.0, mat_maxeps=0.2, mat_ncs=1)
    dama_g = np.array([0.0])
    broken_g = beam_step(fail_g, np.array([350.0]), np.array([100.0]), np.array([0.05]), None, 1e-3, dama_g)
    assert broken_g[0] is True
    assert dama_g[0] == pytest.approx(1.0)

    # Visual (diagnostic only: never broken)
    fail_v = _FailStub("VISUAL", c_min=0.0, c_max=500.0)
    dama_v = np.array([0.0])
    broken_v = beam_step(fail_v, np.array([250.0]), np.array([0.0]), np.array([0.0]), None, 1e-3, dama_v)
    assert broken_v[0] is False
    assert dama_v[0] == pytest.approx(0.5)


def test_inievo_and_tab2_beam_step():
    """Verify INIEVO and TAB2 beam failure steps."""
    # Inievo
    crv = _CurveStub(x=[-1.0, 0.0, 1.0], y=[0.5, 0.3, 0.1])
    fail_ini = _FailStub("INIEVO", crv_ini=crv)
    dama_ini = np.array([0.0])
    broken_ini = beam_step(fail_ini, np.array([100.0]), np.array([0.0]), np.array([0.15]), None, 1e-3, dama_ini)
    assert broken_ini[0] is False
    assert dama_ini[0] == pytest.approx(0.5)

    # Tab2
    fail_tab = _FailStub("TAB2", fcrit=0.2, dcrit=1.0, n=1.0)
    dama_tab = np.array([0.0])
    broken_tab = beam_step(fail_tab, np.array([100.0]), np.array([0.0]), np.array([0.1]), None, 1e-3, dama_tab)
    assert broken_tab[0] is False
    assert dama_tab[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. Integrated Beam Failure Tests (TYPE 18)
# ---------------------------------------------------------------------------


def test_integrated_beam_steps():
    """Verify integrated beam failure steps across criteria."""
    # Johnson-Cook at integration point
    fail_jc = _FailStub("JOHNSON", D1=0.1, D2=0.5, D3=-1.5)
    sig = np.array([[150.0, 20.0, 10.0]])
    d_epsp = np.array([0.05])
    dama_jc = np.array([0.0])
    broken_jc = integrated_beam_step(fail_jc, sig, d_epsp, sig, 1e-3, dama_jc, ip=1, npg=4)
    assert broken_jc[0] is False
    assert dama_jc[0] > 0.0

    # Bi-quadratic at integration point
    fail_bq = _FailStub("BIQUAD", c3=0.2, m_flag=1)
    dama_bq = np.array([0.0])
    broken_bq = integrated_beam_step(fail_bq, sig, d_epsp, sig, 1e-3, dama_bq, ip=1, npg=4)
    assert broken_bq[0] is False
    assert dama_bq[0] > 0.0

    # Specific energy at integration point
    fail_en = _FailStub("ENERGY", e1=10.0, e2=30.0)
    deps = np.array([[0.1, 0.0, 0.0]])
    dama_en = np.array([0.0])
    broken_en = integrated_beam_step(fail_en, np.array([[200.0, 0.0, 0.0]]), d_epsp, deps, 1e-3, dama_en)
    # work = 200 * 0.1 = 20.0 -> (20 - 10) / (30 - 10) = 0.5
    assert broken_en[0] is False
    assert dama_en[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. Thick Shell FLD Failure Tests
# ---------------------------------------------------------------------------


def test_fld_thick_shell_step():
    """Verify thick shell FLD failure step (true strain, engineering strain, non-linear path)."""
    crv = _CurveStub(x=[-0.2, 0.0, 0.2], y=[0.4, 0.3, 0.5])
    fail = _FailStub("FLD", function=crv, istrain=0, fact_margin=0.1)

    # True strain input: eps_xx=0.15, eps_yy=0.0, eps_xy=0.0 -> emaj=0.15, emin=0.0
    deps = np.array([[0.15, 0.0, 0.0]])
    dama = np.array([0.0])
    broken = thick_shell_step(fail, None, np.array([0.15]), deps, 1e-3, dama)
    assert broken[0] is False
    # at emin=0.0, EM=0.3 -> dam = 0.15 / 0.3 = 0.5
    assert dama[0] == pytest.approx(0.5)
    assert hasattr(fail, "fld_zone")
    assert fail.fld_zone[0] in (1, 2, 3, 4, 5, 6)

    # Engineering strain input: istrain=1
    fail_eng = _FailStub("FLD", function=crv, istrain=1)
    dama_eng = np.array([0.0])
    thick_shell_step(fail_eng, None, np.array([0.15]), deps, 1e-3, dama_eng)
    assert dama_eng[0] > 0.0

    # Non-linear strain path: istrain=2
    fail_nl = _FailStub("FLD", function=crv, istrain=2)
    dama_nl = np.array([0.0])
    thick_shell_step(fail_nl, None, np.array([0.15]), deps, 1e-3, dama_nl, pla=np.array([0.15]))
    assert dama_nl[0] > 0.0


# ---------------------------------------------------------------------------
# 5. XFEM Crack Failure Tests
# ---------------------------------------------------------------------------


def test_xfem_step_johnson_and_fld():
    """Verify XFEM crack initiation and crack advancement logic."""
    # Johnson-Cook XFEM
    fail_jc = _FailStub("JOHNSON", D1=0.1, D2=0.0, D3=0.0)  # eps_f = 0.1
    sig = np.array([[100.0, 0.0, 0.0]])
    deps = np.array([[0.05, 0.0, 0.0]])
    d_epsp = np.array([0.15])  # Exceeds eps_f=0.1
    dt = 1e-3
    dama = np.array([0.0])
    elcrkini = np.array([0])  # Uncracked initially

    # Crack initiation test: elcrkini transitions 0 -> -1
    broken = xfem_step(fail_jc, sig, d_epsp, deps, dt, dama, elcrkini, dadv=0.8)
    assert broken[0] is True
    assert elcrkini[0] == -1  # Initiated!

    # Crack advancement test: element adjacent to crack (elcrkini = 2)
    elcrkini_adv = np.array([2])
    dama_adv = np.array([0.0])
    d_epsp_sub = np.array([0.085])  # damage = 0.085 / 0.1 = 0.85 >= dadv=0.8
    broken_adv = xfem_step(fail_jc, sig, d_epsp_sub, deps, dt, dama_adv, elcrkini_adv, dadv=0.8)
    assert broken_adv[0] is True
    assert elcrkini_adv[0] == 1  # Advanced!

    # Phantom element test
    dama_ph = np.array([0.0])
    broken_ph = xfem_step(fail_jc, sig, d_epsp, deps, dt, dama_ph, np.array([0]), is_phantom=True)
    assert broken_ph[0] is True
