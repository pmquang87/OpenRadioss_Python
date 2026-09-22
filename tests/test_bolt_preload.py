"""
Unit tests for Bolt Pretension Load (/PRELOAD, /PRELOAD/BOLT, /PRELOAD/AXIAL).

Fortran origins:
  - ``starter/source/loads/bolt/iniboltprel.F``
  - ``starter/source/loads/bolt/sboltini.F``
  - ``starter/source/loads/bolt/sectarea.F``
  - ``starter/source/loads/general/preload/hm_read_preload.F``
  - ``engine/source/elements/spring/preload_axial.F90``
  - ``engine/source/elements/solid/solide/boltst.F``
  - ``engine/source/elements/solid/solide/sboltlaw.F``
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.engine.bolt_preload import (
    BoltPreloadEngine,
    BoltPreloadParams,
    PreloadMethod,
    PreloadPhase,
    build_bolt_preloads,
)
from pyradioss.model.entities import NodeGroup, Preload, PreloadAxial, PreloadBolt, Section
from pyradioss.model.model import Model


def test_bolt_preload_ramp_and_hold():
    """Verify linear force ramp F(t) = F0 * min(1, t/t_ramp), hold phase, and equal-opposite forces."""
    f0 = 50000.0  # 50 kN
    t_ramp = 0.001
    t_hold = 0.003
    k_bolt = 1.0e8  # 100 kN/mm

    params = BoltPreloadParams(
        id=1,
        title="M12_Bolt",
        idx1=0,
        idx2=1,
        f0=f0,
        t_ramp=t_ramp,
        t_hold=t_hold,
        k_bolt=k_bolt,
        axis=(1.0, 0.0, 0.0),
        method=PreloadMethod.SPRING_ELEMENT,
    )
    engine = BoltPreloadEngine(params)

    # Initial nodal coordinates: node 0 at x=0, node 1 at x=0.05 m
    x = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64)

    # 1. Ramp Phase (t = 0.5 * t_ramp)
    t = 0.0005
    fint = np.zeros_like(x)
    f_applied = engine.apply_preload(t=t, dt=1e-5, fint=fint, x=x)

    expected_f = f0 * (t / t_ramp)
    assert f_applied == pytest.approx(expected_f, rel=1e-5)
    assert engine.phase == PreloadPhase.RAMP
    # Equal and opposite forces on endpoints
    np.testing.assert_allclose(fint[0] + fint[1], [0.0, 0.0, 0.0], atol=1e-10)
    assert fint[0, 0] == pytest.approx(expected_f, rel=1e-5)
    assert fint[1, 0] == pytest.approx(-expected_f, rel=1e-5)

    # 2. End of Ramp (t = t_ramp)
    t = 0.001
    fint = np.zeros_like(x)
    f_applied = engine.apply_preload(t=t, dt=1e-5, fint=fint, x=x)
    assert f_applied == pytest.approx(f0, rel=1e-5)
    assert engine.phase == PreloadPhase.HOLD

    # 3. Holding Phase (t_ramp < t <= t_hold)
    t = 0.002
    fint = np.zeros_like(x)
    f_applied = engine.apply_preload(t=t, dt=1e-5, fint=fint, x=x)
    assert f_applied == pytest.approx(f0, rel=1e-5)
    assert engine.phase == PreloadPhase.HOLD
    np.testing.assert_allclose(fint[0] + fint[1], [0.0, 0.0, 0.0], atol=1e-10)

    # 4. Locked Phase transition (t > t_hold)
    t = 0.0035
    fint = np.zeros_like(x)
    f_applied = engine.apply_preload(t=t, dt=1e-5, fint=fint, x=x)
    assert engine.phase == PreloadPhase.LOCKED
    assert engine.is_locked
    assert f_applied == pytest.approx(f0, rel=1e-5)
    np.testing.assert_allclose(fint[0] + fint[1], [0.0, 0.0, 0.0], atol=1e-10)


def test_bolt_preload_clamped_equilibrium():
    """Verify static equilibrium between bolt tension and compressed structural plates."""
    f0 = 100000.0  # 100 kN
    k_bolt = 5.0e7  # 50 kN/mm
    k_plates = 2.0e8  # 200 kN/mm
    t_ramp = 0.001
    t_hold = 0.002
    l0 = 0.10  # 100 mm bolt length

    params = BoltPreloadParams(
        id=2,
        title="Flange_Clamped_Bolt",
        idx1=0,
        idx2=1,
        f0=f0,
        t_ramp=t_ramp,
        t_hold=t_hold,
        k_bolt=k_bolt,
        l0=l0,
        axis=(1.0, 0.0, 0.0),
        method=PreloadMethod.SPRING_ELEMENT,
    )
    engine = BoltPreloadEngine(params)

    # Under preload F0, the clamped plates compress by delta_x_p = F0 / k_plates
    delta_x_p = f0 / k_plates  # 0.5 mm = 0.0005 m
    # Node 0 fixed at x=0; node 1 compressed inwards to x = l0 - delta_x_p
    x_equil = np.array([[0.0, 0.0, 0.0], [l0 - delta_x_p, 0.0, 0.0]], dtype=np.float64)

    # Apply preload at holding phase (t = t_hold)
    fint = np.zeros_like(x_equil)
    f_bolt = engine.apply_preload(t=t_hold, dt=1e-5, fint=fint, x=x_equil)

    # Bolt tension force matches target F0
    assert f_bolt == pytest.approx(f0, rel=1e-5)

    # Plate compressive reaction force
    f_plates = k_plates * delta_x_p  # 100 kN
    assert f_plates == pytest.approx(f0, rel=1e-5)

    # Net force on the clamped interface (node 1) in equilibrium:
    # fint[1, 0] is -f_bolt (-100 kN), plate reaction on interface is +f_plates (+100 kN)
    net_interface_force = -fint[1, 0] - f_plates
    assert net_interface_force == pytest.approx(0.0, abs=1e-6)

    # Verify shortening displacement delta_L
    # delta_L = f0 / k_bolt - elongation = f0 / k_bolt - (-delta_x_p) = f0/k_bolt + f0/k_plates
    expected_delta_L = f0 / k_bolt + delta_x_p
    assert engine.delta_L == pytest.approx(expected_delta_L, rel=1e-5)


def test_bolt_preload_lock_phase_stiffness_ratio():
    """Verify locked phase joint load sharing under external tensile pull (VDI 2230).

    Delta F_bolt = P_ext * [K_bolt / (K_bolt + K_plates)]
    Delta F_plates = -P_ext * [K_plates / (K_bolt + K_plates)]
    """
    f0 = 50000.0  # 50 kN preload
    k_bolt = 2.0e8  # 200 kN/mm
    k_plates = 8.0e8  # 800 kN/mm
    t_ramp = 0.001
    t_hold = 0.002
    l0 = 0.10

    # Joint stiffness ratio Phi = K_bolt / (K_bolt + K_plates)
    phi = k_bolt / (k_bolt + k_plates)
    assert phi == pytest.approx(0.20)

    params = BoltPreloadParams(
        id=3,
        title="VDI2230_Joint",
        idx1=0,
        idx2=1,
        f0=f0,
        t_ramp=t_ramp,
        t_hold=t_hold,
        k_bolt=k_bolt,
        l0=l0,
        axis=(1.0, 0.0, 0.0),
        method=PreloadMethod.SPRING_ELEMENT,
    )
    engine = BoltPreloadEngine(params)

    # Step 1: Preload to holding phase at t = t_hold
    delta_x_p = f0 / k_plates
    x_preloaded = np.array([[0.0, 0.0, 0.0], [l0 - delta_x_p, 0.0, 0.0]], dtype=np.float64)
    fint = np.zeros_like(x_preloaded)
    engine.apply_preload(t=t_hold, dt=1e-5, fint=fint, x=x_preloaded)
    assert engine.phase == PreloadPhase.HOLD
    assert engine.f_current == pytest.approx(f0, rel=1e-5)

    # Step 2: Lock the bolt at t > t_hold
    t_locked = 0.0025
    fint = np.zeros_like(x_preloaded)
    engine.apply_preload(t=t_locked, dt=1e-5, fint=fint, x=x_preloaded)
    assert engine.phase == PreloadPhase.LOCKED
    assert engine.is_locked
    assert engine.f_current == pytest.approx(f0, rel=1e-5)

    # Step 3: Apply external tensile pull P_ext = 30 kN separating the joint
    p_ext = 30000.0
    delta_joint = p_ext / (k_bolt + k_plates)  # 30,000 / 1.0e9 = 3.0e-5 m

    # Displaced joint geometry under external load
    x_loaded = np.array([[0.0, 0.0, 0.0], [l0 - delta_x_p + delta_joint, 0.0, 0.0]], dtype=np.float64)

    fint = np.zeros_like(x_loaded)
    f_bolt_loaded = engine.apply_preload(t=0.005, dt=1e-5, fint=fint, x=x_loaded)

    # Theoretical additional bolt force: Delta F_bolt = P_ext * Phi
    expected_delta_f_bolt = p_ext * phi  # 30,000 * 0.20 = 6,000 N
    expected_f_bolt = f0 + expected_delta_f_bolt  # 56,000 N
    assert f_bolt_loaded == pytest.approx(expected_f_bolt, rel=1e-5)

    # Remaining clamping force in plates: F_plates = F0 - P_ext * (1 - Phi)
    remaining_plate_clamping = k_plates * (delta_x_p - delta_joint)
    expected_plate_clamping = f0 - p_ext * (1.0 - phi)  # 50,000 - 24,000 = 26,000 N
    assert remaining_plate_clamping == pytest.approx(expected_plate_clamping, rel=1e-5)

    # Joint equilibrium under external load: F_bolt - F_plates = P_ext
    assert (f_bolt_loaded - remaining_plate_clamping) == pytest.approx(p_ext, rel=1e-5)


def test_bolt_preload_section_cut_stress():
    """Verify section cut preload force F0 = sigma0 * Area and multi-node internal force distribution."""
    sigma0 = 200.0e6  # 200 MPa preload stress
    area = 5.0e-4  # 500 mm^2 cross-sectional cut area
    expected_f0 = sigma0 * area  # 100,000 N = 100 kN

    # 4 nodes on side 1 (indices 0..3), 4 nodes on side 2 (indices 4..7)
    side1 = [0, 1, 2, 3]
    side2 = [4, 5, 6, 7]
    axis = (0.0, 0.0, 1.0)  # Along Z

    params = BoltPreloadParams(
        id=4,
        title="3D_Solid_Bolt_Cut",
        side1_indices=side1,
        side2_indices=side2,
        sigma0=sigma0,
        area=area,
        t_ramp=0.001,
        t_hold=0.002,
        axis=axis,
        method=PreloadMethod.SECTION_CUT,
    )
    assert params.get_target_force() == pytest.approx(expected_f0, rel=1e-5)

    engine = BoltPreloadEngine(params)

    # Coordinate array for 8 nodes
    x = np.zeros((8, 3), dtype=np.float64)
    fint = np.zeros_like(x)

    # Apply preload at holding phase (t = 0.0015 s)
    f_applied = engine.apply_preload(t=0.0015, dt=1e-5, fint=fint, x=x)
    assert f_applied == pytest.approx(expected_f0, rel=1e-5)

    # Each side 1 node receives +f0 / 4 along Z
    expected_side1_nodal_force = np.array([0.0, 0.0, expected_f0 / 4.0])
    for idx in side1:
        np.testing.assert_allclose(fint[idx], expected_side1_nodal_force, rtol=1e-5)

    # Each side 2 node receives -f0 / 4 along Z
    expected_side2_nodal_force = np.array([0.0, 0.0, -expected_f0 / 4.0])
    for idx in side2:
        np.testing.assert_allclose(fint[idx], expected_side2_nodal_force, rtol=1e-5)

    # Total internal force sum over all cut nodes must be exactly zero (conservation of linear momentum)
    total_fint = np.sum(fint, axis=0)
    np.testing.assert_allclose(total_fint, [0.0, 0.0, 0.0], atol=1e-10)

    # Diagnostic status dictionary
    status = engine.get_status()
    assert status["id"] == 4
    assert status["phase"] == "HOLD"
    assert status["f_current"] == pytest.approx(expected_f0, rel=1e-5)
    assert not status["is_locked"]


def test_build_bolt_preloads_from_model():
    """Verify model builder parses /PRELOAD, /PRELOAD/BOLT, and /PRELOAD/AXIAL."""
    model = Model()

    # 1. /PRELOAD/BOLT with section link
    model.preload_bolts[1] = PreloadBolt(
        id=1,
        title="Bolt_1_Sect",
        sect_id=10,
        preload=45000.0,
        tstart=0.001,
        tstop=0.002,
    )
    # Section 10 with node group 101
    sec10 = Section(id=10, grnod_id=101, title="Cut_Section_10")
    model.sections.append(sec10)
    grp101 = NodeGroup(id=101, node_ids=[10, 11, 12, 13])
    model.node_groups[101] = grp101

    # 2. /PRELOAD with stress preload (itype=2)
    model.preloads[2] = Preload(
        id=2,
        title="Preload_Stress",
        sect_id=20,
        itype=2,
        preload=2.5e8,  # 250 MPa
        tstart=0.0005,
        tstop=0.0015,
    )

    # 3. /PRELOAD/AXIAL 1D spring preload
    model.preload_axials[3] = PreloadAxial(
        id=3,
        title="Axial_Spring_Preload",
        set_id=30,
        preload=15000.0,
    )

    # Map node IDs
    for i, nid in enumerate([10, 11, 12, 13]):
        model._id2idx[nid] = i

    engines = build_bolt_preloads(model)
    assert len(engines) == 3

    # Check Bolt 1
    eng1 = next(e for e in engines if e.id == 1)
    assert eng1.params.method == PreloadMethod.SECTION_CUT
    assert eng1.f0 == pytest.approx(45000.0)
    assert eng1.t_ramp == pytest.approx(0.001)
    assert eng1.t_hold == pytest.approx(0.002)
    assert eng1.side1_indices == [0, 1, 2, 3]

    # Check Bolt 2
    eng2 = next(e for e in engines if e.id == 2)
    assert eng2.params.method == PreloadMethod.SECTION_CUT
    assert eng2.params.sigma0 == pytest.approx(2.5e8)
    assert eng2.t_ramp == pytest.approx(0.0005)
    assert eng2.t_hold == pytest.approx(0.0015)

    # Check Bolt 3
    eng3 = next(e for e in engines if e.id == 3)
    assert eng3.params.method == PreloadMethod.SPRING_ELEMENT
    assert eng3.f0 == pytest.approx(15000.0)
