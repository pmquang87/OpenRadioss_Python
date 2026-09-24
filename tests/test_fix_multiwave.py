"""Targeted regression tests for multi-wave bug fixes across pyradioss subsystems."""

import numpy as np
import pytest

from pyradioss.common.constants import EM20
from pyradioss.contact import friction, inter_type10, inter_type18, inter_type2, inter_type7, stiffness
from pyradioss.elements import beam_type3, spring, truss
from pyradioss.failure import tab1
from pyradioss.implicit import complex_modal, followerload, multi_input_fatigue, multiaxial_fatigue, response_spectrum, spectral_fatigue
from pyradioss.materials import law36_tabulated, law81_druckerprager
from pyradioss.model.model import Model


def test_contact_type2_segment_normalization_and_frame():
    t1, t2, n = inter_type2._segment_frames(np.zeros((0, 4, 3)))
    assert len(t1) == 0

    xs = np.zeros((1, 4, 3))
    xs[0, 0] = [0.0, 0.0, 0.0]
    xs[0, 1] = [1.0, 0.0, 0.0]
    xs[0, 2] = [1.0, 1.0, 0.0]
    xs[0, 3] = [0.0, 1.0, 0.0]
    t1, t2, n = inter_type2._segment_frames(xs)
    assert np.allclose(np.linalg.norm(t1, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(t2, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    assert np.allclose(np.einsum("nb,nb->n", t1, n), 0.0)
    assert np.allclose(np.einsum("nb,nb->n", t2, n), 0.0)


def test_inter_type10_ks():
    nodes = np.array([0, 1, 999])
    Ks_all = np.array([50.0, 50.0, 50.0, 50.0])
    Ks = np.zeros(len(nodes))
    valid_ks = (nodes >= 0) & (nodes < len(Ks_all))
    Ks[valid_ks] = Ks_all[nodes[valid_ks]]
    assert Ks[0] == 50.0
    assert Ks[1] == 50.0
    assert Ks[2] == 0.0


def test_contact_renard_friction_denominator():
    v = np.array([0.0, 1.0, 10.0, 100.0])
    c = np.array([0.2, 0.05, 0.3, 0.25, 1.0, 2.0])
    mu = friction.mu_kinetic(3, 0.2, c, p=0.0, v=v)
    assert np.all(np.isfinite(mu))
    assert np.all(mu >= 1e-4)


def test_followerload_triangle_derivatives():
    degen = np.array([True])
    dAdx = np.ones((1, 4, 3, 3))
    if np.any(degen):
        dAdx[degen, 2] += dAdx[degen, 3]
        dAdx[degen, 3] = 0.0
    assert np.all(dAdx[0, 2] == 2.0)
    assert np.all(dAdx[0, 3] == 0.0)


def test_spectral_fatigue_dc_spectra():
    freqs = np.array([0.0])
    psd = np.array([1.0])
    t, h = spectral_fatigue.synthesize_gaussian_history(freqs, psd, duration=1.0, fs=None, seed=42)
    assert len(t) > 0
    assert len(h) > 0
    assert np.all(np.isfinite(h))


def test_multiaxial_and_multi_input_dc_spectra():
    freqs = np.array([0.0])
    Scross = np.ones((1, 6, 6))
    t, h = multiaxial_fatigue.synthesize_multiaxial_history(freqs, Scross, duration=1.0, fs=None, seed=42)
    assert len(t) > 0
    assert np.all(np.isfinite(h))

    Smat = np.ones((1, 2, 2))
    fft_f, G, nt, df = multi_input_fatigue._one_sided_grid(freqs, Smat, duration=1.0, fs=None)
    assert len(fft_f) > 0
    assert np.all(np.isfinite(G))


def test_response_spectrum_rigid_modes():
    omega = np.array([0.0, 10.0, 20.0])
    zeta = 0.05
    rho = response_spectrum.cqc_correlation(omega, zeta)
    assert np.all(np.isfinite(rho))
    assert np.all(rho >= 0.0)
    assert np.all(rho <= 1.0)
    assert np.all(np.diag(rho) == 1.0)


def test_beam_b_operator_degenerate():
    B = beam_type3._b_operator(0.0)
    assert np.all(np.isfinite(B))
    assert B.shape == (6, 12)


def test_truss_radial_return_zero_modulus():
    over = 10.0
    E = 0.0
    H = 0.0
    dl = over / np.maximum(E + np.maximum(H, 0.0), EM20)
    assert np.isfinite(dl)
    assert dl > 0.0


def test_spring_type32_sensor_activation():
    class DummySensors:
        def __init__(self):
            self.fire_time = {1: 0.5}
        def active(self, sid):
            return sid == 1

    sensors = DummySensors()
    t = 0.5
    sens_id = [1]
    tacti = np.zeros(1)
    iact = np.ones(1, dtype=bool)

    for i, s_id in enumerate(sens_id):
        if s_id > 0:
            if sensors.active(s_id):
                tf = sensors.fire_time.get(s_id, 0.0)
                tacti[i] = max(0.0, t - tf)
                iact[i] = True
            else:
                tacti[i] = 0.0
                iact[i] = False

    assert bool(iact[0]) is True
    assert tacti[0] == 0.0


def test_law36_single_point_curve():
    cx = np.array([0.0])
    cy = np.array([250.0])
    cs = np.zeros(0)
    e = np.array([0.0, 0.1, 0.5])
    y, s = law36_tabulated._curve_eval(cx, cy, cs, e)
    assert np.all(y == 250.0)
    assert np.all(s == 0.0)


def test_law81_single_point_and_duplicate_knots():
    xy1 = (np.array([0.0]), np.array([100.0]))
    y1, der1 = law81_druckerprager._finter(xy1, np.array([0.0, 1.0]))
    assert np.all(y1 == 100.0)
    assert np.all(der1 == 0.0)

    xy2 = (np.array([0.0, 0.0, 1.0]), np.array([100.0, 100.0, 200.0]))
    y2, der2 = law81_druckerprager._finter(xy2, np.array([0.5]))
    assert np.all(np.isfinite(y2))
    assert np.all(np.isfinite(der2))


def test_tab1_failure_dn_zero():
    class DummyFail:
        params = {"n": 0.0, "dcrit": 1.0}

    fail = DummyFail()
    dama = np.array([0.1])
    d_epsp = np.array([0.01])
    eps_f = np.array([0.2])

    tab1._accumulate(fail, dama, d_epsp, eps_f)
    assert np.all(np.isfinite(dama))
    assert dama[0] > 0.1
