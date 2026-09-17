"""Tests for LAW24 concrete constitutive model and BUG-MAT-06 verification.

Fortran reference: m24law.F, conc24.F, plas24.F, carm24.F.
Verifies:
- Cap ellipse division guard when rob == rok (BUG-MAT-06)
- Cyclic plastic rebar hardening monotonic accumulation (BUG-MAT-06)
- Solid update integration with 1D steel reinforcement
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.materials.law24_concrete import (
    _carm24,
    _plas24_one,
    build_conc,
    solid_update,
)
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
from types import SimpleNamespace


def test_law24_cap_ellipse_division_guard():
    """BUG-MAT-06: division guard when rob == rok in cap ellipse calculation."""
    params = {
        "MAT_E": 30000.0, "MAT_NU": 0.2, "RHO0": 2400.0, "Iflag": 0,
        "MAT_SIGY": 30.0, "MAT_FtFc": 0.1, "MAT_FbFc": 1.15, "MAT_F2Fc": 4.0, "MAT_SoFc": 1.25,
        "MAT_ETAN": 0.0, "MAT_DAMAGE": 0.99, "MAT_EPS": 1e20, "MAT_BETA": 0.5,
        "MAT_PPRES": 0.0, "MAT_YPRES": 0.0, "MAT_BPMOD": 0.0, "MAT_ETC": 0.0,
        "MAT_DIL_Y": -0.2, "MAT_DIL_F": -0.1, "MAT_COMPAC": -0.35,
        "MAT_CAP_BEG": -10.0, "MAT_CAP_END": -24.0, "MAT_TPMOD": 6000.0,
    }
    rec = SimpleNamespace(id=1, title="mat", density=2400.0, params=params)
    mat = build_conc(rec)
    p = mat.params

    sigc = np.array([-15.0, -15.0, -15.0, 0.0, 0.0, 0.0])
    dam = np.zeros(3)
    crak = np.zeros(3)
    eps6 = np.zeros(6)

    vk0_a = np.array(0.5)
    vk_a = np.array(0.5)
    rob_a = np.array(p["RO0"])
    eint = 0.0
    rho = 2400.0

    _plas24_one(
        p, sigc, dam, crak, eps6, scle2=1.0,
        vk0_a=vk0_a, vk_a=vk_a, rob_a=rob_a,
        eint=eint, rho=rho
    )
    assert np.all(np.isfinite(sigc))


def test_law24_cyclic_plastic_rebar_hardening():
    """BUG-MAT-06: epsa += np.abs(d_eps_plas) ensures monotonic plastic accumulation under cyclic loading."""
    yms = 200000.0
    y0s = 400.0
    ets = 2000.0
    epsa = np.zeros(1)
    siga = np.zeros(1)

    deps_pos = np.array([0.005])
    _carm24(yms, y0s, ets, epsa, siga, deps_pos)
    epsa_after_step1 = float(epsa[0])
    assert epsa_after_step1 > 0.0
    assert siga[0] > 400.0

    deps_neg = np.array([-0.010])
    _carm24(yms, y0s, ets, epsa, siga, deps_neg)
    epsa_after_step2 = float(epsa[0])

    assert epsa_after_step2 > epsa_after_step1


def test_law24_solid_update_integration():
    """Verify solid_update runs end-to-end with concrete and reinforcement."""
    params = {
        "MAT_E": 30000.0, "MAT_NU": 0.2, "RHO0": 2400.0, "Iflag": 0,
        "MAT_SIGY": 30.0, "MAT_FtFc": 3.0, "MAT_FbFc": 36.0, "MAT_F2Fc": 120.0, "MAT_SoFc": 1.25,
        "MAT_ETAN": -30000.0, "MAT_DAMAGE": 0.99, "MAT_EPS": 1e20, "MAT_BETA": 0.5,
        "MAT_PPRES": 10.0, "MAT_YPRES": -10.0, "MAT_BPMOD": 0.0, "MAT_ETC": 0.0,
        "MAT_DIL_Y": -0.2, "MAT_DIL_F": -0.1, "MAT_COMPAC": -0.35,
        "MAT_CAP_BEG": -10.0, "MAT_CAP_END": -24.0, "MAT_TPMOD": 6000.0,
        "MAT_E2": 210000.0, "MAT_SSIG": 500.0, "MAT_SETAN": 2100.0,
        "MAT_PDIR1": 0.1, "MAT_PDIR2": 0.0, "MAT_PDIR3": 0.0
    }
    rec = SimpleNamespace(id=1, title="mat", density=2400.0, params=params)
    mat = MAT_PHYSICS_REGISTRY["LAW24"](rec)

    nip = 1
    sig = np.zeros((nip, 6))
    deps = np.zeros((nip, 6))
    deps[0, 0] = 0.005

    extra = {
        "strain24": np.zeros((nip, 6)),
        "sigc24": np.zeros((nip, 6)),
        "crak24": np.zeros((nip, 3)),
        "dam24": np.zeros((nip, 3)),
        "ang24": np.zeros((nip, 6)),
        "epsf24": np.full((nip, 3), -1.0),
        "siga24": np.zeros((nip, 3)),
        "epsa24": np.zeros((nip, 3)),
        "vk024": np.zeros(nip),
        "vk24": np.zeros(nip),
        "rob24": np.zeros(nip),
        "off24": np.ones(nip),
        "ini24": np.zeros(nip)
    }

    sig_out, epsp_out, c_out = solid_update(mat, sig, deps, None, dt=1e-5, extra=extra)
    assert np.all(np.isfinite(sig_out))
    assert sig_out[0, 0] > 0.0
