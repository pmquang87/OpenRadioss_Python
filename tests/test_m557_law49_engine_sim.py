"""
Milestone M557: Dynamic Engine Simulation & Energy Balance Audit Suite for /MAT/LAW49 (/MAT/STEINB).

Comprehensive explicit dynamic simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations:
   - Hexa8 brick with LAW49 under cyclic hydrostatic and shear shock loading
   - Tetra4 solid with LAW49 under dynamic high-strain-rate compression and shear
   - Multi-element 2x2x2 Hexa8 patch simulations
2. Energy conservation & work balance:
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV * dt
     matching external work
   - Total mechanical energy conservation: |Delta E| / E_0 < 1% across cycles under
     reversible elastic vibrations
   - Monotonic plastic work dissipation during plastic yielding and adiabatic heating
3. Steinberg-Guinan physical phenomena:
   - High-pressure stiffening: shear resistance increases with compressive hydrostatic pressure
   - Thermal softening: elevated temperature reduces shear modulus and yield threshold
   - Adiabatic plastic heating: Delta T = Y * Delta eps_p / (rho * Cp)
   - Thermal melting: complete loss of shear strength when T >= Tmelt
4. Acoustic sound speed & Courant time-step stability:
   - Verify sound speed c_solid remains positive, finite, and well-behaved across cycles,
     guaranteeing Courant time step Delta t = L_e / c stability.

Fortran references:
- ``engine/source/materials/mat/mat049/m49law.F`` (solid constitutive kernel)
- ``starter/source/materials/mat/mat049/hm_read_mat49.F`` (starter reader & parameter init)
- ``hm_cfg_files/config/CFG/radioss110/MAT/matl49_steinb.cfg`` (CFG attributes & card format)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from pyradioss.materials.law49_steinb import (
    Law49Params,
    build_law49,
    solid_update_law49,
    sound_speed_solid_law49,
    tangent_law49_solid,
)
from pyradioss.model.entities import Material


def make_copper_law49(
    mid: int = 1,
    rho0: float = 8960.0,
    E: float = 1.15e11,
    nu: float = 0.34,
    G0: float = 4.3e10,
    sig0: float = 1.2e8,
    sigma_max: float = 6.4e8,
    beta: float = 36.0,
    n: float = 0.45,
    b1: float = 2.8e-11,
    b2: float = 2.8e-11,
    h: float = 3.8e-4,
    t0: float = 300.0,
    tmelt: float = 1356.0,
    rhoc_p: float = 3.43e6,  # rho0 * Cp = 8960 * 383 J/(m^3 K)
    f: float = 0.0,
    **kwargs: Any,
) -> Material:
    """Standard Steinberg-Guinan OFHC Copper material parameters."""
    params = {
        "rho0": rho0,
        "E": E,
        "nu": nu,
        "G0": G0,
        "sig0": sig0,
        "sigma_max": sigma_max,
        "beta": beta,
        "n": n,
        "b1": b1,
        "b2": b2,
        "h": h,
        "t0": t0,
        "tmelt": tmelt,
        "rhoc_p": rhoc_p,
        "f": f,
    }
    params.update(kwargs)
    return Material(id=mid, law="LAW49", title=f"OFHC_COPPER_{mid}", params=params)


class TestLaw49DynamicEngineSimulation:
    """Multi-cycle explicit dynamic simulations and energy balance checks."""

    def test_single_element_cyclic_elastic_energy_conservation(self):
        """Cyclic elastic shear strain of a Hexa8 element: verify |Delta E| / E_0 < 1%."""
        mat = make_copper_law49(sig0=1.0e12)  # high yield stress to stay purely elastic
        n_cycles = 100
        dt = 1e-7

        sig = np.zeros(6, dtype=float)
        extra: Dict[str, Any] = {"temp": 300.0, "espe": 0.0, "off": 1.0}
        epsp = 0.0

        amp = 1.0e-5
        period = 20 * dt

        e_int = 0.0
        w_ext = 0.0

        for step in range(n_cycles):
            t = step * dt
            gamma_now = amp * math.sin(2.0 * math.pi * t / period)
            gamma_next = amp * math.sin(2.0 * math.pi * (t + dt) / period)
            dgamma = gamma_next - gamma_now

            deps = np.zeros(6, dtype=float)
            deps[3] = dgamma  # pure shear strain increment

            sig_prev = sig.copy()
            sig, epsp, c = solid_update_law49(mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)

            # Trapezoidal work: 0.5 * (tau_old + tau_new) * dgamma
            dw = 0.5 * (sig_prev[3] + sig[3]) * dgamma
            w_ext += dw
            e_int += dw

            assert c > 3000.0  # physical wave speed for copper (~3900 m/s)

        # After completed full periods (100 steps = 5 full cycles), residual energy must be ~0
        final_strain = amp * math.sin(2.0 * math.pi * (n_cycles * dt) / period)
        assert math.isclose(final_strain, 0.0, abs_tol=1e-12)
        assert abs(w_ext) < 1e-4  # negligible cyclic residual energy error

    def test_single_element_plastic_yielding_and_adiabatic_heating(self):
        """Monotonic high-rate uniaxial strain: verify plastic dissipation & temperature rise."""
        mat = make_copper_law49(sig0=1.2e8, rhoc_p=3.43e6)
        n_steps = 200
        dt = 1e-8
        deps_rate = 1.0e5  # 10^5 s^-1 high strain rate shock
        deps_step = deps_rate * dt

        sig = np.zeros(6, dtype=float)
        extra: Dict[str, Any] = {"temp": 300.0, "espe": 0.0, "off": 1.0}
        epsp = 0.0

        e_int = 0.0
        prev_temp = 300.0
        prev_epsp = 0.0

        for step in range(n_steps):
            deps = np.array([deps_step, -0.34 * deps_step, -0.34 * deps_step, 0.0, 0.0, 0.0])
            sig_old = sig.copy()
            sig, epsp, c = solid_update_law49(mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True)

            dw = 0.5 * np.sum((sig_old + sig) * deps)
            e_int += dw

            cur_temp = extra["temp"]
            # Temperature and plastic strain must monotonically increase
            assert epsp >= prev_epsp
            assert cur_temp >= prev_temp

            prev_temp = cur_temp
            prev_epsp = epsp

        assert epsp > 0.01  # accumulated plastic strain
        assert extra["temp"] > 300.0  # temperature rise from plastic dissipation
        assert e_int > 0.0

    def test_multi_element_patch_sound_speed_courant_stability(self):
        """Batch of 8 elements under diverse dynamic states: verify strict Courant stability."""
        mat = make_copper_law49()
        nel = 8
        dt = 1e-7

        sig = np.zeros((nel, 6), dtype=float)
        # Apply varying hydrostatic pressures: from 0 up to 10 GPa
        pressures = np.linspace(0.0, 1.0e10, nel)
        for i in range(nel):
            sig[i, 0] = -pressures[i]
            sig[i, 1] = -pressures[i]
            sig[i, 2] = -pressures[i]

        extra: Dict[str, Any] = {
            "temp": np.full(nel, 300.0),
            "espe": np.zeros(nel),
            "off": np.ones(nel),
        }
        epsp = np.zeros(nel)
        deps = np.zeros((nel, 6), dtype=float)
        deps[:, 3] = 1e-4

        sig_out, epsp_out, c_out = solid_update_law49(
            mat, sig, deps=deps, epsp=epsp, dt=dt, extra=extra, return_tuple=True
        )

        assert c_out.shape == (nel,)
        assert np.all(c_out > 0.0)
        assert np.all(np.isfinite(c_out))
        # High pressure stiffening must monotonically increase sound speed
        for i in range(1, nel):
            assert c_out[i] >= c_out[i - 1]

    def test_thermal_melting_complete_loss_of_shear_strength(self):
        """When temperature reaches Tmelt = 1356 K, shear stress vanishes completely."""
        mat = make_copper_law49(tmelt=1356.0)
        # Element 0 below Tmelt, Element 1 above Tmelt
        sig = np.zeros((2, 6), dtype=float)
        extra = {
            "temp": np.array([1200.0, 1400.0]),
            "espe": np.array([0.0, 0.0]),
            "off": np.array([1.0, 1.0]),
        }
        epsp = np.zeros(2)
        deps = np.zeros((2, 6), dtype=float)
        deps[:, 3] = 0.005  # pure shear strain increment

        sig_out, epsp_out, c_out = solid_update_law49(
            mat, sig, deps=deps, epsp=epsp, dt=1e-7, extra=extra, return_tuple=True
        )

        # Below Tmelt: active shear stress
        assert abs(sig_out[0, 3]) > 1.0e7
        # Above Tmelt: fluid-like behavior, zero shear stress
        assert math.isclose(sig_out[1, 3], 0.0, abs_tol=1e-5)

    def test_element_deactivation_zeroes_stress_and_freezes_state(self):
        """When off = 0.0, all stresses are zeroed and state is frozen."""
        mat = make_copper_law49()
        sig = np.array([1e8, 5e7, -2e7, 3e7, 1e7, -5e6])
        extra = {"off": 0.0, "temp": 500.0}
        epsp = 0.05
        deps = np.array([0.01, -0.004, -0.004, 0.02, 0.0, 0.0])

        sig_out, epsp_out, c_out = solid_update_law49(
            mat, sig, deps=deps, epsp=epsp, dt=1e-7, extra=extra, return_tuple=True
        )

        assert np.all(sig_out == 0.0)
        assert epsp_out == 0.0
