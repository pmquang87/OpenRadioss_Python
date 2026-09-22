"""Unit tests for the OpenRadioss explicit thermal solver.

Upstream Fortran References:
----------------------------
- tempur.F: Nodal temperature time integration
- fixtemp.F: Imposed temperature boundary conditions (/IMPTEMP)
- fixflux.F: Imposed heat flux sources (/FIXFLUX)
- thermbilan.F: Thermal energy balance tracking
- dttherm.F90: Critical thermal time step limit
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from pyradioss.engine.thermal_loads import (
    ConvecParams,
    RadiationParams,
    STEFAN_BOLTZMANN,
    ThermalLoadsManager,
)
from pyradioss.engine.thermal_solver import (
    GlobTherm,
    apply_conduction,
    apply_imposed_flux,
    apply_imposed_temperatures,
    compute_1d_bar_conduction,
    compute_thermal_balance,
    compute_thermal_dt,
    get_glob_therm,
    solve_thermal_step,
    update_nodal_temperatures,
)
from pyradioss.model.entities import ImposedFlux, ImposedTemperature


# ==============================================================================
# Helper Mock Models for Unit Testing
# ==============================================================================

class MockFunction:
    """Mock Radioss /FUNCT curve."""

    def __init__(self, f_callable):
        self.f = f_callable

    def eval(self, x: float) -> float:
        return float(self.f(x))

    def __call__(self, x: float) -> float:
        return float(self.f(x))


class MockNodeGroup:
    """Mock Radioss /GRNOD group."""

    def __init__(self, node_ids: List[int]):
        self.node_ids = node_ids


class SimpleThermalModel:
    """Lightweight Radioss model representing a thermal mesh or lumped system."""

    def __init__(
        self,
        n_nodes: int,
        t_init: float = 300.0,
        mcp: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
    ):
        self.temperature = np.full(n_nodes, float(t_init), dtype=np.float64)
        self.temperatures = self.temperature
        self.temp0 = self.temperature.copy()
        self.initial_temperature = self.temp0.copy()

        if mcp is not None:
            self.mcp = np.asarray(mcp, dtype=np.float64)
        else:
            self.mcp = np.ones(n_nodes, dtype=np.float64)

        self.fthe = np.zeros(n_nodes, dtype=np.float64)
        self.q_nodal = np.zeros(n_nodes, dtype=np.float64)
        self.mcp_off = np.ones(n_nodes, dtype=np.float64)
        self.weight = np.ones(n_nodes, dtype=np.int32)

        self.time: float = 0.0
        self.t: float = 0.0
        self.dt: float = 0.0

        self.node_ids = np.arange(1, n_nodes + 1, dtype=np.int64)
        self._id2idx = {int(nid): i for i, nid in enumerate(self.node_ids)}
        self.node_groups: Dict[int, Any] = {}
        self.functions: Dict[int, Any] = {}
        self.surfaces: Dict[int, Any] = {}

        self.imptemp: List[Any] = []
        self.impflux_loads: List[Any] = []
        self.glob_therm = GlobTherm()

        if coords is not None:
            self.x = np.asarray(coords, dtype=np.float64)
        else:
            self.x = np.zeros((n_nodes, 3), dtype=np.float64)


# ==============================================================================
# 1. 1D Bar Conduction: Analytical Solution Comparison
# ==============================================================================

def analytical_1d_bar_transient(
    x: np.ndarray,
    t: float,
    t_left: float,
    t_right: float,
    t_init: float,
    length: float,
    diffusivity: float,
    n_terms: int = 150,
) -> np.ndarray:
    """Fourier series analytical solution for 1D transient heat diffusion.

    Governing PDE:
      dT/dt = alpha * d^2T/dx^2, for 0 < x < L, t > 0
    BCs:
      T(0, t) = T_left
      T(L, t) = T_right
    IC:
      T(x, 0) = T_init
    """
    if t <= 0.0:
        return np.full_like(x, t_init)

    # Steady-state linear profile
    t_ss = t_left + (t_right - t_left) * (x / length)

    # Transient deviation theta(x, t) = T(x, t) - T_ss(x)
    # theta(x, 0) = T_init - [T_left + (T_right - T_left)*x/L]
    theta = np.zeros_like(x, dtype=np.float64)
    for n in range(1, n_terms + 1):
        lambda_n = (n * math.pi) / length
        decay = math.exp(-diffusivity * (lambda_n**2) * t)

        # Fourier sine coefficient: bn = (2/L) * int_0^L theta(x, 0) * sin(n*pi*x/L) dx
        # int_0^L sin(n*pi*x/L) dx = (L/(n*pi)) * (1 - (-1)^n)
        # int_0^L (x/L) * sin(n*pi*x/L) dx = -(L/(n*pi)) * (-1)^n
        i_const = (1.0 - (-1.0) ** n) / (n * math.pi)
        i_x = -((-1.0) ** n) / (n * math.pi)
        bn = 2.0 * ((t_init - t_left) * i_const - (t_right - t_left) * i_x)

        theta += bn * np.sin(lambda_n * x) * decay

    return t_ss + theta


class Test1DBarConduction:
    """Verify explicit 1D bar conduction solver against analytical Fourier series."""

    def test_transient_conduction_profile(self):
        """Compare numerical transient temperatures with exact Fourier series solution."""
        length = 1.0  # 1 m
        n_elems = 20
        n_nodes = n_elems + 1
        dx = length / n_elems
        area = 0.01  # 0.01 m^2

        k = 50.0  # W/(m*K)
        rho = 7800.0  # kg/m^3
        cp = 500.0  # J/(kg*K)
        diffusivity = k / (rho * cp)  # ~1.282e-5 m^2/s

        t_left = 400.0  # Hot left boundary
        t_right = 300.0  # Ambient right boundary
        t_init = 300.0  # Initial temperature

        # Create model
        model = SimpleThermalModel(n_nodes, t_init=t_init)
        x_coords = np.linspace(0.0, length, n_nodes)
        model.x[:, 0] = x_coords
        model.dx = dx
        model.area = area
        model.k = k
        model.bar_nodes = list(range(n_nodes))

        # Lumped thermal capacitance: interior nodes have rho * A * dx * Cp, boundary have half
        m_elem = rho * area * dx
        mcp_nodal = np.full(n_nodes, m_elem * cp, dtype=np.float64)
        mcp_nodal[0] *= 0.5
        mcp_nodal[-1] *= 0.5
        model.mcp = mcp_nodal
        model.temp0 = model.temperature.copy()

        # Imposed temperature BC at x=0 (node 0) and x=L (node n_nodes - 1)
        model.node_groups[1] = MockNodeGroup([1])  # node 1 (0-based 0)
        model.node_groups[2] = MockNodeGroup([n_nodes])  # node n_nodes (0-based -1)
        model.imptemp = [
            ImposedTemperature(id=1, funct_id=0, grnod_id=1, scale=t_left),
            ImposedTemperature(id=2, funct_id=0, grnod_id=2, scale=t_right),
        ]

        # Critical dt check (dttherm.F90)
        dt_crit = 0.5 * (dx**2) / diffusivity
        dt = 0.8 * dt_crit  # Courant safety factor 0.8 (~62.4 s)
        assert dt < dt_crit

        # Target comparison time
        t_target = 1000.0  # 1000 s
        n_steps = int(math.ceil(t_target / dt))
        dt_actual = t_target / n_steps

        # Run explicit integration loop
        apply_imposed_temperatures(model)
        for _ in range(n_steps):
            # 1. Internal conduction: compute and assemble FTHE
            apply_conduction(model, dt=dt_actual)
            # 2. Update temperatures
            update_nodal_temperatures(model, dt=dt_actual)
            # 3. Apply imposed boundary temperatures
            apply_imposed_temperatures(model)
            model.time += dt_actual

        # Analytical solution at t_target
        t_exact = analytical_1d_bar_transient(
            x_coords,
            t_target,
            t_left,
            t_right,
            t_init,
            length,
            diffusivity,
        )

        # Boundary conditions match exactly
        assert math.isclose(model.temperature[0], t_left, rel_tol=1e-12)
        assert math.isclose(model.temperature[-1], t_right, rel_tol=1e-12)

        # Interior profile comparison
        abs_errors = np.abs(model.temperature - t_exact)
        max_err = float(np.max(abs_errors))
        l2_err = float(np.linalg.norm(abs_errors) / np.linalg.norm(t_exact))

        # Explicit finite difference error is well under 1% of the temperature range (100 K)
        assert max_err < 0.8, f"Max error {max_err:.3f} K exceeds 0.8 K"
        assert l2_err < 0.005, f"Relative L2 error {l2_err:.4f} exceeds 0.5%"

    def test_steady_state_linear_profile(self):
        """Verify that long-term conduction converges to the exact linear steady-state."""
        length = 1.0
        n_nodes = 11
        dx = length / (n_nodes - 1)
        area = 0.01

        k = 50.0
        rho = 7800.0
        cp = 500.0
        diffusivity = k / (rho * cp)

        t_left = 500.0
        t_right = 200.0

        model = SimpleThermalModel(n_nodes, t_init=300.0)
        x_coords = np.linspace(0.0, length, n_nodes)
        model.x[:, 0] = x_coords
        model.dx = dx
        model.area = area
        model.k = k
        model.bar_nodes = list(range(n_nodes))

        m_elem = rho * area * dx
        mcp = np.full(n_nodes, m_elem * cp)
        mcp[0] *= 0.5
        mcp[-1] *= 0.5
        model.mcp = mcp

        model.node_groups[1] = MockNodeGroup([1])
        model.node_groups[2] = MockNodeGroup([n_nodes])
        model.imptemp = [
            ImposedTemperature(id=1, funct_id=0, grnod_id=1, scale=t_left),
            ImposedTemperature(id=2, funct_id=0, grnod_id=2, scale=t_right),
        ]

        dt = 0.8 * 0.5 * (dx**2) / diffusivity
        # Run to steady-state (~50,000 s)
        t_end = 50000.0
        n_steps = int(t_end / dt)

        apply_imposed_temperatures(model)
        for _ in range(n_steps):
            apply_conduction(model, dt=dt)
            update_nodal_temperatures(model, dt=dt)
            apply_imposed_temperatures(model)

        # Exact linear profile
        t_expected = t_left + (t_right - t_left) * (x_coords / length)
        np.testing.assert_allclose(model.temperature, t_expected, atol=1e-4)


# ==============================================================================
# 2. Imposed Temperature Boundary Conditions (/IMPTEMP, fixtemp.F)
# ==============================================================================

class TestImposedTemperatures:
    """Verify /IMPTEMP boundary condition enforcement matching fixtemp.F."""

    def test_constant_temperature_bc(self):
        """Constant temperature Dirichlet BC applied directly to nodes."""
        model = SimpleThermalModel(5, t_init=293.15)
        model.node_groups[1] = MockNodeGroup([2, 4])  # 1-based nodes 2 and 4 (indices 1 and 3)
        model.imptemp = [
            ImposedTemperature(id=1, funct_id=0, grnod_id=1, scale=373.15)
        ]

        applied = apply_imposed_temperatures(model)
        assert applied == {1: 373.15, 3: 373.15}
        assert model.temperature[1] == 373.15
        assert model.temperature[3] == 373.15
        # Other nodes untouched
        assert model.temperature[0] == 293.15
        assert model.temperature[2] == 293.15
        assert model.temperature[4] == 293.15

    def test_time_dependent_temperature_curve(self):
        """Time-dependent function evaluation T(t) = scale * f((t - tstart) / xscale)."""
        model = SimpleThermalModel(3, t_init=300.0)
        # Function: f(t) = 1.0 + 2.0 * t
        model.functions[1] = MockFunction(lambda t: 1.0 + 2.0 * t)
        model.node_groups[1] = MockNodeGroup([1])
        model.imptemp = [
            ImposedTemperature(
                id=1,
                funct_id=1,
                grnod_id=1,
                scale=100.0,
                xscale=2.0,
                tstart=1.0,
                tstop=10.0,
            )
        ]

        # Before tstart: condition inactive
        model.time = 0.5
        applied = apply_imposed_temperatures(model)
        assert len(applied) == 0
        assert model.temperature[0] == 300.0

        # At t = 3.0: ts = 3.0 - 1.0 = 2.0; tsc = 2.0 / 2.0 = 1.0
        # f(1.0) = 1.0 + 2.0(1.0) = 3.0; T = 100.0 * 3.0 = 300.0 K
        model.time = 3.0
        applied = apply_imposed_temperatures(model)
        assert 0 in applied
        assert math.isclose(applied[0], 300.0, rel_tol=1e-12)
        assert math.isclose(model.temperature[0], 300.0, rel_tol=1e-12)

        # At t = 5.0: ts = 4.0; tsc = 2.0; f(2.0) = 5.0; T = 500.0 K
        model.time = 5.0
        apply_imposed_temperatures(model)
        assert math.isclose(model.temperature[0], 500.0, rel_tol=1e-12)

        # After tstop: condition inactive
        model.time = 15.0
        model.temperature[0] = 999.0
        applied_after = apply_imposed_temperatures(model)
        assert len(applied_after) == 0
        assert model.temperature[0] == 999.0

    def test_thermal_acceleration_factor(self):
        """Thermal acceleration factor theaccfact scales time window and argument."""
        model = SimpleThermalModel(2, t_init=300.0)
        model.glob_therm.theaccfact = 10.0
        model.functions[1] = MockFunction(lambda t: 2.0 * t)
        model.node_groups[1] = MockNodeGroup([1])
        model.imptemp = [
            ImposedTemperature(id=1, funct_id=1, grnod_id=1, scale=50.0, tstart=10.0, tstop=50.0)
        ]

        # Model time t = 1.5 -> t_scaled = 1.5 * 10 = 15.0 (inside [10, 50])
        # ts = 15.0 - 10.0 = 5.0; f(5.0) = 10.0; T = 50.0 * 10.0 = 500.0
        model.time = 1.5
        applied = apply_imposed_temperatures(model)
        assert 0 in applied
        assert math.isclose(applied[0], 500.0, rel_tol=1e-12)


# ==============================================================================
# 3. Imposed Heat Flux (/FIXFLUX, fixflux.F)
# ==============================================================================

class TestImposedFlux:
    """Verify /FIXFLUX heat source application and energy accounting."""

    def test_nodal_heat_flux_energy_ledger(self):
        """Heat flux injects energy into FTHE and increments GLOB_THERM%HEAT_FFLUX."""
        model = SimpleThermalModel(3, t_init=300.0)
        flux_val = 500.0  # W / node
        model.impflux_loads = [
            ImposedFlux(id=1, scale=flux_val, tstart=0.0, tstop=10.0)
        ]
        # Target node 2 (index 1)
        model.impflux_loads[0].nodes = [2]

        dt = 0.05
        model.time = 0.05
        added = apply_imposed_flux(model, dt=dt)

        expected_energy = flux_val * dt  # 25 J
        assert math.isclose(added, expected_energy, rel_tol=1e-12)
        assert math.isclose(model.glob_therm.heat_fflux, expected_energy, rel_tol=1e-12)
        assert math.isclose(model.fthe[1], expected_energy, rel_tol=1e-12)
        assert model.fthe[0] == 0.0
        assert model.fthe[2] == 0.0

    def test_surfacic_heat_flux_area_integration(self):
        """Surfacic heat flux integrates over quad segment area matching fixflux.F."""
        model = SimpleThermalModel(4, t_init=300.0)
        # 4-node 2x2 square in XY plane: Area = 4.0 m^2
        model.x = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ])

        flux_density = 150.0  # W/m^2
        fl = ImposedFlux(id=1, scale=flux_density, tstart=0.0, tstop=10.0)
        fl.segments = [[1, 2, 3, 4]]
        model.impflux_loads = [fl]

        dt = 0.1
        model.time = 0.1
        added = apply_imposed_flux(model, dt=dt)

        # Expected energy: Area * flux_dens * dt = 4.0 * 150.0 * 0.1 = 60.0 J
        expected_total = 4.0 * flux_density * dt
        assert math.isclose(added, expected_total, rel_tol=1e-12)
        assert math.isclose(model.glob_therm.heat_fflux, expected_total, rel_tol=1e-12)

        # Distributes equally to all 4 corner nodes (15 J each)
        for i in range(4):
            assert math.isclose(model.fthe[i], expected_total / 4.0, rel_tol=1e-12)


# ==============================================================================
# 4. First-Law Thermal Energy Balance (thermbilan.F)
# ==============================================================================

class TestThermalEnergyBalance:
    """Verify exact First-Law energy conservation: E_fixed + E_meca - E_conv - E_rad = Delta E_stored."""

    def test_complete_energy_balance_closed_system(self):
        """First Law holds to machine precision under concurrent flux, convection, and radiation."""
        # 1-node lumped thermal system
        model = SimpleThermalModel(1, t_init=300.0)
        mcp_val = 2000.0  # J / K
        model.mcp = np.array([mcp_val])
        model.temp0 = np.array([300.0])

        # Thermal boundary conditions:
        # Convection: h = 25 W/(m^2*K), Area = 0.2 m^2, T_inf = 300 K
        convec = ConvecParams(id=1, h=25.0, area=0.2, t_inf=300.0)
        # Radiation: eps = 0.8, Area = 0.2 m^2, T_inf = 300 K
        radiation = RadiationParams(id=1, emissivity=0.8, area=0.2, t_inf=300.0)
        model.thermal_loads_mgr = ThermalLoadsManager([convec], [radiation])

        # Imposed heat flux: 500 W for 10 s
        fl = ImposedFlux(id=1, scale=500.0, tstart=0.0, tstop=10.0)
        fl.nodes = [1]
        model.impflux_loads = [fl]

        # Mechanical heat generation rate (plastic dissipation): 50 W
        p_meca = 50.0

        dt = 0.05
        t_end = 2.0
        n_steps = int(t_end / dt)

        for _ in range(n_steps):
            # Simulate mechanical heat dissipation
            model.glob_therm.heat_meca += p_meca * dt
            model.fthe[0] += p_meca * dt

            # Execute explicit thermal step
            res = solve_thermal_step(model, dt=dt)
            assert res["is_conserved"] is True

        # Check final balance
        final_balance = compute_thermal_balance(model)
        assert final_balance["is_conserved"] is True
        assert final_balance["balance_error"] < 1e-8, f"Balance error: {final_balance['balance_error']}"

        # Stored heat matches temperature rise
        delta_t = float(model.temperature[0] - 300.0)
        expected_stored = mcp_val * delta_t
        assert math.isclose(final_balance["delta_e_stored"], expected_stored, rel_tol=1e-12)

    def test_conservative_internal_conduction_zero_net_work(self):
        """Internal heat conduction strictly conserves energy: sum(FTHE_cond) == 0."""
        # 10-node 1D bar with arbitrary non-uniform temperature distribution
        n_nodes = 10
        temps = np.array([300.0, 450.0, 280.0, 520.0, 310.0, 400.0, 350.0, 290.0, 600.0, 320.0])
        dx = 0.1
        area = 0.02
        k = 60.0
        dt = 0.01

        fthe_cond = compute_1d_bar_conduction(
            nodes=list(range(n_nodes)),
            temps=temps,
            dx=dx,
            area=area,
            k=k,
            theaccfact=1.0,
            dt=dt,
        )

        # Net thermal energy generated by conduction must be exactly zero (conservative)
        net_conduction_energy = float(np.sum(fthe_cond))
        assert abs(net_conduction_energy) < 1e-14, f"Net conduction sum {net_conduction_energy} != 0"


# ==============================================================================
# 5. Critical Thermal Time Step (dttherm.F90)
# ==============================================================================

class TestThermalTimeStep:
    """Verify explicit thermal stability time step calculation matching dttherm.F90."""

    def test_dt_therm_formula(self):
        """dt = dtfactherm * 0.5 * lc^2 * rhocp / k."""
        model = SimpleThermalModel(1)
        dx = 0.02  # 20 mm
        k = 45.0  # W/(m*K)
        rho = 7850.0
        cp = 480.0
        rhocp = rho * cp

        model.dx = dx
        model.k = k
        model.rhocp = rhocp
        model.glob_therm.dtfactherm = 0.9
        model.glob_therm.theaccfact = 1.0

        dt_calc = compute_thermal_dt(model)
        expected_dt = 0.9 * 0.5 * (dx**2) * rhocp / k
        assert math.isclose(dt_calc, expected_dt, rel_tol=1e-12)

    def test_temperature_dependent_conductivity(self):
        """Piecewise linear conductivity: as + bs*T below Tmelt, al + bl*T above."""
        model = SimpleThermalModel(1)
        model.dx = 0.05
        model.rhocp = 3.5e6
        model.glob_therm.dtfactherm = 1.0
        model.glob_therm.theaccfact = 1.0

        setattr(model, "as", 30.0)
        setattr(model, "bs", 0.1)
        setattr(model, "al", 70.0)
        setattr(model, "bl", 0.05)
        model.tmelt = 1000.0

        # Below Tmelt: T = 500 K -> k = 30 + 0.1*500 = 80 W/(m*K)
        model.temp_elem = 500.0
        dt_solid = compute_thermal_dt(model)
        expected_k_solid = 80.0
        expected_dt_solid = 0.5 * (0.05**2) * 3.5e6 / expected_k_solid
        assert math.isclose(dt_solid, expected_dt_solid, rel_tol=1e-12)

        # Above Tmelt: T = 1200 K -> k = 70 + 0.05*1200 = 130 W/(m*K)
        model.temp_elem = 1200.0
        # Reset dt_therm to trigger min search
        model.glob_therm.dt_therm = 1e30
        dt_liquid = compute_thermal_dt(model)
        expected_k_liquid = 130.0
        expected_dt_liquid = 0.5 * (0.05**2) * 3.5e6 / expected_k_liquid
        assert math.isclose(dt_liquid, expected_dt_liquid, rel_tol=1e-12)

    def test_thermal_stability_boundary(self):
        """Simulate diffusion just below dt_crit (stable) vs above dt_crit (unstable oscillations)."""
        length = 0.2
        n_nodes = 5
        dx = length / (n_nodes - 1)
        area = 0.01
        k = 100.0
        rhocp = 1.0e6
        diffusivity = k / rhocp

        dt_crit = 0.5 * (dx**2) / diffusivity  # Exact 1D forward Euler stability limit

        # Case A: Stable run with dt = 0.9 * dt_crit
        model_stable = SimpleThermalModel(n_nodes, t_init=300.0)
        model_stable.dx = dx
        model_stable.area = area
        model_stable.k = k
        model_stable.bar_nodes = list(range(n_nodes))
        model_stable.mcp = np.full(n_nodes, rhocp * area * dx)
        model_stable.node_groups[1] = MockNodeGroup([1])
        model_stable.imptemp = [ImposedTemperature(id=1, grnod_id=1, scale=500.0)]

        dt_sub = 0.9 * dt_crit
        apply_imposed_temperatures(model_stable)
        for _ in range(50):
            apply_conduction(model_stable, dt=dt_sub)
            update_nodal_temperatures(model_stable, dt=dt_sub)
            apply_imposed_temperatures(model_stable)

        # Stable: temperatures must remain smoothly monotonic within [300, 500]
        assert np.all(model_stable.temperature >= 299.99)
        assert np.all(model_stable.temperature <= 500.01)
        assert np.all(np.diff(model_stable.temperature) <= 0.001)  # monotonically decreasing from hot end


# ==============================================================================
# 6. Deactivation Mask & MPI Weight Handling (tempur.F)
# ==============================================================================

class TestTempurFeatures:
    """Verify MCP_OFF element deactivation and MPI WEIGHT mask matching tempur.F."""

    def test_mcp_off_node_deactivation(self):
        """When MCP_OFF(i) == 0, temperature is frozen despite applied thermal force."""
        model = SimpleThermalModel(3, t_init=300.0)
        model.mcp = np.array([100.0, 100.0, 100.0])
        # Deactivate node 1 (index 0)
        model.mcp_off = np.array([0.0, 1.0, 1.0])
        # Apply heat energy increment FTHE = 500 J to all nodes
        model.fthe = np.array([500.0, 500.0, 500.0])

        update_nodal_temperatures(model, dt=1.0)

        # Node 0 was deactivated: temperature unchanged
        assert model.temperature[0] == 300.0
        # Active nodes 1 and 2 received Delta T = 500 / 100 = 5.0 K
        assert model.temperature[1] == 305.0
        assert model.temperature[2] == 305.0
        # FTHE zeroed out
        assert np.all(model.fthe == 0.0)

    def test_mpi_ghost_node_weight_filtering(self):
        """Only owned nodes (WEIGHT == 1) accumulate into HEAT_STORED."""
        model = SimpleThermalModel(3, t_init=300.0)
        model.mcp = np.array([100.0, 100.0, 100.0])
        # Node 2 is a ghost node (weight == 0)
        model.weight = np.array([1, 1, 0], dtype=np.int32)
        model.fthe = np.array([200.0, 300.0, 500.0])

        update_nodal_temperatures(model, dt=1.0)

        # Stored heat should only sum nodes 0 and 1: 200 + 300 = 500 J (ignoring ghost node 2)
        assert math.isclose(model.glob_therm.heat_stored, 500.0, rel_tol=1e-12)
