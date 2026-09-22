"""tests.test_airbag_advanced — Unit tests for advanced airbag deployment physics.

Covers:
1. 3D unstructured mesh generation & dynamic rezoning (fvmesh.F, fvrezone.F).
2. Point-in-polyhedron spherical solid angle test (PINPOLH from fvrezone.F).
3. Implicit coupled monitored volume solver and tangent stiffness (monv_imp0.F).
4. Shock-capturing upwind flux limiter across moving/porous faces (fv_up_switch.F).
5. Artificial gas bulk viscosity and Courant time step (fv_up_switch.F).
6. Membrane anti-hourglass damping for fabric shells (mhvis3.F).
7. Liquid fluid control volume with logarithmic bulk compression (volp_lfluid.F).
8. Tabulated and biaxial strain fabric porosity models (porfor4.F, porfor6.F).
"""

import math
import numpy as np
import pytest

from pyradioss.model.model import Model
from pyradioss.model.entities import MonvolLFluid
from pyradioss.engine.airbag_mesh import (
    polygon_clip_plane,
    clip_triangle_to_box,
    compute_polygon_area_normal,
    point_in_polyhedron,
    compute_polyhedron_volume,
    generate_fvm_airbag_mesh,
    rezone_airbag_mesh,
    FvmPolyhedron,
)
from pyradioss.engine.airbag_implicit import (
    compute_implicit_pressure_increment,
    compute_airbag_tangent_stiffness,
    apply_airbag_matvec,
    solve_implicit_airbag_step,
)
from pyradioss.engine.airbag_fvm import (
    compute_upwind_face_flux,
    compute_gas_viscosity_and_cfl,
    compute_membrane_hourglass_viscosity,
)
from pyradioss.engine.airbag import (
    update_monvol_liquid_fluid,
    compute_porosity_porfor4,
    compute_porosity_porfor6,
)


# Helper: construct a unit cube surface triangulation (12 triangles)
def make_unit_cube_triangles(origin=(0.0, 0.0, 0.0), size=1.0):
    ox, oy, oz = origin
    s = size
    # 8 vertices
    p = [
        np.array([ox, oy, oz]),
        np.array([ox + s, oy, oz]),
        np.array([ox + s, oy + s, oz]),
        np.array([ox, oy + s, oz]),
        np.array([ox, oy, oz + s]),
        np.array([ox + s, oy, oz + s]),
        np.array([ox + s, oy + s, oz + s]),
        np.array([ox, oy + s, oz + s]),
    ]
    # 12 triangles (outward normals)
    tris = [
        # -z face
        [p[0], p[2], p[1]], [p[0], p[3], p[2]],
        # +z face
        [p[4], p[5], p[6]], [p[4], p[6], p[7]],
        # -y face
        [p[0], p[1], p[5]], [p[0], p[5], p[4]],
        # +y face
        [p[3], p[6], p[2]], [p[3], p[7], p[6]],
        # -x face
        [p[0], p[4], p[7]], [p[0], p[7], p[3]],
        # +x face
        [p[1], p[2], p[6]], [p[1], p[6], p[5]],
    ]
    return np.array(tris, dtype=np.float64)


# =============================================================================
# 1. 3D Mesh Generation & Sutherland-Hodgman Polygon Clipping (fvmesh.F)
# =============================================================================

class TestAirbagMeshClipping:
    def test_polygon_clip_plane(self):
        """Test clipping a square against a half-space plane (POLCLIP)."""
        square = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        plane_pt = np.array([1.0, 0.0, 0.0])
        plane_norm = np.array([-1.0, 0.0, 0.0])  # keep x <= 1.0

        clipped = polygon_clip_plane(square, plane_pt, plane_norm)
        assert len(clipped) == 4
        # Max x should be <= 1.0
        assert np.max(clipped[:, 0]) == pytest.approx(1.0, abs=1e-12)

    def test_clip_triangle_to_box(self):
        """Test clipping a 3D triangle against a bounding box (ITRIBOX)."""
        tri = np.array([
            [-1.0, 0.5, 0.5],
            [2.0, 0.5, 0.5],
            [0.5, 2.0, 0.5],
        ])
        box_min = np.array([0.0, 0.0, 0.0])
        box_max = np.array([1.0, 1.0, 1.0])

        clipped = clip_triangle_to_box(tri, box_min, box_max)
        assert len(clipped) >= 3
        assert np.all(clipped >= box_min - 1e-12)
        assert np.all(clipped <= box_max + 1e-12)

    def test_compute_polygon_area_normal(self):
        """Test area and normal calculation for planar 3D polygon."""
        rect = np.array([
            [0.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [3.0, 2.0, 1.0],
            [0.0, 2.0, 1.0],
        ])
        area, normal, cent = compute_polygon_area_normal(rect)
        assert area == pytest.approx(6.0, abs=1e-10)
        assert normal[2] == pytest.approx(1.0, abs=1e-10)
        assert cent[0] == pytest.approx(1.5, abs=1e-10)
        assert cent[1] == pytest.approx(1.0, abs=1e-10)

    def test_compute_polyhedron_volume(self):
        """Test divergence theorem volume computation for unit cube."""
        cube_tris = make_unit_cube_triangles(origin=(0.0, 0.0, 0.0), size=2.0)
        vol, cent = compute_polyhedron_volume(cube_tris)
        assert vol == pytest.approx(8.0, abs=1e-9)
        assert cent[0] == pytest.approx(1.0, abs=1e-9)
        assert cent[1] == pytest.approx(1.0, abs=1e-9)
        assert cent[2] == pytest.approx(1.0, abs=1e-9)


# =============================================================================
# 2. Point-in-Polyhedron & Dynamic Rezoning (fvrezone.F)
# =============================================================================

class TestPointInPolyhedronAndRezoning:
    def test_point_in_polyhedron(self):
        """Test spherical solid angle point-in-polyhedron test (PINPOLH)."""
        cube_tris = make_unit_cube_triangles(origin=(0.0, 0.0, 0.0), size=1.0)
        bbox_min = np.array([0.0, 0.0, 0.0])
        bbox_max = np.array([1.0, 1.0, 1.0])

        # Center of cube should be strictly inside
        assert point_in_polyhedron(np.array([0.5, 0.5, 0.5]), cube_tris, bbox_min=bbox_min, bbox_max=bbox_max)
        assert point_in_polyhedron(np.array([0.1, 0.1, 0.1]), cube_tris, bbox_min=bbox_min, bbox_max=bbox_max)

        # Outside points should be False
        assert not point_in_polyhedron(np.array([1.5, 0.5, 0.5]), cube_tris, bbox_min=bbox_min, bbox_max=bbox_max)
        assert not point_in_polyhedron(np.array([-0.1, 0.5, 0.5]), cube_tris, bbox_min=bbox_min, bbox_max=bbox_max)
        assert not point_in_polyhedron(np.array([0.5, 0.5, 2.0]), cube_tris, bbox_min=bbox_min, bbox_max=bbox_max)

    def test_generate_fvm_airbag_mesh(self):
        """Test generating 3D unstructured mesh inside cube boundary (FVMESH1)."""
        cube_tris = make_unit_cube_triangles(origin=(0.0, 0.0, 0.0), size=1.0)
        polyhedra = generate_fvm_airbag_mesh(cube_tris, grid_res=(2, 2, 2))
        assert len(polyhedra) >= 1
        for p in polyhedra:
            assert p.volume > 0.0

    def test_rezone_airbag_mesh(self):
        """Test conservative state variable remapping during dynamic rezoning (FVREZONE1)."""
        # Create an old single-block cell
        tris_old = make_unit_cube_triangles(origin=(0.0, 0.0, 0.0), size=1.0)
        vol_old, cent_old = compute_polyhedron_volume(tris_old)
        p_old = FvmPolyhedron(
            id=1,
            triangles=tris_old,
            volume=vol_old,
            centroid=cent_old,
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([1.0, 1.0, 1.0]),
            mass=1.2,
            momentum=np.array([0.12, 0.0, 0.0]),
            energy=2.5e5,
            gamma=1.4,
            r_spec=287.05,
            cpa=1004.0,
        )

        # Create 8 octants spanning the full [0, 1]^3 cube
        new_cells = []
        cell_id = 1
        for ox in (0.0, 0.5):
            for oy in (0.0, 0.5):
                for oz in (0.0, 0.5):
                    tris = make_unit_cube_triangles(origin=(ox, oy, oz), size=0.5)
                    vol, cent = compute_polyhedron_volume(tris)
                    p_cell = FvmPolyhedron(
                        id=cell_id,
                        triangles=tris,
                        volume=vol,
                        centroid=cent,
                        bbox_min=np.array([ox, oy, oz]),
                        bbox_max=np.array([ox + 0.5, oy + 0.5, oz + 0.5]),
                        gamma=1.4,
                    )
                    new_cells.append(p_cell)
                    cell_id += 1

        result = rezone_airbag_mesh([p_old], new_cells, n_sample_steps=4)
        assert "mass_initial" in result
        assert "mass_rezoned" in result
        assert result["mass_initial"] == pytest.approx(1.2, abs=1e-9)
        assert result["mass_rezoned"] == pytest.approx(1.2, abs=1e-9)
        assert result["energy_rezoned"] == pytest.approx(2.5e5, abs=1e-5)
        for cell in new_cells:
            assert cell.mass > 0.0


# =============================================================================
# 3. Implicit Airbag Coupled Solver & Tangent Stiffness (monv_imp0.F)
# =============================================================================

class TestAirbagImplicitSolver:
    def test_compute_implicit_pressure_increment(self):
        """Test IMP_PVGA pressure increment calculation."""
        v_old = 1.0
        v_new = 0.9  # 10% compression -> pressure should increase
        gamma = 1.4
        e_old = 2.0e5
        # Consistent thermodynamic initial pressure: P = (gamma - 1) * E / V = 80000 Pa
        p_old = (gamma - 1.0) * e_old / v_old

        dp, p_new, e_new = compute_implicit_pressure_increment(
            p_old=p_old,
            e_old=e_old,
            v_old=v_old,
            v_new=v_new,
            gamma=gamma,
        )
        assert dp > 0.0
        assert p_new > p_old
        assert e_new > 0.0

        # Burst pressure check
        dp_burst, p_burst, _ = compute_implicit_pressure_increment(
            p_old=p_old,
            e_old=e_old,
            v_old=v_old,
            v_new=0.1,
            gamma=gamma,
            p_max=2.0e5,
            p_ext=1.0e5,
        )
        assert p_burst == pytest.approx(1.0e5, abs=1e-6)

    def test_compute_airbag_tangent_stiffness(self):
        """Test MONV_KD diagonal tangent stiffness assembly."""
        model = Model()
        # Single triangular facet
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)

        class DummySurface:
            segments = [[0, 1, 2]]

        model.surfaces = {1: DummySurface()}

        class DummyMV:
            surf_id = 1
            volume = 0.1
            vinc = 0.0
            pressure = 2.0e5
            gamma = 1.4
            pext = 1.0e5

        k_diag, a_node = compute_airbag_tangent_stiffness(DummyMV(), model, x)
        assert k_diag.shape == (3, 3)
        assert a_node.shape == (3, 3)
        # Normal is in +z direction
        assert a_node[0, 2] == pytest.approx(0.5 / 3.0, abs=1e-9)
        # Stiffness is positive in z
        assert np.all(k_diag[:, 2] > 0.0)

    def test_apply_airbag_matvec(self):
        """Test MV_MATV matrix-vector directional derivative consistency."""
        model = Model()
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)

        class DummySurface:
            segments = [[0, 1, 2]]

        model.surfaces = {1: DummySurface()}

        class DummyMV:
            surf_id = 1
            volume = 1.0
            volume_old = 1.0
            pressure = 2.0e5
            energy = 3.0e5
            gamma = 1.4
            vinc = 0.0
            pmax = 1e30
            pext = 1.0e5
            de_out = 0.0

        u = np.zeros((3, 3), dtype=np.float64)
        u[:, 2] = -0.01  # compressive displacement

        f_tan = apply_airbag_matvec(DummyMV(), model, x, u, dt=1e-3)
        assert f_tan.shape == (3, 3)

    def test_solve_implicit_airbag_step(self):
        """Test coupled Newton-Raphson equilibrium solver."""
        model = Model()
        x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)

        class DummySurface:
            segments = [[0, 1, 2]]

        model.surfaces = {1: DummySurface()}

        class DummyMV:
            surf_id = 1
            volume = 1.0
            volume_old = 1.0
            pressure = 1.0e5
            energy = 2.0e5
            gamma = 1.4
            vinc = 0.0
            pmax = 1e30
            pext = 1.0e5
            de_out = 0.0

        f_structural = np.zeros((3, 3), dtype=np.float64)
        f_structural[:, 2] = -100.0  # 100 N external load
        k_structural = np.ones((3, 3), dtype=np.float64) * 1.0e5

        u, converged, n_iters = solve_implicit_airbag_step(
            mv=DummyMV(),
            model=model,
            x=x,
            f_structural=f_structural,
            k_structural_diag=k_structural,
            dt=1e-3,
            max_iter=10,
        )
        assert converged
        assert n_iters <= 10
        assert u.shape == (3, 3)


# =============================================================================
# 4. Shock-Capturing Upwind Flux & Artificial Gas Viscosity (fv_up_switch.F, mhvis3.F)
# =============================================================================

class TestFvmAdvancedPhysics:
    def test_compute_upwind_face_flux_positive(self):
        """Test upwind flux limiter when gas flows 1 -> 2 (ss_ > 0)."""
        normal = np.array([1.0, 0.0, 0.0])
        u1 = np.array([10.0, 0.0, 0.0])
        u2 = np.array([2.0, 0.0, 0.0])
        cp_poly = (1004.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        mass_flux, mom_flux, ener_flux, cp_flux, rgas_flux = compute_upwind_face_flux(
            rho1=2.0,
            re1=1.5e5,
            u1=u1,
            gamma1=1.4,
            cp_poly1=cp_poly,
            r_spec1=287.0,
            rho2=1.0,
            re2=1.0e5,
            u2=u2,
            gamma2=1.4,
            cp_poly2=cp_poly,
            r_spec2=287.0,
            normal=normal,
            area=0.5,
            v_grid=np.zeros(3),
            porosity=1.0,
        )
        # v_rel = 6.0 m/s > 0 -> upwind state is cell 1
        assert mass_flux > 0.0
        assert mom_flux[0] > 0.0
        assert ener_flux > 0.0
        # Flow carries cell 1 density (2.0)
        expected_mass_flow = 2.0 * 6.0 * 0.5
        assert mass_flux == pytest.approx(expected_mass_flow, rel=1e-6)

    def test_compute_upwind_face_flux_negative(self):
        """Test upwind flux limiter when gas flows 2 -> 1 (ss_ < 0)."""
        normal = np.array([1.0, 0.0, 0.0])
        u1 = np.array([-10.0, 0.0, 0.0])
        u2 = np.array([-2.0, 0.0, 0.0])
        cp_poly = (1004.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        mass_flux, mom_flux, ener_flux, cp_flux, rgas_flux = compute_upwind_face_flux(
            rho1=2.0,
            re1=1.5e5,
            u1=u1,
            gamma1=1.4,
            cp_poly1=cp_poly,
            r_spec1=287.0,
            rho2=1.0,
            re2=1.0e5,
            u2=u2,
            gamma2=1.4,
            cp_poly2=cp_poly,
            r_spec2=287.0,
            normal=normal,
            area=0.5,
            v_grid=np.zeros(3),
            porosity=1.0,
        )
        # v_rel = -6.0 m/s < 0 -> upwind state is cell 2 (rho=1.0)
        expected_mass_flow = 1.0 * (-6.0) * 0.5
        assert mass_flux == pytest.approx(expected_mass_flow, rel=1e-6)

    def test_compute_gas_viscosity_and_cfl(self):
        """Test artificial gas bulk viscosity QVISC and CFL time step bound."""
        q_visc, ssp_eff, dt_cfl = compute_gas_viscosity_and_cfl(
            volume=0.01,
            mass=0.012,
            energy=3000.0,
            gamma=1.4,
            dm=0.001,
            dt=1e-4,
            u_gas=np.array([50.0, 0.0, 0.0]),
            char_length=0.1,
            qa=1.5,
            qb=0.06,
            cfl_coef=0.9,
        )
        assert q_visc > 0.0
        assert ssp_eff > 300.0  # speed of sound + damping shift
        assert 0.0 < dt_cfl < 1e-2

    def test_compute_membrane_hourglass_viscosity(self):
        """Test anti-hourglass damping for 4-node fabric membrane quad (mhvis3.F)."""
        nodes_v = np.array([
            [1.0, 0.0, 0.5],
            [-1.0, 0.0, -0.5],
            [1.0, 0.0, 0.5],
            [-1.0, 0.0, -0.5],
        ], dtype=np.float64)

        f_hour, e_hour = compute_membrane_hourglass_viscosity(
            nodes_v=nodes_v,
            thk0=0.001,
            rho=1000.0,
            area=0.01,
            sound_speed=1500.0,
            dt=1e-5,
            h4=0.1,
            hvisc=1.0,
        )
        assert f_hour.shape == (4, 3)
        # Damping opposes motion -> dissipated energy must be >= 0
        assert e_hour >= 0.0


# =============================================================================
# 5. Liquid Fluid & Extended Porosity Models (volp_lfluid.F, porfor4.F, porfor6.F)
# =============================================================================

class TestLiquidFluidAndPorosity:
    def test_update_monvol_liquid_fluid(self):
        """Test /MONVOL/LFLUID logarithmic bulk compression and work accounting (volp_lfluid.F)."""
        model = Model()
        mv = MonvolLFluid(
            id=1,
            title="LiquidBladder",
            rho_fluid=1000.0,
            fscale_k=2.2e7,  # Bulk modulus K = 22 MPa
            fscale_padd=1.0e5,  # P0 = 1 bar
            fscale_pmax=1.0e8,
        )
        mv.volume = 0.01
        mv.mass = 10.0  # 10 kg / 1000 kg/m^3 = 0.01 m^3 uncompressed volume
        mv.work = 0.0

        # Step 1: initial uncompressed state
        update_monvol_liquid_fluid(mv, model, dt=1e-3, current_time=0.0)
        assert mv.pressure == pytest.approx(1.0e5, rel=1e-4)
        assert mv.work == 0.0

        # Step 2: compress volume by 5% (vol = 0.0095 m^3) -> pressure should rise
        mv.volume = 0.0095
        update_monvol_liquid_fluid(mv, model, dt=1e-3, current_time=1e-3)
        assert mv.pressure > 1.0e5
        # Expected: K * ln(0.01 / 0.0095) + P0
        expected_p = 2.2e7 * math.log(0.01 / 0.0095) + 1.0e5
        assert mv.pressure == pytest.approx(expected_p, rel=1e-4)
        # External volume change work dW = 0.5*(P + Pold)*dV must be booked
        assert mv.work != 0.0

    def test_compute_porosity_porfor4(self):
        """Test tabulated pressure-drop / area stretch porosity (porfor4.F)."""
        # Function: stretch factor = 1 + 0.5*(RS - 1), pressure factor = 1.0 - 0.2*RP
        func_area = lambda rs: 1.0 + 0.5 * (rs - 1.0)
        func_pres = lambda rp: 1.0 - 0.2 * rp

        svtfac = compute_porosity_porfor4(
            p=2.0e5,
            pext=1.0e5,
            area=1.2,
            area0=1.0,
            fpora=1.0,
            fporp=1.0,
            func_area=func_area,
            func_pres=func_pres,
        )
        # RS = 1.2, RP = 0.5 -> FLC = 1.1, FAC = 0.9 -> SVTFAC = 0.99
        assert svtfac == pytest.approx(1.1 * 0.9, rel=1e-5)

    def test_compute_porosity_porfor6(self):
        """Test Anagonye-Wang biaxial strain coupled porosity (porfor6.F)."""
        # SVTFAC = (X0 + X2 * RP)/RS + X1 + X3 * RP
        x0, x1, x2, x3 = 0.1, 0.05, 0.02, 0.01
        p = 2.0e5
        pext = 1.0e5
        area = 1.5
        area0 = 1.0

        svtfac = compute_porosity_porfor6(
            p=p,
            pext=pext,
            area=area,
            area0=area0,
            x0=x0,
            x1=x1,
            x2=x2,
            x3=x3,
        )
        # RS = max(1.5, 1.0) = 1.5, RP = min(0.5, 1.0) = 0.5
        # SVTFAC = (0.1 + 0.02 * 0.5)/1.5 + 0.05 + 0.01 * 0.5
        #        = 0.11 / 1.5 + 0.055 = 0.073333... + 0.055 = 0.128333...
        expected = (0.1 + 0.02 * 0.5) / 1.5 + 0.05 + 0.01 * 0.5
        assert svtfac == pytest.approx(expected, rel=1e-6)
