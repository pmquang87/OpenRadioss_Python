"""
M55 validations: 2D solid elements (/QUAD, /ANALY).

Analytic validations (the port's philosophy):
- N2D=2 (Plane strain): Uniaxial y-compression gives exactly the 1D constrained modulus P-wave stress.
- N2D=1 (Axisymmetric): Radial expansion of a ring gives exactly the Lame cylinder hoop/radial stresses.
"""

import numpy as np
import pytest

import numpy as np

from pyradioss.model.model import Model, ElementGroup
from pyradioss.model.entities import Material
from pyradioss.elements import solid_quad

def test_quad_plane_strain():
    """N2D=2 (Plane Strain): Pure compression in Z (axial).
    sig_zz = (K + 4/3 G) eps_zz.
    """
    model = Model()
    model.n2d = 2
    # Node coordinates (n, 3). X=0, Y=1, Z=2
    # Unit square: Y in [0, 1], Z in [0, 1]
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    
    mat = Material(id=1, law=1, title="mat", rho0=1.0, params={"E": 210000.0, "nu": 0.3})
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    group.state = {"slices": [(slice(0, 1), mat, None)]}
    
    solid_quad.init_group(group, model, None)
    
    # Expand in Z by 0.1%
    dt = 1.0
    x = model.x0.copy()
    v = np.zeros_like(x)
    # v_z = 0.001 for top nodes
    v[2, 2] = 0.001
    v[3, 2] = 0.001
    
    fint = np.zeros_like(x)
    dt_crit = solid_quad.forces(group, x, v, None, dt, fint, None)
    
    # eps_zz = -0.001
    # E = 210000, nu = 0.3
    # K = E / (3*(1-2*nu)) = 210000 / 1.2 = 175000
    # G = E / (2*(1+nu)) = 210000 / 2.6 = 80769.23
    # K + 4/3 G = 175000 + 107692.3 = 282692.3
    # sig_zz = +282.6923
    # sig_yy = +121.1538
    
    sig = group.state["sig"][0]
    np.testing.assert_allclose(sig[2], 282.6923, rtol=1e-4) # ZZ
    np.testing.assert_allclose(sig[1], 121.1538, rtol=1e-4) # YY
    np.testing.assert_allclose(sig[0], 121.1538, rtol=1e-4)      # XX
    
    # Force on top nodes should be sig_zz * Area_face = 282.6923 * 1.0
    # fint accumulates resisting force (minus sign). Element is stretched, so it pulls DOWN (negative Z).
    np.testing.assert_allclose(fint[2, 2], -141.34615, rtol=1e-4) # half to each node
    np.testing.assert_allclose(fint[3, 2], -141.34615, rtol=1e-4)


def test_quad_axisymmetric():
    """N2D=1 (Axisymmetric): Radial expansion.
    """
    model = Model()
    model.n2d = 1
    # Unit square from Y=10 to Y=11, Z=0 to Z=1
    model.x0 = np.array([
        [0.0, 10.0, 0.0],
        [0.0, 11.0, 0.0],
        [0.0, 11.0, 1.0],
        [0.0, 10.0, 1.0],
    ])
    conn = np.array([[0, 1, 2, 3]])
    
    mat = Material(id=1, law=1, title="mat", rho0=1.0, params={"E": 210000.0, "nu": 0.0}) # nu=0 for simplicity
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    group.state = {"slices": [(slice(0, 1), mat, None)]}
    
    solid_quad.init_group(group, model, None)
    
    dt = 1.0
    x = model.x0.copy()
    v = np.zeros_like(x)
    # v_y = 1.0 for all nodes (pure translation radially)
    v[:, 1] = 1.0
    
    # DYY = 0, DZZ = 0, DTT = v_y / Y_avg = 1.0 / 10.5 = 0.095238
    # sig_xx (hoop) = E * DTT = 210000 * 0.095238 = 20000
    
    fint = np.zeros_like(x)
    solid_quad.forces(group, x, v, None, dt, fint, None)
    
    sig = group.state["sig"][0]
    np.testing.assert_allclose(sig[0], 210000.0 / 10.5, rtol=1e-4) # XX (hoop)
    np.testing.assert_allclose(sig[1], 0.0, atol=1e-10) # YY
    np.testing.assert_allclose(sig[2], 0.0, atol=1e-10) # ZZ
    
    # AX1 = (S3 - S1) * Area / (4 * Yavg) = 20000 * 1.0 / 42.0 = 476.19
    # Each node should have resisting FY = -AX1 (pulling inward)
    np.testing.assert_allclose(fint[:, 1], -210000.0 / 10.5 / 4.0 / 10.5, rtol=1e-4)
