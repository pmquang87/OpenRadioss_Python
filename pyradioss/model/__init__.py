"""
pyradioss.model — the in-memory model shared by Starter and Engine.

Fortran origin: the derived types and modules of ``common_source/modules``
(``elbufdef_mod.F`` element buffers, ``groupdef_mod.F`` groups,
``intbufdef_mod.F`` contact buffers, ...) plus the big Starter arrays
(``X``, ``IXS``, ``IXC``, ``IPART``, ``PM``, ``GEO`` ...).

Naming note for readers of the Fortran: the classic array names are kept in
comments next to their Python equivalents —

    X(3,N)      -> Model.x           nodal coordinates
    V(3,N)      -> Model.v           nodal velocities
    MS(N)       -> Model.mass        lumped nodal mass
    IXS(11,N)   -> BrickGroup.conn   solid connectivity (+mat/prop ids)
    IXC(7,N)    -> ShellGroup.conn   shell connectivity
    PM(:, m)    -> Material.*        material constants
    GEO(:, p)   -> Property.*        property (geometry set) constants
"""

from .model import Model, EngineControls  # noqa: F401
from .entities import (  # noqa: F401
    Material, Property, Part, NodeGroup, Surface, BoundaryCondition,
    InitialVelocity, Gravity, ConcentratedLoad, ImposedVelocity,
    RigidWall, Interface, Line, THRequest, Box,
    MatLaw25, MatCompPlas, MatCompsh, MatTsaiWu, MatCrasurv, MatCompositePlas,
    MatLaw58, MatFabrA, MatFabricA,
    MatLaw52, MatGurson, MatPlasGurs,
    MatLaw21, MatDprag,
    MatLaw49, MatSteinb, MatSteinberg, MatSteinbergGuinan,
    MatLaw79, MatJohnHolm, MatJohnsonHolmquist, MatJH2,
    MatLaw87, MatBarlat2000, MatBarlat20002D, MaterialLaw87,
    MaterialLaw88, MatLaw88, MatTabulatedHyperelastic, MatHyperElas, MatTabHyp,
    MaterialLaw109, MatLaw109, MatTabPlas, MaterialTabPlas, MatElastoPlasTab, MatLaw109TabPlas,
)

