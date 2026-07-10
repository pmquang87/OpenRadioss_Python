"""
pyradioss — a Python port of OpenRadioss.

OpenRadioss (https://github.com/OpenRadioss/OpenRadioss) is an open-source
explicit finite-element solver for crash, impact and highly non-linear
transient dynamics, written in Fortran (with C/C++ glue). This package
reproduces its functional architecture in pure Python + NumPy:

    pyradioss.starter   the "Starter"  : reads <Run>_0000.rad, checks and
                                         initializes the model, writes the
                                         restart file for the Engine.
    pyradioss.engine    the "Engine"   : reads <Run>_0001.rad + restart,
                                         runs the explicit time loop and
                                         writes all results.
    pyradioss.input     deck reader + keyword parsers
    pyradioss.model     in-memory model (nodes, elements, materials, ...)
    pyradioss.elements  element force/time-step kernels
    pyradioss.materials constitutive laws (LAW1, LAW2, ...)
    pyradioss.contact   contact interfaces (TYPE7, ...)
    pyradioss.output    listings, time history, animation files

Start reading with PORTING_GUIDE.md at the repository root: it maps every
module here back to the original Fortran directory it was ported from.
"""

__version__ = "0.1.0"

# A short banner in the spirit of the original solver's startup header
# (see engine/source/output/message/ in the Fortran tree).
BANNER = r"""
 ************************************************************************
 **                                                                    **
 **                    pyradioss  —  OpenRadioss port                  **
 **                                                                    **
 **        Python port of the OpenRadioss explicit FE solver           **
 **   Non-linear finite element analysis software from OpenRadioss     **
 **                                                                    **
 **               https://github.com/OpenRadioss/OpenRadioss           **
 **                                                                    **
 ************************************************************************
 ** pyradioss version {version:<48}**
 ************************************************************************
"""


def banner() -> str:
    """Return the startup banner string (printed by both Starter and Engine)."""
    return BANNER.format(version=__version__)
