"""
pyradioss.starter — the Starter program.

Fortran origin: the ``starter/`` half of the OpenRadioss tree. The Starter
never integrates in time: it reads the deck, *checks* it, converts user IDs
to internal indices, builds every derived structure (element buffers,
lumped masses, contact surfaces, node groups) and hands the initialized
model to the Engine through the restart file.

Pipeline (see starter.py):

    read_deck -> parse keywords -> finalize (ids->idx, groups, surfaces)
              -> element init (buffers + lumped mass) -> checks
              -> write listing *_0000.out -> write restart *_0000.rst
"""

from .starter import run_starter  # noqa: F401
from .inivel import (  # noqa: F401
    InivelRecord,
    InivelType,
    apply_bcs_mask,
    apply_inivel,
    build_cylindrical_inivel,
    build_rotational_inivel,
    build_spherical_inivel,
    build_translational_inivel,
)
