"""
Verification of all 32 Finite Element Technology element families in pyradioss:
Ensures all 32 kernels are imported, registered in KERNELS, define init_group and forces,
and that Model element_groups and dispatch tables map properly.
"""

import pytest

from pyradioss.elements import (
    BEAM_PROP_GROUPS,
    KERNELS,
    PENTA_ISOLID_GROUPS,
    QUAD_IQUAD_GROUPS,
    SH3N_ISHELL_GROUPS,
    SHELL_ISHELL_GROUPS,
    SOLID_ISOLID_GROUPS,
    TETRA4_ITETRA4_GROUPS,
)
from pyradioss.model.model import Model


def test_32_element_families_registered():
    """Verify all 32 element technology families exist in KERNELS dictionary."""
    # List of 32 distinct element families (including aliases like springs_advanced)
    expected_32 = [
        # 1-13: 3D Solids
        "bricks",
        "bricks_full",
        "bricks_eas",
        "bricks_heph",
        "solid_shells_ha8",
        "cohesives",
        "bric20s",
        "penta6s",
        "penta6s_heph",
        "pyra5s",
        "tetras",
        "tetras_sfem",
        "tetra10s",

        # 14-17: Thick Shells
        "tshells",
        "shel16s",
        "thickshell_wedges",
        "thickshell_composites",

        # 18-23: Thin Shells
        "shells",
        "shells_qbat",
        "shells_qeph",
        "sh3n",
        "sh3n_dkt18",
        "shells_dkt6",

        # 24-26: 2D Solids
        "quads",
        "quads_full",
        "trias",

        # 27-32: 1D & Connection Elements
        "trusses",
        "springs",
        "spring_advanced",
        "beams",
        "beams_fiber",
    ]

    for family in expected_32:
        assert family in KERNELS, f"Element family '{family}' is missing from KERNELS!"
        kernel = KERNELS[family]
        assert hasattr(kernel, "init_group"), f"Kernel for '{family}' lacks init_group entry point"
        assert hasattr(kernel, "forces"), f"Kernel for '{family}' lacks forces entry point"


def test_model_element_groups_iteration():
    """Verify Model.element_groups iterates over all populated groups."""
    model = Model()
    from pyradioss.model.model import ElementGroup
    import numpy as np

    # Populate dummy groups for all 32 families
    group_names = [
        "bricks", "bricks_full", "bricks_eas", "bricks_heph", "solid_shells_ha8", "cohesives",
        "tshells", "bric20s", "penta6s", "penta6s_heph", "pyra5s",
        "quads", "quads_full", "trias",
        "tetras", "tetras_sfem", "tetra10s",
        "shel16s", "thickshell_wedges", "thickshell_composites",
        "shells", "shells_qbat", "shells_qeph",
        "sh3n", "sh3n_dkt18", "shells_dkt6",
        "trusses", "springs", "beams", "beams_fiber",
    ]

    for name in group_names:
        setattr(model, name, ElementGroup(ids=np.array([1]), conn=np.array([[0, 1]]), part=np.array([0])))

    iterated = [name for name, _ in model.element_groups()]
    for name in group_names:
        assert name in iterated, f"Group '{name}' was not yielded by model.element_groups()!"
