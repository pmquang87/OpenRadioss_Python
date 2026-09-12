"""
pyradioss.input.checks — input and starter sanity checks for materials, properties, and elements.
"""

from __future__ import annotations

from ..starter.checks import (
    check_mat_law106,
    check_mat_jcook_alm,
    check_mat_johns_cook_alm,
    check_mat_law104,
    check_mat_drucker,
    check_mat_johns_voce_drucker,
    check_mat_plas_druck,
    check_mat_law103,
    check_mat_hensel_spittel,
    check_mat_plas_hens,
    check_mat_law102,
    check_mat_dprag2,
    check_mat_law101,
    check_mat_law100,
    check_mat_law95,
    check_mat_law93,
    check_mat_law94,
    check_mat_law92,
    check_mat_law88,
    check_mat_law87,
    check_materials,
    check_model,
)

check_all = check_model

__all__ = [
    "check_mat_law106",
    "check_mat_jcook_alm",
    "check_mat_johns_cook_alm",
    "check_mat_law104",
    "check_mat_drucker",
    "check_mat_johns_voce_drucker",
    "check_mat_plas_druck",
    "check_mat_law103",
    "check_mat_hensel_spittel",
    "check_mat_plas_hens",
    "check_mat_law102",
    "check_mat_dprag2",
    "check_mat_law101",
    "check_mat_law100",
    "check_mat_law95",
    "check_mat_law93",
    "check_mat_law94",
    "check_mat_law92",
    "check_mat_law88",
    "check_mat_law87",
    "check_materials",
    "check_model",
    "check_all",
]
