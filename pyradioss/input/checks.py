"""
pyradioss.input.checks — input and starter sanity checks for materials, properties, and elements.
"""

from __future__ import annotations

from ..starter.checks import (
    check_mat_law88,
    check_mat_law87,
    check_materials,
    check_model,
)

check_all = check_model

__all__ = [
    "check_mat_law88",
    "check_mat_law87",
    "check_materials",
    "check_model",
    "check_all",
]
