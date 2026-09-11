"""
CFG schema catalogue and law name mappings (M539).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from .mat_reader import CfgCatalogue, catalogue, CfgLawSchema

# Mapping of material law names and synonyms to integer law numbers.
LAW_MAP: Dict[str, int] = {
    "VOID": 0,
    "LAW0": 0,
    "ELAST": 1,
    "LAW1": 1,
    "PLAS_JOHNS": 2,
    "JOHNSON_COOK": 2,
    "LAW2": 2,
    "PLAS_ZERIL": 3,
    "LAW3": 3,
    "HYDRO": 4,
    "LAW4": 4,
    "JWL": 5,
    "LAW5": 5,
    "HYD_JCOOK": 6,
    "LAW6": 6,
    "SOIL": 10,
    "SOIL_CONC": 10,
    "DPRAG": 10,
    "LAW10": 10,
    "LAW12": 12,
    "3D_COMP": 12,
    "COMP_3D": 12,
    "3PARBI": 12,
    "RAGAB": 12,
    "LAW14": 14,
    "COMPSO": 14,
    "COMP_SOL": 14,
    "LAW15": 15,
    "CHANG": 15,
    "PLAS_ANISO": 15,
    "COMP_CHANG": 15,
    "FABRI": 19,
    "LAW19": 19,
    "LAW22": 22,
    "DAMA": 22,
    "PLAS_DAMA": 22,
    "CONC": 24,
    "LAW24": 24,
    "COMP_PLAS": 25,
    "COMPSH": 25,
    "TSAI_WU": 25,
    "CRASURV": 25,
    "COMPOSITE_PLAS": 25,
    "LAW25": 25,
    "HONEYCOMB": 28,
    "HONEYCOMB_SOL": 28,
    "LAW28": 28,
    "HILL": 32,
    "LAW32": 32,
    "FOAM_PLAS": 33,
    "LAW33": 33,
    "BOLTZMAN": 34,
    "VISC_MAXW": 34,
    "BOLTZMANN": 34,
    "LAW34": 34,
    "FOAM_VISC": 35,
    "LAW35": 35,
    "PLAS_TAB": 36,
    "LAW36": 36,
    "BIPHAS": 37,
    "BIPHASIC": 37,
    "LAW37": 37,
    "VISC_TAB": 38,
    "LAW38": 38,
    "LAW43": 43,
    "HILL_TAB": 43,
    "LAW43_HILL_TAB": 43,
    "COWPER_SYMONDS": 44,
    "LAW44": 44,
    "DONEA": 46,
    "HYD_VISC": 46,
    "LAW46": 46,
    "VISC_HYP": 62,
    "LAW62": 62,
    "FOAM_TAB": 66,
    "LAW66": 66,
    "OGDEN": 88,
    "HYPER_ELAS": 88,
    "LAW88": 88,
    "ARRUDA_BOYCE": 92,
    "LAW92": 92,
    "YEOH": 94,
    "LAW94": 94,
}

LAW_SYNONYMS: Dict[str, str] = {
    "COMP_PLAS": "LAW25",
    "COMPSH": "LAW25",
    "TSAI_WU": "LAW25",
    "CRASURV": "LAW25",
    "COMPOSITE_PLAS": "LAW25",
    "LAW25": "LAW25",
    "HILL": "LAW32",
    "LAW32": "LAW32",
    "VISC_TAB": "LAW38",
    "LAW38": "LAW38",
    "BIPHAS": "LAW37",
    "BIPHASIC": "LAW37",
    "LAW37": "LAW37",
    "BOLTZMAN": "LAW34",
    "VISC_MAXW": "LAW34",
    "BOLTZMANN": "LAW34",
    "LAW34": "LAW34",
    "HONEYCOMB": "LAW28",
    "JWL": "LAW5",
    "SOIL": "LAW10",
    "SOIL_CONC": "LAW10",
    "LAW12": "LAW12",
    "3D_COMP": "LAW12",
    "COMP_3D": "LAW12",
    "3PARBI": "LAW12",
    "RAGAB": "LAW12",
    "LAW14": "LAW14",
    "COMPSO": "LAW14",
    "COMP_SOL": "LAW14",
    "LAW15": "LAW15",
    "CHANG": "LAW15",
    "PLAS_ANISO": "LAW15",
    "COMP_CHANG": "LAW15",
    "LAW22": "LAW22",
    "DAMA": "LAW22",
    "PLAS_DAMA": "LAW22",
    "LAW43": "LAW43",
    "HILL_TAB": "LAW43",
    "LAW43_HILL_TAB": "LAW43",
}

KEYWORD_NAME_MAP = LAW_MAP
SYNONYMS = LAW_SYNONYMS
_SYNONYMS = LAW_SYNONYMS


def law_number(name: str) -> Optional[int]:
    """Map a law name, alias, or keyword spelling to its integer law number."""
    uname = str(name).strip().upper()
    if uname.startswith("/"):
        uname = uname.lstrip("/")
    if uname.startswith("MAT/"):
        uname = uname[4:]
    elif uname.startswith("MAT_"):
        uname = uname[4:]
    if uname in LAW_MAP:
        return LAW_MAP[uname]
    m = re.fullmatch(r"LAW(\d+)", uname)
    if m:
        return int(m.group(1))
    try:
        return int(uname)
    except ValueError:
        return None


get_law_number = law_number


def canonical_law_name(name: str) -> str:
    """Map a law spelling to canonical 'LAW<n>' or registered synonym."""
    uname = str(name).strip().upper()
    if uname.startswith("/"):
        uname = uname.lstrip("/")
    if uname.startswith("MAT/"):
        uname = uname[4:]
    elif uname.startswith("MAT_"):
        uname = uname[4:]
    if uname in LAW_SYNONYMS:
        return LAW_SYNONYMS[uname]
    num = law_number(uname)
    if num is not None:
        return f"LAW{num}"
    return uname


__all__ = [
    "CfgCatalogue",
    "catalogue",
    "CfgLawSchema",
    "LAW_MAP",
    "LAW_SYNONYMS",
    "_SYNONYMS",
    "KEYWORD_NAME_MAP",
    "SYNONYMS",
    "law_number",
    "get_law_number",
    "canonical_law_name",
]
