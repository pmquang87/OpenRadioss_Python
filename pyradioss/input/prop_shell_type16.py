"""
User-defined Layered Shell Property (/PROP/TYPE16, /PROP/SH_FABR).

Ported from OpenRadioss Fortran:
- ``starter/source/properties/shell/hm_read_prop16.F`` (SUBROUTINE HM_READ_PROP16)

Formulation Details
-------------------
/PROP/TYPE16 defines a layered composite shell property with explicit user-defined
through-thickness layers (up to 100 plies).

For each ply k = 1 ... nply:
  - mat_id: Material ID (IPMAT)
  - thick: Layer thickness t_k (IPTHK)
  - angle: Fiber orientation angle phi_k in degrees (IPANG)
  - z_pos: Mid-plane thickness coordinate z_k (IPPOS)
  - weight: Numerical integration quadrature weight w_k (IPWEIGHT)
  - alpha: Second fiber angle (defaults to 90 degrees)

Through-thickness Layer Positioning (hm_read_prop16.F lines 384-416):
1. IPOS == 0 (Automatic calculation from layer thicknesses):
   - Total thickness: T_total = sum_{k=1}^{nply} t_k
   - Mid-plane offset: zshift = 0.0
   - Layer 1 (bottom): z_1 = -T_total / 2 + t_1 / 2
   - Subsequent layers: z_k = z_{k-1} + (t_{k-1} + t_k) / 2
2. IPOS > 0 (User-defined z positions provided):
   - T_min = min_k (z_k - t_k / 2)
   - T_max = max_k (z_k + t_k / 2)
   - Total thickness: T_total = T_max - T_min
   - Mid-plane offset: zshift = (T_max + T_min) / 2

Integration Points:
  The method `get_integration_points()` yields (z_coords, weights, mat_ids, angles)
  enabling direct evaluation of through-thickness stress and moment integrals:
      N = sum_k sigma_k * w_k
      M = sum_k sigma_k * z_k * w_k
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..common.messages import MessageLog
from ..model.entities import Property
from .deck_reader import Card, KeywordBlock, parse_fortran_float


@dataclass
class PropType16Layer:
    """Individual layer / ply in a /PROP/TYPE16 composite shell.

    Fortran reference: ``hm_read_prop16.F`` (IPMAT, IPTHK, IPANG, IPPOS, IPWEIGHT).

    Attributes
    ----------
    mat_id : int
        Material ID of this layer (IPMAT).
    thick : float
        Thickness of this layer t_k (IPTHK).
    angle : float
        Fiber orientation angle in degrees (IPANG / phi).
    z_pos : float
        Mid-plane thickness coordinate position z_k (IPPOS).
    weight : float
        Numerical quadrature weight for through-thickness integration (IPWEIGHT).
        In physical units, weight = thick; in normalized units, weight = thick / T_total.
    alpha : float
        Second fiber axis orientation in degrees (Card 5 alpha, default 90.0).
    """

    mat_id: int = 0
    thick: float = 0.0
    angle: float = 0.0
    z_pos: float = 0.0
    weight: float = 0.0
    alpha: float = 90.0

    @property
    def phi(self) -> float:
        """Alias for angle (fiber orientation in degrees)."""
        return self.angle

    @phi.setter
    def phi(self, val: float) -> None:
        self.angle = float(val)

    @property
    def z(self) -> float:
        """Alias for z_pos (mid-plane coordinate position)."""
        return self.z_pos

    @z.setter
    def z(self, val: float) -> None:
        self.z_pos = float(val)


@dataclass
class PropType16(Property):
    """/PROP/TYPE16 or /PROP/SH_FABR composite layered shell property.

    Fortran reference: ``starter/source/properties/shell/hm_read_prop16.F``.

    Attributes
    ----------
    prop_id : int
        User property ID.
    title : str
        Property title / comment.
    ishell : int
        Shell formulation (1=Belytschko-Tsay, 2=Batoz, 3=QB3, 4=BATOZ, 12/24=QEPH, etc.).
    ismstr : int
        Large strain flag (1=small strain, 2=full large strain, 4=default).
    ithick : int
        Thickness change formulation (0=constant thickness, 1=variable thickness).
    nply : int
        Number of user-defined plies / layers (up to 100).
    total_thickness : float
        Sum of layer thicknesses T_total.
    layers : List[PropType16Layer]
        List of ply specifications through thickness.
    ashear : float
        Transverse shear correction factor (default 5/6 = 0.8333333333333334).
    zshift : float
        Mid-surface offset relative to reference geometry.
    ish3n : int
        3-node shell formulation flag.
    p_thick_fail : float
        Shell thickness ratio threshold for element deletion.
    hm, hf, hr : float
        Hourglass viscosity parameters for membrane, out-of-plane and rotation.
    dm, dn : float
        Damping parameters.
    istrain : int
        Strain formulation flag.
    vx, vy, vz : float
        Reference direction vector for fiber orientation.
    skew_id : int
        Skew coordinate system ID for orientation.
    ipos : int
        Layer positioning flag (0 = auto compute from thicknesses, 1 = user coordinates).
    ip : int
        Hourglass stabilization formulation option.
    """

    prop_id: int = 0
    ishell: int = 24
    ismstr: int = 4
    ithick: int = 0
    nply: int = 0
    total_thickness: float = 0.0
    layers: List[PropType16Layer] = field(default_factory=list)
    ashear: float = 5.0 / 6.0
    zshift: float = 0.0
    ish3n: int = 0
    p_thick_fail: float = 0.0
    hm: float = 0.01
    hf: float = 0.01
    hr: float = 0.01
    dm: float = 0.0
    dn: float = 0.0
    istrain: int = 1
    vx: float = 1.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    ipos: int = 0
    ip: int = 0

    def __init__(
        self,
        prop_id: int = 0,
        title: str = "",
        ishell: int = 24,
        ismstr: int = 4,
        ithick: int = 0,
        nply: int = 0,
        total_thickness: float = 0.0,
        layers: Optional[List[PropType16Layer]] = None,
        ashear: float = 5.0 / 6.0,
        zshift: float = 0.0,
        ish3n: int = 0,
        p_thick_fail: float = 0.0,
        hm: float = 0.01,
        hf: float = 0.01,
        hr: float = 0.01,
        dm: float = 0.0,
        dn: float = 0.0,
        istrain: int = 1,
        vx: float = 1.0,
        vy: float = 0.0,
        vz: float = 0.0,
        skew_id: int = 0,
        ipos: int = 0,
        ip: int = 0,
        id: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        actual_id = id if id is not None else prop_id
        self.prop_id = actual_id
        self.ishell = ishell or 24
        self.ismstr = ismstr or 4
        self.ithick = ithick
        self.layers = layers if layers is not None else []
        self.nply = nply or len(self.layers)
        self.total_thickness = float(total_thickness)
        self.ashear = ashear if ashear > 0.0 else 5.0 / 6.0
        self.zshift = float(zshift)
        self.ish3n = ish3n
        self.p_thick_fail = p_thick_fail
        self.hm = hm if hm != 0.0 else 0.01
        self.hf = hf if hf != 0.0 else 0.01
        self.hr = hr if hr != 0.0 else 0.01
        self.dm = dm
        self.dn = dn
        self.istrain = istrain
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.skew_id = skew_id
        self.ipos = ipos
        self.ip = ip

        # Normalize reference direction
        norm_v = math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)
        if norm_v < 1e-10:
            self.vx, self.vy, self.vz = 1.0, 0.0, 0.0
        else:
            self.vx /= norm_v
            self.vy /= norm_v
            self.vz /= norm_v

        # Recalculate thickness and mid-points if layers present
        if self.layers:
            self._update_geometry()

        # Build Property.params dictionary
        p_dict = params.copy() if params is not None else {}
        p_dict.update({
            "thick": self.total_thickness,
            "total_thickness": self.total_thickness,
            "ashear": self.ashear,
            "nip": self.nply,
            "nply": self.nply,
            "ishell": self.ishell,
            "ismstr": self.ismstr,
            "ish3n": self.ish3n,
            "p_thick_fail": self.p_thick_fail,
            "hm": self.hm,
            "hf": self.hf,
            "hr": self.hr,
            "dm": self.dm,
            "dn": self.dn,
            "istrain": self.istrain,
            "ithick": self.ithick,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "skew_id": self.skew_id,
            "ipos": self.ipos,
            "ip": self.ip,
            "zshift": self.zshift,
            "layers": [
                {
                    "mat_id": l.mat_id,
                    "thick": l.thick,
                    "angle": l.angle,
                    "phi": l.angle,
                    "z_pos": l.z_pos,
                    "z": l.z_pos,
                    "weight": l.weight,
                    "alpha": l.alpha,
                }
                for l in self.layers
            ],
        })

        super().__init__(id=actual_id, type=16, title=title, params=p_dict)

    def _update_geometry(self) -> None:
        """Compute layer mid-points, total thickness, and quadrature weights."""
        if not self.layers:
            return

        self.nply = len(self.layers)

        if self.ipos > 0:
            # User defined coordinates z_k (hm_read_prop16.F 384-399)
            t_min = min(l.z_pos - 0.5 * l.thick for l in self.layers)
            t_max = max(l.z_pos + 0.5 * l.thick for l in self.layers)
            calc_thick = t_max - t_min
            if calc_thick > 0.0:
                self.total_thickness = calc_thick
            self.zshift = 0.5 * (t_max + t_min)
            for l in self.layers:
                if l.weight == 0.0:
                    l.weight = l.thick
        else:
            # Automatic positioning from bottom to top (hm_read_prop16.F 401-415)
            sum_thk = sum(l.thick for l in self.layers)
            if sum_thk > 0.0:
                self.total_thickness = sum_thk
            elif self.total_thickness > 0.0 and self.nply > 0:
                # Divide evenly if total thickness given but layers 0
                thk_per_ply = self.total_thickness / self.nply
                for l in self.layers:
                    l.thick = thk_per_ply

            self.zshift = 0.0
            z_curr = -0.5 * self.total_thickness
            for l in self.layers:
                l.z_pos = z_curr + 0.5 * l.thick
                z_curr += l.thick
                if l.weight == 0.0:
                    l.weight = l.thick

    compute_layer_positions = _update_geometry

    @property
    def nip(self) -> int:
        """Number of through-thickness integration points (number of plies)."""
        return self.nply

    @property
    def thick(self) -> float:
        """Total laminate thickness."""
        return self.total_thickness

    def validate_weights(self, rtol: float = 1e-4) -> bool:
        """Validate that the sum of layer weights matches total thickness or unity.

        Returns
        -------
        bool
            True if weights sum to total_thickness or 1.0 within tolerance.
        """
        if not self.layers:
            return True
        weight_sum = sum(l.weight for l in self.layers)
        if math.isclose(weight_sum, self.total_thickness, rel_tol=rtol, abs_tol=1e-8):
            return True
        if math.isclose(weight_sum, 1.0, rel_tol=rtol, abs_tol=1e-8):
            return True
        return False

    def get_integration_points(
        self, normalized: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays of through-thickness integration point data.

        Parameters
        ----------
        normalized : bool, default False
            If True, returns normalized coordinates in [-0.5, 0.5] and weights
            summing to 1.0. If False (default), returns physical coordinates
            in [-T/2, T/2] and physical weights summing to total_thickness.

        Returns
        -------
        z_coords : np.ndarray of float
            Through-thickness coordinate positions (shape: nply).
        weights : np.ndarray of float
            Integration weights (shape: nply).
        mat_ids : np.ndarray of int
            Material IDs for each ply (shape: nply).
        angles : np.ndarray of float
            Fiber orientation angles in degrees (shape: nply).
        """
        if not self.layers:
            return (
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=int),
                np.empty(0, dtype=float),
            )

        mat_ids = np.array([l.mat_id for l in self.layers], dtype=int)
        angles = np.array([l.angle for l in self.layers], dtype=float)

        if normalized:
            tot = max(self.total_thickness, 1e-20)
            z_coords = np.array([l.z_pos / tot for l in self.layers], dtype=float)
            weights = np.array([l.weight / tot for l in self.layers], dtype=float)
        else:
            z_coords = np.array([l.z_pos for l in self.layers], dtype=float)
            weights = np.array([l.weight for l in self.layers], dtype=float)

        return z_coords, weights, mat_ids, angles


# ----------------------------------------------------------------------------
# Parser helpers
# ----------------------------------------------------------------------------

def _fv(val: Any, default: float = 0.0) -> float:
    if val in ("", None):
        return default
    try:
        return parse_fortran_float(str(val))
    except (ValueError, TypeError):
        return default


def _iv(val: Any, default: int = 0) -> int:
    if val in ("", None):
        return default
    try:
        return int(parse_fortran_float(str(val)))
    except (ValueError, TypeError):
        return default


def parse_prop16(
    block: Optional[KeywordBlock] = None,
    card_lines: Optional[Sequence[str]] = None,
    log: Optional[MessageLog] = None,
) -> PropType16:
    """Parse a /PROP/TYPE16 (/PROP/SH_FABR) composite shell property.

    Fortran reference: ``starter/source/properties/shell/hm_read_prop16.F``.

    Parameters
    ----------
    block : KeywordBlock, optional
        Keyword block from deck reader.
    card_lines : Sequence[str], optional
        Direct lines of card data (optional alternative to block).
    log : MessageLog, optional
        Message logger for warnings/errors.

    Returns
    -------
    PropType16
        The parsed composite shell property.
    """
    prop_id = 0
    title = ""
    is_fixed = False
    valid_cards: List[Card] = []

    if block is not None:
        prop_id = block.user_id or 0
        is_fixed = getattr(block, "fixed", False)
        # Extract title and valid data cards
        if is_fixed and callable(getattr(block, "fixed_cards", None)):
            raw_cards = list(block.fixed_cards())
            while raw_cards and raw_cards[-1].is_blank:
                raw_cards.pop()
            if raw_cards:
                title = raw_cards[0].raw.strip()
                valid_cards = [c for c in raw_cards[1:] if not c.is_blank]
        else:
            raw_cards = list(block.cards)
            if raw_cards and raw_cards[0].tokens():
                first_toks = raw_cards[0].tokens()
                # If first card has non-numeric token, it is title
                is_num = True
                for t in first_toks:
                    try:
                        parse_fortran_float(t)
                    except ValueError:
                        is_num = False
                        break
                if not is_num:
                    title = raw_cards[0].raw.strip()
                    valid_cards = [c for c in raw_cards[1:] if not c.is_blank]
                else:
                    valid_cards = [c for c in raw_cards if not c.is_blank]
    elif card_lines is not None:
        lines = [str(l) for l in card_lines]
        if lines:
            # Check if first line is title
            tokens0 = lines[0].split()
            is_num0 = True
            for t in tokens0:
                try:
                    parse_fortran_float(t)
                except ValueError:
                    is_num0 = False
                    break
            if not is_num0:
                title = lines[0].strip()
                card_strs = lines[1:]
            else:
                card_strs = lines
            valid_cards = [Card(s) for s in card_strs if s.strip()]

    if not valid_cards:
        if log is not None:
            log.error(f"/PROP/TYPE16/{prop_id}: missing data cards", getattr(block, "source", ""))
        return PropType16(prop_id=prop_id, title=title)

    ishell, ismstr, ish3n = 24, 4, 0
    p_thick_fail = 0.0
    hm, hf, hr, dm, dn = 0.01, 0.01, 0.01, 0.0, 0.0
    nip, istrain = 0, 1
    thick, ashear = 0.0, 5.0 / 6.0
    ithick = 0
    vx, vy, vz = 1.0, 0.0, 0.0
    skew_id, ipos, ip = 0, 0, 0
    layers: List[PropType16Layer] = []

    if is_fixed:
        from .card_layouts import CARD_LAYOUTS

        # Card 1: PROP_TYPE16_1
        c0 = valid_cards[0]
        if "PROP_TYPE16_1" in CARD_LAYOUTS:
            f1 = c0.cut("PROP_TYPE16_1")
        else:
            f1 = [c0.raw[0:10], c0.raw[10:20], c0.raw[20:30], c0.raw[30:60], c0.raw[60:80]]
        ishell = _iv(f1[0], 24) if len(f1) > 0 else 24
        ismstr = _iv(f1[1], 4) if len(f1) > 1 else 4
        ish3n = _iv(f1[2], 0) if len(f1) > 2 else 0
        p_thick_fail = _fv(f1[4], 0.0) if len(f1) > 4 else 0.0

        # Card 2: PROP_TYPE16_2
        if len(valid_cards) > 1:
            c1 = valid_cards[1]
            if "PROP_TYPE16_2" in CARD_LAYOUTS:
                f2 = c1.cut("PROP_TYPE16_2")
            else:
                f2 = [c1.raw[0:20], c1.raw[20:40], c1.raw[40:60], c1.raw[60:80], c1.raw[80:100]]
            hm = _fv(f2[0], 0.01) if len(f2) > 0 else 0.01
            hf = _fv(f2[1], 0.01) if len(f2) > 1 else 0.01
            hr = _fv(f2[2], 0.01) if len(f2) > 2 else 0.01
            dm = _fv(f2[3], 0.0) if len(f2) > 3 else 0.0
            dn = _fv(f2[4], 0.0) if len(f2) > 4 else 0.0

        # Card 3: PROP_TYPE16_3
        if len(valid_cards) > 2:
            c2 = valid_cards[2]
            if "PROP_TYPE16_3" in CARD_LAYOUTS:
                f3 = c2.cut("PROP_TYPE16_3")
            else:
                f3 = [c2.raw[0:10], c2.raw[10:20], c2.raw[20:40], c2.raw[40:60], c2.raw[60:70], c2.raw[70:80]]
            nip = _iv(f3[0], 0) if len(f3) > 0 else 0
            istrain = _iv(f3[1], 1) if len(f3) > 1 else 1
            thick = _fv(f3[2], 0.0) if len(f3) > 2 else 0.0
            ashear = _fv(f3[3], 5.0 / 6.0) if len(f3) > 3 else 5.0 / 6.0
            ithick = _iv(f3[5], 0) if len(f3) > 5 else 0

        # Card 4: PROP_TYPE16_4
        if len(valid_cards) > 3:
            c3 = valid_cards[3]
            if "PROP_TYPE16_4" in CARD_LAYOUTS:
                f4 = c3.cut("PROP_TYPE16_4")
            else:
                f4 = [c3.raw[0:20], c3.raw[20:40], c3.raw[40:60], c3.raw[60:70], c3.raw[70:80], c3.raw[80:90], c3.raw[90:100]]
            vx = _fv(f4[0], 1.0) if len(f4) > 0 else 1.0
            vy = _fv(f4[1], 0.0) if len(f4) > 1 else 0.0
            vz = _fv(f4[2], 0.0) if len(f4) > 2 else 0.0
            skew_id = _iv(f4[3], 0) if len(f4) > 3 else 0
            ipos = _iv(f4[4], 0) if len(f4) > 4 else 0
            ip = _iv(f4[6], 0) if len(f4) > 6 else 0

        # Layer Cards: PROP_TYPE16_5
        for card in valid_cards[4:]:
            if "PROP_TYPE16_5" in CARD_LAYOUTS:
                fl = card.cut("PROP_TYPE16_5")
            else:
                fl = [card.raw[0:20], card.raw[20:40], card.raw[40:60], card.raw[60:80], card.raw[80:90]]
            if fl and any(x.strip() for x in fl):
                phi = _fv(fl[0], 0.0) if len(fl) > 0 else 0.0
                alpha = _fv(fl[1], 90.0) if len(fl) > 1 and fl[1].strip() else 90.0
                thick_i = _fv(fl[2], 0.0) if len(fl) > 2 else 0.0
                z_i = _fv(fl[3], 0.0) if len(fl) > 3 else 0.0
                mat_i = _iv(fl[4], 0) if len(fl) > 4 else 0
                layers.append(
                    PropType16Layer(
                        mat_id=mat_i,
                        thick=thick_i,
                        angle=phi,
                        z_pos=z_i,
                        weight=thick_i,
                        alpha=alpha,
                    )
                )
    else:
        # Free format tokens
        toks1 = valid_cards[0].tokens()
        ishell = _iv(toks1[0], 24) if len(toks1) > 0 else 24
        ismstr = _iv(toks1[1], 4) if len(toks1) > 1 else 4
        ish3n = _iv(toks1[2], 0) if len(toks1) > 2 else 0
        p_thick_fail = _fv(toks1[3], 0.0) if len(toks1) > 3 else 0.0

        if len(valid_cards) > 1:
            toks2 = valid_cards[1].tokens()
            hm = _fv(toks2[0], 0.01) if len(toks2) > 0 else 0.01
            hf = _fv(toks2[1], 0.01) if len(toks2) > 1 else 0.01
            hr = _fv(toks2[2], 0.01) if len(toks2) > 2 else 0.01
            dm = _fv(toks2[3], 0.0) if len(toks2) > 3 else 0.0
            dn = _fv(toks2[4], 0.0) if len(toks2) > 4 else 0.0

        if len(valid_cards) > 2:
            toks3 = valid_cards[2].tokens()
            nip = _iv(toks3[0], 0) if len(toks3) > 0 else 0
            istrain = _iv(toks3[1], 1) if len(toks3) > 1 else 1
            thick = _fv(toks3[2], 0.0) if len(toks3) > 2 else 0.0
            ashear = _fv(toks3[3], 5.0 / 6.0) if len(toks3) > 3 else 5.0 / 6.0
            ithick = _iv(toks3[4], 0) if len(toks3) > 4 else 0

        if len(valid_cards) > 3:
            toks4 = valid_cards[3].tokens()
            vx = _fv(toks4[0], 1.0) if len(toks4) > 0 else 1.0
            vy = _fv(toks4[1], 0.0) if len(toks4) > 1 else 0.0
            vz = _fv(toks4[2], 0.0) if len(toks4) > 2 else 0.0
            skew_id = _iv(toks4[3], 0) if len(toks4) > 3 else 0
            ipos = _iv(toks4[4], 0) if len(toks4) > 4 else 0
            ip = _iv(toks4[5], 0) if len(toks4) > 5 else 0

        for card in valid_cards[4:]:
            tl = card.tokens()
            if tl:
                phi = _fv(tl[0], 0.0) if len(tl) > 0 else 0.0
                alpha = _fv(tl[1], 90.0) if len(tl) > 1 else 90.0
                thick_i = _fv(tl[2], 0.0) if len(tl) > 2 else 0.0
                z_i = _fv(tl[3], 0.0) if len(tl) > 3 else 0.0
                mat_i = _iv(tl[4], 0) if len(tl) > 4 else 0
                layers.append(
                    PropType16Layer(
                        mat_id=mat_i,
                        thick=thick_i,
                        angle=phi,
                        z_pos=z_i,
                        weight=thick_i,
                        alpha=alpha,
                    )
                )

    prop = PropType16(
        prop_id=prop_id,
        title=title,
        ishell=ishell,
        ismstr=ismstr,
        ithick=ithick,
        nply=nip or len(layers),
        total_thickness=thick,
        layers=layers,
        ashear=ashear,
        ish3n=ish3n,
        p_thick_fail=p_thick_fail,
        hm=hm,
        hf=hf,
        hr=hr,
        dm=dm,
        dn=dn,
        istrain=istrain,
        vx=vx,
        vy=vy,
        vz=vz,
        skew_id=skew_id,
        ipos=ipos,
        ip=ip,
    )

    # Perform weight consistency verification
    if not prop.validate_weights():
        if log is not None:
            log.warning(
                f"/PROP/TYPE16/{prop_id}: sum of layer weights ({sum(l.weight for l in prop.layers)}) "
                f"does not match total thickness ({prop.total_thickness})",
                getattr(block, "source", ""),
            )

    return prop


parse_prop_type16 = parse_prop16
