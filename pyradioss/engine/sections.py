"""
/SECT — section-force output (M5).

Fortran origin: ``engine/source/tools/sect/`` (``section.F``,
``section_io.F``, ``forint.F``): the original tags the elements of one
side of the cut and accumulates their internal-force contributions at the
section nodes into the FSAV time-history blocks.

The port uses the equivalent (and fully vectorized) *side-sum* identity.
For any element, the assembled internal nodal forces are self-equilibrated
— they sum to zero force AND zero moment over the element's own nodes
(rigid translations and rotations do no internal work; this is the
partition-of-unity property every kernel in pyradioss/elements satisfies
by construction). Hence, summing the ASSEMBLED internal force array over
all nodes of one complete side of a cut cancels every element interior to
the side, leaving exactly the contributions of the OTHER side's elements
at the shared cut nodes:

    F_sect = sum_{n in side} fint_n
    M_sect = sum_{n in side} [(x_n - x_ref) x fint_n + mint_n]

This is the force (and moment about x_ref) that the excluded side
transmits to the included side through the cut — the section resultants.
Note the ledger sign convention: ``fint`` holds the force ON the nodes
(see the elements package doc), so a bar pulled in tension with the far
side excluded reports a POSITIVE force pointing away from the included
side — the pull it feels.

External loads, contact and inertia never enter the formula: they act on
the nodes directly, not *through* the cut. (This is why the momentum-flux
form  sum m a - f_ext  used in some textbooks is identical: subtracting
Newton's law per node leaves the same internal sum.)

The moment reference is either a NODE (its current position — the
reference then rides the deformation, which is what a load-path
engineer usually wants) or the fixed initial centroid of the side set.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..model.model import Model


class SectionForces:
    """All /SECT requests, engine-side. ``compute`` is called at every
    time-history write (the resultants are output quantities, not solver
    state)."""

    def __init__(self, model: Model, log):
        self.model = model
        self.sections = []          # (sect, side_idx, ref_node_or_None, x_ref0)
        for sc in model.sections:
            side = model.node_groups[sc.grnod_id].node_idx
            ref = model.node_index(sc.node_id_ref) if sc.node_id_ref else -1
            x_ref0 = (model.x0[side].mean(axis=0) if ref < 0
                      else None)
            self.sections.append((sc, side, ref, x_ref0))
            log.info(f"     /SECT/{sc.id}: SIDE SET OF {len(side)} "
                     f"NODE(S)")

    def __len__(self):
        return len(self.sections)

    @property
    def ids(self) -> List[int]:
        return [sc.id for sc, _, _, _ in self.sections]

    # ------------------------------------------------------------------
    def compute(self, x: np.ndarray, fint: np.ndarray,
                mint: np.ndarray) -> dict:
        """Section resultants from the current assembled internal forces.
        Returns {sect_id: (F (3,), M (3,))}."""
        out = {}
        for sc, side, ref, x_ref0 in self.sections:
            x_ref = x[ref] if ref >= 0 else x_ref0
            f = fint[side]
            F = f.sum(axis=0)
            M = (np.cross(x[side] - x_ref, f).sum(axis=0)
                 + mint[side].sum(axis=0))
            out[sc.id] = (F, M)
        return out
