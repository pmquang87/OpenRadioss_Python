"""
pyradioss.contact — contact interfaces.

Fortran origin: ``engine/source/interfaces/`` —

* TYPE7, penalty node-to-surface (the workhorse): ``int07/i7*.F`` with
  the bucket search in ``intsort/i7buce.F`` → :mod:`.inter_type7`;
* TYPE2, tied (kinematic gluing): ``int02/i2*.F`` → :mod:`.inter_type2`;
* TYPE11, penalty edge-to-edge: ``int11/i11*.F`` → :mod:`.inter_type11`;
* Starter-side stiffness/gap setup ``inter3d1/i7sti3.F, i11sti3.F`` →
  :mod:`.stiffness`; deletion bookkeeping (IDEL) → :mod:`.tracking`.
"""

from .inter_type2 import ContactType2   # noqa: F401
from .inter_type7 import ContactType7   # noqa: F401
from .inter_type11 import ContactType11  # noqa: F401
from .inter_type18 import ContactType18  # noqa: F401
from .inter_type24 import ContactType24  # noqa: F401
from .inter_type10 import ContactType10 # noqa: F401


def build_contacts(model, log):
    """Instantiate the engine-side contact objects from the model's
    /INTER cards. Returns (penalty, tied): the penalty list (TYPE7 and
    TYPE11 — both expose ``forces()`` with the same contract) and the
    tied list (TYPE2 — kinematic, hooked differently into the cycle)."""
    penalty, tied = [], []
    for itf in model.interfaces:
        if getattr(itf, 'lagmul', False):
            log.warning(f"Engine logic for /INTER/LAGMUL/TYPE{itf.type} not implemented, ignoring", f"Interface {itf.id}")
            continue
        if itf.type == 7:
            penalty.append(ContactType7(itf, model, log))
        elif itf.type == 11:
            penalty.append(ContactType11(itf, model, log))
        elif itf.type == 2:
            tied.append(ContactType2(itf, model, log))
        elif itf.type == 24:
            penalty.append(ContactType24(itf, model, log))
        elif itf.type == 18:
            penalty.append(ContactType18(itf, model, log))
        elif itf.type == 10:
            penalty.append(ContactType10(itf, model, log))
    return penalty, tied
