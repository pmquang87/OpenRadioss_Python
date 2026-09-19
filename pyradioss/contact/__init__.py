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

from .inter_type1 import ContactType1   # noqa: F401
from .inter_type2 import ContactType2   # noqa: F401
from .inter_type3 import ContactType3   # noqa: F401
from .inter_type5 import ContactType5   # noqa: F401
from .inter_type6 import ContactType6   # noqa: F401
from .inter_type7 import ContactType7   # noqa: F401
from .inter_type8 import ContactType8   # noqa: F401
from .inter_type9 import ContactType9   # noqa: F401
from .inter_type11 import ContactType11  # noqa: F401
from .inter_type18 import ContactType18  # noqa: F401
from .inter_type24 import ContactType24  # noqa: F401
from .inter_type10 import ContactType10 # noqa: F401
from .inter_type12 import ContactType12 # noqa: F401
from .inter_type14 import ContactType14 # noqa: F401
from .inter_type15 import ContactType15 # noqa: F401
from .inter_type20 import ContactType20 # noqa: F401
from .inter_type21 import ContactType21 # noqa: F401
from .inter_type23 import ContactType23 # noqa: F401
from .inter_type16 import ContactType16 # noqa: F401
from .inter_type17 import ContactType17 # noqa: F401
from .inter_type22 import ContactType22 # noqa: F401
from .inter_type25 import ContactType25 # noqa: F401
from .inter_guided_cable import ContactGuidedCable # noqa: F401


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
        if itf.type == 1:
            penalty.append(ContactType1(itf, model, log))
        elif itf.type == 3:
            penalty.append(ContactType3(itf, model, log))
        elif itf.type == 5:
            penalty.append(ContactType5(itf, model, log))
        elif itf.type == 6:
            penalty.append(ContactType6(itf, model, log))
        elif itf.type == 7:
            penalty.append(ContactType7(itf, model, log))
        elif itf.type == 8:
            penalty.append(ContactType8(itf, model, log))
        elif itf.type == 9:
            penalty.append(ContactType9(itf, model, log))
        elif itf.type == 11:
            penalty.append(ContactType11(itf, model, log))
        elif itf.type == 12:
            penalty.append(ContactType12(itf, model, log))
        elif itf.type == 14:
            penalty.append(ContactType14(itf, model, log))
        elif itf.type == 15:
            penalty.append(ContactType15(itf, model, log))
        elif itf.type == 16:
            penalty.append(ContactType16(itf, model, log))
        elif itf.type == 17:
            penalty.append(ContactType17(itf, model, log))
        elif itf.type == 19:
            penalty.append(ContactType7(itf, model, log))
            if getattr(itf, 'line_id1', 0) > 0 and getattr(itf, 'line_id2', 0) > 0:
                penalty.append(ContactType11(itf, model, log))
        elif itf.type == 2:
            tied.append(ContactType2(itf, model, log))
        elif itf.type == 20:
            penalty.append(ContactType20(itf, model, log))
        elif itf.type == 21:
            penalty.append(ContactType21(itf, model, log))
        elif itf.type == 22:
            penalty.append(ContactType22(itf, model, log))
        elif itf.type == 23:
            penalty.append(ContactType23(itf, model, log))
        elif itf.type == 24:
            penalty.append(ContactType24(itf, model, log))
        elif itf.type == 25:
            penalty.append(ContactType25(itf, model, log))
        elif itf.type in (26, 29):
            penalty.append(ContactGuidedCable(itf, model, log))
        elif itf.type == 18:
            penalty.append(ContactType18(itf, model, log))
        elif itf.type == 10:
            penalty.append(ContactType10(itf, model, log))

    for gc in getattr(model, "guided_cables", {}).values():
        penalty.append(ContactGuidedCable(gc, model, log))

    return penalty, tied


