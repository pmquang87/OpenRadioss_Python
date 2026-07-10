"""
pyradioss.contact — contact interfaces.

Fortran origin: ``engine/source/interfaces/`` — TYPE7 (the workhorse
penalty node-to-surface interface) lives in ``inter3d/i7*.F`` with its
bucket search in ``intsort/i7buce.F``.
"""

from .inter_type7 import ContactType7  # noqa: F401
