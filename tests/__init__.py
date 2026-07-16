"""Mark tests/ as a regular package (deliberately: do not delete this file).

Twelve fatigue-era test modules (m22-m34) share deck-builder helpers via
``from tests.test_m21_multiaxfatig import _brick_deck``-style imports.
Without this file, ``tests`` is only a PEP 420 namespace package, and PEP 420
gives ANY regular ``tests`` package installed in site-packages (several
sloppily-packaged distributions ship one) priority over the repo directory --
collection then dies with ``ModuleNotFoundError: No module named
'tests.test_m21_multiaxfatig'``.  With this ``__init__.py`` the repo's
``tests`` is a regular package too, pytest inserts the repo root at the front
of ``sys.path`` (prepend import mode), and the repo package wins the import
race regardless of what is installed in the environment.
"""
