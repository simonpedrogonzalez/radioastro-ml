"""Compatibility imports for :mod:`scripts.corruption.tables`."""

from .corruption.tables import (
    FilterSpec,
    GCOLS,
    GTab,
    GTabQuery,
    get_unflagged_antennas,
    make_corrtab_identity,
    make_template_gain_corrtab,
    verify_corrtab_is_identity,
)

__all__ = [
    "FilterSpec",
    "GCOLS",
    "GTab",
    "GTabQuery",
    "get_unflagged_antennas",
    "make_corrtab_identity",
    "make_template_gain_corrtab",
    "verify_corrtab_is_identity",
]
