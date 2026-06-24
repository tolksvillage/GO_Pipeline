"""
signature_utils.py

Unified detection of "original signature" vs. "derived dilution artifact".

Until now, different parts of the repository used slightly different checks
(e.g. name.startswith("diluted_")), and in at least one place
(dilute_signatures.py) there was NO check at all. This caused issues where
a second run with a different --mode would accidentally dilute already
diluted signatures again.

This module is now the SINGLE source of truth for this distinction.
Everywhere else in the code should import is_original_signature() /
extract_original_name() from here instead of implementing their own prefix checks.
"""

import re

DILUTED_PREFIX = "diluted_"

_STEP_SUFFIX_PATTERNS = [
    r"_step\d+_cumulative\d+_total\d+$",
    r"_step\d+_fixed\d+_total\d+$",
]


def is_original_signature(name: str) -> bool:
    """
    Returns True if 'name' (file name without .txt OR folder name) is an
    ORIGINAL user-provided signature, meaning it is NOT a dilution artifact
    created by dilute_signatures.py.
    """
    return not name.startswith(DILUTED_PREFIX)


def extract_original_name(name: str) -> str:
    """
    Extracts the base original signature name from a (possibly diluted) name.

    Examples:
        "diluted_MySig_step03_fixed300_total450"      -> "MySig"
        "diluted_MySig_step03_cumulative300_total450" -> "MySig"
        "MySig"                                        -> "MySig"
    """
    base = name
    if base.startswith(DILUTED_PREFIX):
        base = base[len(DILUTED_PREFIX):]

    for pattern in _STEP_SUFFIX_PATTERNS:
        match = re.search(pattern, base)
        if match:
            return base[: match.start()]

    return base


def get_dilution_mode(name: str):
    """Returns 'cumulative', 'fixed', or None depending on what the name contains."""
    if "cumulative" in name or "totalrandom" in name:
        return "cumulative"
    if "fixed" in name:
        return "fixed"
    return None