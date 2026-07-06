"""Sample-id inference and the canonical output filenames for the finalize pipeline."""
from __future__ import annotations

import os
import re
from typing import Optional

# exaSPIM sample ids are 6-7 digit numbers (e.g. 720164, 791116). Prefer a targeted match
# (before a known suffix, or after 'exaSPIM_') over a bare digit run so we don't grab a date.
_TARGETED = [
    re.compile(r"(\d{6,7})_(?:10um|manual|transformed|inverted|displacement)"),
    re.compile(r"exaSPIM[_-](\d{6,7})"),
]
_BARE = re.compile(r"(?<!\d)(\d{6,7})(?!\d)")


def infer_sample_id(*hints: Optional[str]) -> Optional[str]:
    """Return the sample id found in any hint (path/filename), or None."""
    cleaned = [str(h) for h in hints if h]
    for pat in _TARGETED:
        for h in cleaned:
            m = pat.search(os.path.basename(h)) or pat.search(h)
            if m:
                return m.group(1)
    for h in cleaned:  # last resort: first standalone 6-7 digit run in the basename
        m = _BARE.search(os.path.basename(h))
        if m:
            return m.group(1)
    return None


def output_names(sample_id: str) -> dict:
    """The four required artifact names for a sample."""
    sid = str(sample_id)
    return {
        "forward_field": f"{sid}_manual_displacement_field.nrrd",
        "inverse_field": f"{sid}_manual_inverse_displacement_field.nrrd",
        "transformed": f"{sid}_transformed_final.nii.gz",
        "inverted": f"{sid}_inverted_transform.nii.gz",
    }
