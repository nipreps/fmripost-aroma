"""Utilities to handle BIDS inputs."""

from __future__ import annotations

import json
import typing as ty
from collections import defaultdict
from pathlib import Path

from bids.layout import BIDSLayout

from fmripost_aroma.data import load as load_data


def collect_derivatives(
    raw_dir: Path | None,
    derivatives_dir: Path,
    entities: dict,
    fieldmap_id: str | None,
    spec: dict | None = None,
    patterns: ty.List[str] | None = None,
):
    """Gather existing derivatives and compose a cache."""
    if spec is None or patterns is None:
        _spec, _patterns = tuple(
            json.loads(load_data.readable("io_spec.json").read_text()).values()
        )

        if spec is None:
            spec = _spec
        if patterns is None:
            patterns = _patterns

    derivs_cache = defaultdict(list, {})

    layout = BIDSLayout(derivatives_dir, config=["bids", "derivatives"], validate=False)
    derivatives_dir = Path(derivatives_dir)

    # Search for preprocessed BOLD data
    for k, q in spec["baseline"]["derivatives"].items():
        query = {**q, **entities}
        item = layout.get(return_type="filename", **query)
        if not item:
            continue
        derivs_cache[k] = item[0] if len(item) == 1 else item

    # Search for raw BOLD data
    if not derivs_cache and raw_dir is not None:
        raw_layout = BIDSLayout(raw_dir, config=["bids"], validate=False)
        raw_dir = Path(raw_dir)

        for k, q in spec["baseline"]["raw"].items():
            query = {**q, **entities}
            item = raw_layout.get(return_type="filename", **query)
            if not item:
                continue
            derivs_cache[k] = item[0] if len(item) == 1 else item

    for xfm, q in spec["transforms"].items():
        query = {**q, **entities}
        if xfm == "boldref2fmap":
            query["to"] = fieldmap_id
        item = layout.get(return_type="filename", **q)
        if not item:
            continue
        derivs_cache[xfm] = item[0] if len(item) == 1 else item
    return derivs_cache


def collect_derivatives_old(
    layout,
    subject_id,
    task_id=None,
    bids_filters=None,
):
    """Collect preprocessing derivatives."""
    subj_data = {
        "bold_raw": "",
        "bold_boldref": "",
        "bold_MNI152NLin6": "",
    }
    query = {
        "bold": {
            "space": "MNI152NLin6Asym",
            "res": 2,
            "desc": "preproc",
            "suffix": "bold",
            "extension": [".nii", ".nii.gz"],
        }
    }
    subj_data = layout.get(subject=subject_id, **query)
    return subj_data


def collect_run_data(
    layout,
    bold_file,
):
    """Collect files and metadata related to a given BOLD file."""
    queries = {}
    run_data = {
        "mask": {"desc": "brain", "suffix": "mask", "extension": [".nii", ".nii.gz"]},
        "confounds": {"desc": "confounds", "suffix": "timeseries", "extension": ".tsv"},
    }
    for k, v in queries.items():
        run_data[k] = layout.get_nearest(bold_file, **v)

    return run_data
