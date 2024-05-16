"""Utilities to handle BIDS inputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from bids.layout import BIDSLayout
from bids.utils import listify

from fmripost_aroma.data import load as load_data


def extract_entities(file_list: str | list[str]) -> dict:
    """Return a dictionary of common entities given a list of files.

    Parameters
    ----------
    file_list : str | list[str]
        File path or list of file paths.

    Returns
    -------
    entities : dict
        Dictionary of entities.

    Examples
    --------
    >>> extract_entities("sub-01/anat/sub-01_T1w.nii.gz")
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_T1w.nii.gz"] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_run-1_T1w.nii.gz",
    ...                   "sub-01/anat/sub-01_run-2_T1w.nii.gz"])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}

    """
    from collections import defaultdict

    from bids.layout import parse_file_entities

    entities = defaultdict(list)
    for e, v in [
        ev_pair for f in listify(file_list) for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist

    return {k: _unique(v) for k, v in entities.items()}


def collect_derivatives(
    raw_dataset: Path | BIDSLayout | None,
    derivatives_dataset: Path | BIDSLayout | None,
    entities: dict,
    fieldmap_id: str | None,
    spec: dict | None = None,
    patterns: list[str] | None = None,
    allow_multiple: bool = False,
) -> dict:
    """Gather existing derivatives and compose a cache.

    Parameters
    ----------
    raw_dataset : Path | BIDSLayout | None
        Path to the raw dataset or a BIDSLayout instance.
    derivatives_dataset : Path | BIDSLayout
        Path to the derivatives dataset or a BIDSLayout instance.
    entities : dict
        Dictionary of entities to use for filtering.
    fieldmap_id : str | None
        Fieldmap ID to use for filtering.
    spec : dict | None
        Specification dictionary.
    patterns : list[str] | None
        List of patterns to use for filtering.
    allow_multiple : bool
        Allow multiple files to be returned for a given query.

    Returns
    -------
    derivs_cache : dict
        Dictionary with keys corresponding to the derivatives and values
        corresponding to the file paths.
    """
    if spec is None or patterns is None:
        _spec = json.loads(load_data.readable('io_spec.json').read_text())

        if spec is None:
            spec = _spec['queries']

        if patterns is None:
            patterns = _spec['patterns']

    # Search for derivatives data
    derivs_cache = defaultdict(list, {})
    if derivatives_dataset is not None:
        layout = derivatives_dataset
        if isinstance(derivatives_dataset, Path):
            derivatives_dataset = BIDSLayout(
                derivatives_dataset,
                config=['bids', 'derivatives'],
                validate=False,
            )

        for k, q in spec['derivatives'].items():
            # Combine entities with query. Query values override file entities.
            query = {**entities, **q}
            print(query)
            item = layout.get(return_type='filename', **query)
            if not item:
                derivs_cache[k] = None
            elif not allow_multiple and len(item) > 1:
                raise ValueError(f'Multiple files found for {k}: {item}')
            else:
                derivs_cache[k] = item[0] if len(item) == 1 else item

        for k, q in spec['transforms'].items():
            # Combine entities with query. Query values override file entities.
            # TODO: Drop functional entities (task, run, etc.) from anat transforms.
            query = {**entities, **q}
            if k == 'boldref2fmap':
                query['to'] = fieldmap_id

            item = layout.get(return_type='filename', **query)
            if not item:
                derivs_cache[k] = None
            elif not allow_multiple and len(item) > 1:
                raise ValueError(f'Multiple files found for {k}: {item}')
            else:
                derivs_cache[k] = item[0] if len(item) == 1 else item

    # Search for raw BOLD data
    if not derivs_cache and raw_dataset is not None:
        if isinstance(raw_dataset, Path):
            raw_layout = BIDSLayout(raw_dataset, config=['bids'], validate=False)
        else:
            raw_layout = raw_dataset

        for k, q in spec['raw'].items():
            # Combine entities with query. Query values override file entities.
            query = {**entities, **q}
            item = raw_layout.get(return_type='filename', **query)
            if not item:
                derivs_cache[k] = None
            elif not allow_multiple and len(item) > 1:
                raise ValueError(f'Multiple files found for {k}: {item}')
            else:
                derivs_cache[k] = item[0] if len(item) == 1 else item

    return derivs_cache


def collect_derivatives_old(
    layout,
    subject_id,
    task_id=None,
    bids_filters=None,
):
    """Collect preprocessing derivatives."""
    subj_data = {
        'bold_raw': '',
        'bold_boldref': '',
        'bold_MNI152NLin6': '',
    }
    query = {
        'bold': {
            'space': 'MNI152NLin6Asym',
            'res': 2,
            'desc': 'preproc',
            'suffix': 'bold',
            'extension': ['.nii', '.nii.gz'],
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
        'mask': {'desc': 'brain', 'suffix': 'mask', 'extension': ['.nii', '.nii.gz']},
        'confounds': {'desc': 'confounds', 'suffix': 'timeseries', 'extension': '.tsv'},
    }
    for k, v in queries.items():
        run_data[k] = layout.get_nearest(bold_file, **v)

    return run_data
