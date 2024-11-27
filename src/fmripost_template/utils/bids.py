# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Utilities to handle BIDS inputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from bids.layout import BIDSLayout
from bids.utils import listify
from niworkflows.utils.spaces import SpatialReferences

from fmripost_template.data import load as load_data


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
    entities: dict | None,
    fieldmap_id: str | None,
    spec: dict | None = None,
    patterns: list[str] | None = None,
    allow_multiple: bool = False,
    spaces: SpatialReferences | None = None,
) -> dict:
    """Gather existing derivatives and compose a cache.

    TODO: Ingress 'spaces' and search for BOLD+mask in the spaces *or* xfms.

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
    spaces : SpatialReferences | None
        Spatial references to select for.

    Returns
    -------
    derivs_cache : dict
        Dictionary with keys corresponding to the derivatives and values
        corresponding to the file paths.
    """
    if not entities:
        entities = {}

    _spec = None
    if spec is None or patterns is None:
        _spec = json.loads(load_data.readable('io_spec.json').read_text())

        if spec is None:
            spec = _spec['queries']

        if patterns is None:
            patterns = _spec['default_path_patterns']

        _spec.pop('queries')

    config = ['bids', 'derivatives']
    if _spec:
        config = ['bids', 'derivatives', _spec]

    # Search for derivatives data
    derivs_cache = defaultdict(list, {})
    if derivatives_dataset is not None:
        layout = derivatives_dataset
        if isinstance(layout, Path):
            layout = BIDSLayout(
                layout,
                config=config,
                validate=False,
            )

        for k, q in spec['derivatives'].items():
            if k.startswith('anat'):
                # Allow anatomical derivatives at session level or subject level
                query = {**{'session': [entities.get('session'), None]}, **q}
            else:
                # Combine entities with query. Query values override file entities.
                query = {**entities, **q}

            item = layout.get(return_type='filename', **query)
            if not item:
                derivs_cache[k] = None
            elif not allow_multiple and len(item) > 1 and k.startswith('anat'):
                # Anatomical derivatives are allowed to have multiple files (e.g., T1w and T2w)
                # but we just grab the first one
                derivs_cache[k] = item[0]
            elif not allow_multiple and len(item) > 1:
                raise ValueError(f'Multiple files found for {k}: {item}')
            else:
                derivs_cache[k] = item[0] if len(item) == 1 else item

        for k, q in spec['transforms'].items():
            if k.startswith('anat'):
                # Allow anatomical derivatives at session level or subject level
                query = {**{'session': [entities.get('session'), None]}, **q}
            else:
                # Combine entities with query. Query values override file entities.
                query = {**entities, **q}

            if k == 'boldref2fmap':
                query['to'] = fieldmap_id

            item = layout.get(return_type='filename', **query)
            if not item:
                derivs_cache[k] = None
            elif not allow_multiple and len(item) > 1 and k.startswith('anat'):
                # Anatomical derivatives are allowed to have multiple files (e.g., T1w and T2w)
                # but we just grab the first one
                derivs_cache[k] = item[0]
            elif not allow_multiple and len(item) > 1:
                raise ValueError(f'Multiple files found for {k}: {item}')
            else:
                derivs_cache[k] = item[0] if len(item) == 1 else item

    # Search for requested output spaces
    if spaces is not None:
        # Put the output-space files/transforms in lists so they can be parallelized with
        # template_iterator_wf.
        spaces_found, bold_outputspaces, bold_mask_outputspaces = [], [], []
        for space in spaces.references:
            # First try to find processed BOLD+mask files in the requested space
            bold_query = {**entities, **spec['derivatives']['bold_mni152nlin6asym']}
            bold_query['space'] = space.space
            bold_query = {**bold_query, **space.spec}
            bold_item = layout.get(return_type='filename', **bold_query)
            bold_outputspaces.append(bold_item[0] if bold_item else None)

            mask_query = {**entities, **spec['derivatives']['bold_mask_mni152nlin6asym']}
            mask_query['space'] = space.space
            mask_query = {**mask_query, **space.spec}
            mask_item = layout.get(return_type='filename', **mask_query)
            bold_mask_outputspaces.append(mask_item[0] if mask_item else None)

            spaces_found.append(bool(bold_item) and bool(mask_item))

        if all(spaces_found):
            derivs_cache['bold_outputspaces'] = bold_outputspaces
            derivs_cache['bold_mask_outputspaces'] = bold_mask_outputspaces
        else:
            # The requested spaces were not found, try to find transforms
            print(
                'Not all requested output spaces were found. '
                'We will try to find transforms to these spaces and apply them to the BOLD data.',
                flush=True,
            )

        spaces_found, anat2outputspaces_xfm = [], []
        for space in spaces.references:
            # Now try to find transform to the requested space
            anat2space_query = {
                **{'session': [entities.get('session'), None]},
                **spec['transforms']['anat2mni152nlin6asym'],
            }
            anat2space_query['to'] = space.space
            item = layout.get(return_type='filename', **anat2space_query)
            anat2outputspaces_xfm.append(item[0] if item else None)
            spaces_found.append(bool(item))

        if all(spaces_found):
            derivs_cache['anat2outputspaces_xfm'] = anat2outputspaces_xfm
        else:
            missing_spaces = ', '.join(
                [
                    s.space
                    for s, found in zip(spaces.references, spaces_found, strict=False)
                    if not found
                ]
            )
            raise ValueError(
                f'Transforms to the following requested spaces not found: {missing_spaces}.'
            )

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


def write_bidsignore(deriv_dir):
    bids_ignore = (
        '*.html',
        'logs/',
        'figures/',  # Reports
        '*_xfm.*',  # Unspecified transform files
        '*.surf.gii',  # Unspecified structural outputs
        # Unspecified functional outputs
        '*_boldref.nii.gz',
        '*_bold.func.gii',
        '*_mixing.tsv',
        '*_timeseries.tsv',
    )
    ignore_file = Path(deriv_dir) / '.bidsignore'

    ignore_file.write_text('\n'.join(bids_ignore) + '\n')


def write_derivative_description(input_dir, output_dir, dataset_links=None):
    """Write dataset_description.json file for derivatives.

    Parameters
    ----------
    input_dir : :obj:`str`
        Path to the primary input dataset being ingested.
        This may be a raw BIDS dataset (in the case of raw+derivatives workflows)
        or a preprocessing derivatives dataset (in the case of derivatives-only workflows).
    output_dir : :obj:`str`
        Path to the output xcp-d dataset.
    dataset_links : :obj:`dict`, optional
        Dictionary of dataset links to include in the dataset description.
    """
    import json
    import os

    from packaging.version import Version

    from fmripost_template import __version__

    DOWNLOAD_URL = f'https://github.com/nipreps/fmripost_template/archive/{__version__}.tar.gz'

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    orig_dset_description = os.path.join(input_dir, 'dataset_description.json')
    if not os.path.isfile(orig_dset_description):
        raise FileNotFoundError(f'Dataset description does not exist: {orig_dset_description}')

    with open(orig_dset_description) as fobj:
        desc = json.load(fobj)

    # Update dataset description
    desc['Name'] = 'fMRIPost-template- ICA-template Postprocessing Outputs'
    desc['BIDSVersion'] = '1.9.0dev'
    desc['DatasetType'] = 'derivative'
    desc['HowToAcknowledge'] = 'Include the generated boilerplate in the methods section.'

    # Start with GeneratedBy from the primary input dataset's dataset_description.json
    desc['GeneratedBy'] = desc.get('GeneratedBy', [])

    # Add GeneratedBy from derivatives' dataset_description.jsons
    for name, link in enumerate(dataset_links):
        if name not in ('templateflow', 'input'):
            dataset_desc = Path(link) / 'dataset_description.json'
            if dataset_desc.is_file():
                with open(dataset_desc) as fobj:
                    dataset_desc_dict = json.load(fobj)

                if 'GeneratedBy' in dataset_desc_dict:
                    desc['GeneratedBy'].insert(0, dataset_desc_dict['GeneratedBy'][0])

    # Add GeneratedBy from fMRIPost-template
    desc['GeneratedBy'].insert(
        0,
        {
            'Name': 'fMRIPost-template',
            'Version': __version__,
            'CodeURL': DOWNLOAD_URL,
        },
    )

    # Keys that can only be set by environment
    if 'FMRIPOST_TEMPLATE_DOCKER_TAG' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'docker',
            'Tag': f'nipreps/fmripost_template:{os.environ["FMRIPOST_TEMPLATE_DOCKER_TAG"]}',
        }

    if 'FMRIPOST_TEMPLATE_SINGULARITY_URL' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'singularity',
            'URI': os.getenv('FMRIPOST_TEMPLATE_SINGULARITY_URL'),
        }

    # Replace local templateflow path with URL
    dataset_links = dataset_links.copy()
    dataset_links['templateflow'] = 'https://github.com/templateflow/templateflow'

    # Add DatasetLinks
    desc['DatasetLinks'] = desc.get('DatasetLinks', {})
    for k, v in dataset_links.items():
        if k in desc['DatasetLinks'].keys() and str(desc['DatasetLinks'][k]) != str(v):
            print(f'"{k}" is already a dataset link. Overwriting.')

        desc['DatasetLinks'][k] = str(v)

    out_desc = Path(output_dir / 'dataset_description.json')
    if out_desc.is_file():
        old_desc = json.loads(out_desc.read_text())
        old_version = old_desc['GeneratedBy'][0]['Version']
        if Version(__version__).public != Version(old_version).public:
            print(f'Previous output generated by version {old_version} found.')
    else:
        out_desc.write_text(json.dumps(desc, indent=4))


def validate_input_dir(exec_env, bids_dir, participant_label, need_T1w=True):
    # Ignore issues and warnings that should not influence FMRIPREP
    import subprocess
    import sys
    import tempfile

    validator_config_dict = {
        'ignore': [
            'EVENTS_COLUMN_ONSET',
            'EVENTS_COLUMN_DURATION',
            'TSV_EQUAL_ROWS',
            'TSV_EMPTY_CELL',
            'TSV_IMPROPER_NA',
            'VOLUME_COUNT_MISMATCH',
            'BVAL_MULTIPLE_ROWS',
            'BVEC_NUMBER_ROWS',
            'DWI_MISSING_BVAL',
            'INCONSISTENT_SUBJECTS',
            'INCONSISTENT_PARAMETERS',
            'BVEC_ROW_LENGTH',
            'B_FILE',
            'PARTICIPANT_ID_COLUMN',
            'PARTICIPANT_ID_MISMATCH',
            'TASK_NAME_MUST_DEFINE',
            'PHENOTYPE_SUBJECTS_MISSING',
            'STIMULUS_FILE_MISSING',
            'DWI_MISSING_BVEC',
            'EVENTS_TSV_MISSING',
            'TSV_IMPROPER_NA',
            'ACQTIME_FMT',
            'Participants age 89 or higher',
            'DATASET_DESCRIPTION_JSON_MISSING',
            'FILENAME_COLUMN',
            'WRONG_NEW_LINE',
            'MISSING_TSV_COLUMN_CHANNELS',
            'MISSING_TSV_COLUMN_IEEG_CHANNELS',
            'MISSING_TSV_COLUMN_IEEG_ELECTRODES',
            'UNUSED_STIMULUS',
            'CHANNELS_COLUMN_SFREQ',
            'CHANNELS_COLUMN_LOWCUT',
            'CHANNELS_COLUMN_HIGHCUT',
            'CHANNELS_COLUMN_NOTCH',
            'CUSTOM_COLUMN_WITHOUT_DESCRIPTION',
            'ACQTIME_FMT',
            'SUSPICIOUSLY_LONG_EVENT_DESIGN',
            'SUSPICIOUSLY_SHORT_EVENT_DESIGN',
            'MALFORMED_BVEC',
            'MALFORMED_BVAL',
            'MISSING_TSV_COLUMN_EEG_ELECTRODES',
            'MISSING_SESSION',
        ],
        'error': ['NO_T1W'] if need_T1w else [],
        'ignoredFiles': ['/dataset_description.json', '/participants.tsv'],
    }
    # Limit validation only to data from requested participants
    if participant_label:
        all_subs = {s.name[4:] for s in bids_dir.glob('sub-*')}
        selected_subs = {s[4:] if s.startswith('sub-') else s for s in participant_label}
        bad_labels = selected_subs.difference(all_subs)
        if bad_labels:
            error_msg = (
                'Data for requested participant(s) label(s) not found. Could '
                'not find data for participant(s): %s. Please verify the requested '
                'participant labels.'
            )
            if exec_env == 'docker':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the docker container. Please make sure all '
                    'volumes are mounted properly (see https://docs.docker.com/'
                    'engine/reference/commandline/run/#mount-volume--v---read-only)'
                )
            if exec_env == 'singularity':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the singularity container. Please make sure '
                    'all paths are mapped properly (see https://www.sylabs.io/'
                    'guides/3.0/user-guide/bind_paths_and_mounts.html)'
                )
            raise RuntimeError(error_msg % ','.join(bad_labels))

        ignored_subs = all_subs.difference(selected_subs)
        if ignored_subs:
            for sub in ignored_subs:
                validator_config_dict['ignoredFiles'].append(f'/sub-{sub}/**')
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as temp:
        temp.write(json.dumps(validator_config_dict))
        temp.flush()
        try:
            subprocess.check_call(['bids-validator', str(bids_dir), '-c', temp.name])  # noqa: S607
        except FileNotFoundError:
            print('bids-validator does not appear to be installed', file=sys.stderr)
