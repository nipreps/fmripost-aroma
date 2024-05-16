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
    entities: dict | None,
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
    if not entities:
        entities = {}

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


def write_derivative_description(bids_dir, deriv_dir, dataset_links=None):
    import os

    from fmripost_aroma import __version__

    DOWNLOAD_URL = f'https://github.com/nipreps/fmripost_aroma/archive/{__version__}.tar.gz'

    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    desc = {
        'Name': 'fMRIPost-AROMA- ICA-AROMA Postprocessing Outputs',
        'BIDSVersion': '1.9.0dev',
        'DatasetType': 'derivative',
        'GeneratedBy': [
            {
                'Name': 'fMRIPost-AROMA',
                'Version': __version__,
                'CodeURL': DOWNLOAD_URL,
            }
        ],
        'HowToAcknowledge': 'Please cite fMRIPost-AROMA when using these results.',
    }

    # Keys that can only be set by environment
    if 'FMRIPOST_AROMA_DOCKER_TAG' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'docker',
            'Tag': f"nipreps/fmriprep:{os.environ['FMRIPOST_AROMA__DOCKER_TAG']}",
        }
    if 'FMRIPOST_AROMA__SINGULARITY_URL' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'singularity',
            'URI': os.getenv('FMRIPOST_AROMA__SINGULARITY_URL'),
        }

    # Keys deriving from source dataset
    orig_desc = {}
    fname = bids_dir / 'dataset_description.json'
    if fname.exists():
        orig_desc = json.loads(fname.read_text())

    if 'DatasetDOI' in orig_desc:
        desc['SourceDatasets'] = [
            {'URL': f'https://doi.org/{orig_desc["DatasetDOI"]}', 'DOI': orig_desc['DatasetDOI']}
        ]
    if 'License' in orig_desc:
        desc['License'] = orig_desc['License']

    # Add DatasetLinks
    if dataset_links:
        desc['DatasetLinks'] = {k: str(v) for k, v in dataset_links.items()}
        if 'templateflow' in dataset_links:
            desc['DatasetLinks']['templateflow'] = 'https://github.com/templateflow/templateflow'

    Path.write_text(deriv_dir / 'dataset_description.json', json.dumps(desc, indent=4))


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
