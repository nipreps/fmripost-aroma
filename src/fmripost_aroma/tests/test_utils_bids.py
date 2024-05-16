"""Lightweight tests for fmripost_aroma.utils.bids."""

import os
from pathlib import Path

from bids.layout import BIDSLayout, BIDSLayoutIndexer

from fmripost_aroma.tests.utils import get_test_data_path
from fmripost_aroma.utils import bids as xbids


def test_collect_derivatives_raw(base_ignore_list):
    """Test collect_derivatives with a raw dataset."""
    data_dir = get_test_data_path()

    raw_dataset = Path(data_dir) / 'ds000005-fmriprep' / 'sourcedata'
    raw_layout = BIDSLayout(
        raw_dataset,
        config=['bids'],
        validate=False,
        indexer=BIDSLayoutIndexer(validate=False, ignore=base_ignore_list),
    )

    subject_data = xbids.collect_derivatives(
        raw_dataset=raw_layout,
        derivatives_dataset=None,
        entities={'subject': '01', 'task': 'mixedgamblestask', 'run': 1},
        fieldmap_id=None,
        spec=None,
        patterns=None,
    )
    expected = {
        'bold_raw': 'sub-01_task-mixedgamblestask_run-01_bold.nii.gz',
    }
    check_expected(subject_data, expected)


def test_collect_derivatives_minimal(minimal_ignore_list):
    """Test collect_derivatives with a minimal-mode dataset."""

    data_dir = get_test_data_path()

    derivatives_dataset = Path(data_dir) / 'ds000005-fmriprep'
    derivatives_layout = BIDSLayout(
        derivatives_dataset,
        config=['bids', 'derivatives'],
        validate=False,
        indexer=BIDSLayoutIndexer(validate=False, ignore=minimal_ignore_list),
    )

    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=derivatives_layout,
        entities={'subject': '01', 'task': 'mixedgamblestask', 'run': 1},
        fieldmap_id=None,
        spec=None,
        patterns=None,
    )
    expected = {
        'bold_std': None,
        'bold_mask_std': None,
        'bold_mask': 'sub-01_task-mixedgamblestask_run-01_desc-brain_mask.nii.gz',
        'confounds': None,
        'hmc': 'sub-01_task-mixedgamblestask_run-01_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt',
        'boldref2anat': 'sub-01_task-mixedgamblestask_run-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt',
        'boldref2fmap': None,
        'anat2mni152nlin6asym': 'sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5',
    }
    check_expected(subject_data, expected)


def test_collect_derivatives_full(full_ignore_list):
    """Test collect_derivatives with a full-mode dataset."""

    data_dir = get_test_data_path()

    derivatives_dataset = Path(data_dir) / 'ds000005-fmriprep'
    derivatives_layout = BIDSLayout(
        derivatives_dataset,
        config=['bids', 'derivatives'],
        validate=False,
        indexer=BIDSLayoutIndexer(validate=False, ignore=full_ignore_list),
    )

    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=derivatives_layout,
        entities={'subject': '01', 'task': 'mixedgamblestask', 'run': 1},
        fieldmap_id=None,
        spec=None,
        patterns=None,
    )
    expected = {
        'bold_std': 'sub-01_task-mixedgamblestask_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
        'bold_mask_std': 'sub-01_task-mixedgamblestask_run-01_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz',
        'bold_mask': None,
        'confounds': 'sub-01_task-mixedgamblestask_run-01_desc-confounds_timeseries.tsv',
        'hmc': 'sub-01_task-mixedgamblestask_run-01_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt',
        'boldref2anat': 'sub-01_task-mixedgamblestask_run-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt',
        'boldref2fmap': None,
        'anat2mni152nlin6asym': 'sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5',
    }
    check_expected(subject_data, expected)


def check_expected(subject_data, expected):
    """Check expected values."""
    for key, value in expected.items():
        if isinstance(value, str):
            assert subject_data[key] is not None
            assert os.path.basename(subject_data[key]) == value
        else:
            assert subject_data[key] is value
