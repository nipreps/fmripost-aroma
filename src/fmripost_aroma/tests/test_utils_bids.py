"""Lightweight tests for fmripost_aroma.utils.bids."""

import os

import pytest
from bids.layout import BIDSLayout, BIDSLayoutIndexer

from fmripost_aroma.tests.utils import get_test_data_path
from fmripost_aroma.utils import bids as xbids


def test_collect_derivatives_raw(base_ignore_list):
    """Test collect_derivatives with a raw dataset."""
    data_dir = get_test_data_path()

    raw_dataset = data_dir / 'ds000005-fmriprep' / 'sourcedata' / 'raw'
    raw_layout = BIDSLayout(
        raw_dataset,
        config=['bids'],
        validate=False,
        indexer=BIDSLayoutIndexer(validate=False, index_metadata=False, ignore=base_ignore_list),
    )

    subject_data = xbids.collect_derivatives(
        raw_dataset=raw_layout,
        derivatives_dataset=None,
        entities={'subject': '01', 'task': 'mixedgamblestask'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=True,
    )
    expected = {
        'bold_raw': [
            'sub-01_task-mixedgamblestask_run-01_bold.nii.gz',
            'sub-01_task-mixedgamblestask_run-02_bold.nii.gz',
        ],
    }
    check_expected(subject_data, expected)

    with pytest.raises(ValueError, match='Multiple files found'):
        xbids.collect_derivatives(
            raw_dataset=raw_layout,
            derivatives_dataset=None,
            entities={'subject': '01', 'task': 'mixedgamblestask'},
            fieldmap_id=None,
            spec=None,
            patterns=None,
            allow_multiple=False,
        )


def test_collect_derivatives_minimal(minimal_ignore_list):
    """Test collect_derivatives with a minimal-mode dataset."""
    data_dir = get_test_data_path()

    derivatives_dataset = data_dir / 'ds000005-fmriprep'
    derivatives_layout = BIDSLayout(
        derivatives_dataset,
        config=['bids', 'derivatives'],
        validate=False,
        indexer=BIDSLayoutIndexer(
            validate=False,
            index_metadata=False,
            ignore=minimal_ignore_list,
        ),
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
        'bold_mni152nlin6asym': None,
        'bold_mask_mni152nlin6asym': None,
        # TODO: Add bold_mask_native to the dataset
        # 'bold_mask_native': 'sub-01_task-mixedgamblestask_run-01_desc-brain_mask.nii.gz',
        'bold_mask_native': None,
        'bold_confounds': None,
        'bold_hmc': (
            'sub-01_task-mixedgamblestask_run-01_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt'
        ),
        'boldref2anat': (
            'sub-01_task-mixedgamblestask_run-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt'
        ),
        'boldref2fmap': None,
        'anat2mni152nlin6asym': 'sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5',
    }
    check_expected(subject_data, expected)


def test_collect_derivatives_full(full_ignore_list):
    """Test collect_derivatives with a full-mode dataset."""
    data_dir = get_test_data_path()

    derivatives_dataset = data_dir / 'ds000005-fmriprep'
    derivatives_layout = BIDSLayout(
        derivatives_dataset,
        config=['bids', 'derivatives'],
        validate=False,
        indexer=BIDSLayoutIndexer(validate=False, index_metadata=False, ignore=full_ignore_list),
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
        'bold_mni152nlin6asym': (
            'sub-01_task-mixedgamblestask_run-01_space-MNI152NLin6Asym_res-2_desc-preproc_'
            'bold.nii.gz'
        ),
        'bold_mask_mni152nlin6asym': (
            'sub-01_task-mixedgamblestask_run-01_space-MNI152NLin6Asym_res-2_desc-brain_'
            'mask.nii.gz'
        ),
        'bold_mask_native': None,
        'bold_confounds': 'sub-01_task-mixedgamblestask_run-01_desc-confounds_timeseries.tsv',
        'bold_hmc': (
            'sub-01_task-mixedgamblestask_run-01_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt'
        ),
        'boldref2anat': (
            'sub-01_task-mixedgamblestask_run-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt'
        ),
        'boldref2fmap': None,
        'anat2mni152nlin6asym': 'sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5',
    }
    check_expected(subject_data, expected)


def check_expected(subject_data, expected):
    """Check expected values."""
    for key, value in expected.items():
        if isinstance(value, str):
            assert subject_data[key] is not None, f'Key {key} is None.'
            assert os.path.basename(subject_data[key]) == value
        elif isinstance(value, list):
            assert subject_data[key] is not None, f'Key {key} is None.'
            assert len(subject_data[key]) == len(value)
            for item, expected_item in zip(subject_data[key], value):
                assert os.path.basename(item) == expected_item
        else:
            assert subject_data[key] is value
