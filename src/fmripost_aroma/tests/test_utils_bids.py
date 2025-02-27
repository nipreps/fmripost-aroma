"""Lightweight tests for fmripost_aroma.utils.bids."""

import os

import pytest
from bids.layout import BIDSLayout, BIDSLayoutIndexer
from niworkflows.utils.testing import generate_bids_skeleton

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
            for item, expected_item in zip(subject_data[key], value, strict=False):
                assert os.path.basename(item) == expected_item
        else:
            assert subject_data[key] is value


def test_collect_derivatives_longitudinal_01(tmpdir):
    """Test collect_derivatives with a mocked up longitudinal dataset."""
    # Generate a BIDS dataset
    bids_dir = tmpdir / 'collect_derivatives_longitudinal_01'
    dset_yaml = get_test_data_path() / 'skeletons' / 'skeleton_longitudinal_01.yml'
    generate_bids_skeleton(str(bids_dir), dset_yaml)

    layout = BIDSLayout(bids_dir, config=['bids', 'derivatives'], validate=False)

    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=True,
    )
    expected = {
        'bold_mni152nlin6asym': [
            'sub-102_ses-1_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz',
            'sub-102_ses-2_task-rest_space-MNI152NLin6Asym_res-02_desc-preproc_bold.nii.gz',
        ],
    }
    check_expected(subject_data, expected)


def test_collect_derivatives_longitudinal_02(tmpdir):
    """Test collect_derivatives with a mocked up longitudinal dataset."""
    # Generate a BIDS dataset
    bids_dir = tmpdir / 'collect_derivatives_longitudinal_02'
    dset_yaml = get_test_data_path() / 'skeletons' / 'skeleton_longitudinal_02.yml'
    generate_bids_skeleton(str(bids_dir), dset_yaml)

    layout = BIDSLayout(bids_dir, config=['bids', 'derivatives'], validate=False)

    # Query for all sessions should return all bold derivatives
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=True,
    )
    expected = {
        'bold_mni152nlin6asym': [
            'sub-102_ses-1_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
            'sub-102_ses-2_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
            'sub-102_ses-3_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
        ],
    }
    check_expected(subject_data, expected)

    # Query for session 2 should return anat from session 2
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102', 'session': '2'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=False,
    )
    expected = {
        'anat_mni152nlin6asym': (
            'sub-102_ses-2_space-MNI152NLin6Asym_res-02_desc-preproc_T1w.nii.gz'
        ),
    }
    check_expected(subject_data, expected)

    # Query for session 3 (no anat available) should raise an error
    with pytest.raises(
        ValueError,
        match='Multiple anatomical derivatives found for anat_mni152nlin6asym',
    ):
        subject_data = xbids.collect_derivatives(
            raw_dataset=None,
            derivatives_dataset=layout,
            entities={'subject': '102', 'session': '3'},
            fieldmap_id=None,
            spec=None,
            patterns=None,
            allow_multiple=False,
        )


def test_collect_derivatives_longitudinal_03(tmpdir):
    """Test collect_derivatives with a mocked up longitudinal dataset."""
    # Generate a BIDS dataset
    bids_dir = tmpdir / 'collect_derivatives_longitudinal_03'
    dset_yaml = get_test_data_path() / 'skeletons' / 'skeleton_longitudinal_03.yml'
    generate_bids_skeleton(str(bids_dir), dset_yaml)

    layout = BIDSLayout(bids_dir, config=['bids', 'derivatives'], validate=False)

    # Query for session 1 should return anat from session 1
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102', 'session': '1'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=False,
    )
    expected = {
        'anat_mni152nlin6asym': (
            'sub-102_ses-1_space-MNI152NLin6Asym_res-02_desc-preproc_T1w.nii.gz'
        ),
    }
    check_expected(subject_data, expected)

    # Query for session 2 should return anat from session 1 if no anat is present for session 2
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102', 'session': '2'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=False,
    )
    expected = {
        'anat_mni152nlin6asym': (
            'sub-102_ses-1_space-MNI152NLin6Asym_res-02_desc-preproc_T1w.nii.gz'
        ),
    }
    check_expected(subject_data, expected)


def test_collect_derivatives_xsectional_04(tmpdir):
    """Test collect_derivatives with a mocked up cross-sectional dataset."""
    # Generate a BIDS dataset
    bids_dir = tmpdir / 'collect_derivatives_xsectional_04'
    dset_yaml = get_test_data_path() / 'skeletons' / 'skeleton_crosssectional_01.yml'
    generate_bids_skeleton(str(bids_dir), dset_yaml)

    layout = BIDSLayout(bids_dir, config=['bids', 'derivatives'], validate=False)

    # Query for subject 102 should return anat from subject 102,
    # even though there are multiple subjects with anat derivatives,
    # because the subject is specified in the entities dictionary.
    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=layout,
        entities={'subject': '102'},
        fieldmap_id=None,
        spec=None,
        patterns=None,
        allow_multiple=False,
    )
    expected = {
        'anat_mni152nlin6asym': 'sub-102_space-MNI152NLin6Asym_res-02_desc-preproc_T1w.nii.gz',
    }
    check_expected(subject_data, expected)
