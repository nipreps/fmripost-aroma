"""Lightweight tests for fmripost_aroma.utils.bids."""

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
    assert subject_data['bold'] == ''


def test_collect_derivatives_minimal(minimal_ignore_list):
    """Test collect_derivatives with a minimal-mode dataset."""
    from pathlib import Path

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
    assert subject_data['bold_std'] == ''
    assert subject_data['bold_mask_std'] == ''
    assert subject_data['confounds'] is None


def test_collect_derivatives_full(base_ignore_list):
    """Test collect_derivatives with a full-mode dataset."""
    from pathlib import Path

    data_dir = get_test_data_path()

    derivatives_dataset = Path(data_dir) / 'ds000005-fmriprep'
    derivatives_layout = BIDSLayout(
        derivatives_dataset,
        config=['bids', 'derivatives'],
        validate=False,
        indexer=BIDSLayoutIndexer(validate=False, ignore=base_ignore_list),
    )

    subject_data = xbids.collect_derivatives(
        raw_dataset=None,
        derivatives_dataset=derivatives_layout,
        entities={'subject': '01', 'task': 'mixedgamblestask', 'run': 1},
        fieldmap_id=None,
        spec=None,
        patterns=None,
    )
    assert subject_data['bold_std'] == ''
    assert subject_data['bold_mask_std'] == ''
    assert subject_data['confounds'] is None
