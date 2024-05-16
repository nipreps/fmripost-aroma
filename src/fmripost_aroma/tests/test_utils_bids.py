"""Lightweight tests for fmripost_aroma.utils.bids."""

from fmripost_aroma.tests.utils import get_test_data_path


def test_collect_derivatives_raw():
    """Test collect_derivatives with a raw dataset."""
    from pathlib import Path

    from fmripost_aroma.utils.bids import collect_derivatives

    data_dir = get_test_data_path()

    raw_dataset = Path(data_dir) / 'ds000005'

    subject_data = collect_derivatives(raw_dataset=raw_dataset)
    assert subject_data['bold'] == ''


def test_collect_derivatives_minimal(minimal_ignore_list):
    """Test collect_derivatives with a minimal dataset."""
    from pathlib import Path

    from bids.layout import BIDSLayout, BIDSLayoutIndexer
    from fmripost_aroma.utils.bids import collect_derivatives

    data_dir = get_test_data_path()

    derivatives_dataset = Path(data_dir) / 'ds000005-fmriprep'
    derivatives_layout = BIDSLayout(
        derivatives_dataset,
        config=['bids', 'derivatives'],
        validate=False,
        indexer=BIDSLayoutIndexer(validate=False, ignore=minimal_ignore_list),
    )

    subject_data = collect_derivatives(derivatives_dataset=derivatives_layout)
    assert subject_data['bold_std'] == ''
    assert subject_data['bold_mask_std'] == ''
    assert subject_data['confounds'] is None
