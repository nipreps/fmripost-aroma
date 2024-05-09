"""Tests for fmripost_aroma.workflows."""

from fmripost_aroma.tests.tests import mock_config


def test_init_ica_aroma_wf():
    from fmripost_aroma.workflows.aroma import init_ica_aroma_wf

    with mock_config():
        wf = init_ica_aroma_wf(
            bold_file='sub-01_task-rest_bold.nii.gz',
            metadata={'RepetitionTime': 2.0},
        )
        assert wf.name == 'aroma_task_rest_wf'


def test_init_denoise_wf():
    from fmripost_aroma.workflows.aroma import init_denoise_wf

    with mock_config():
        wf = init_denoise_wf(bold_file='sub-01_task-rest_bold.nii.gz')
        assert wf.name == 'denoise_task_rest_wf'
