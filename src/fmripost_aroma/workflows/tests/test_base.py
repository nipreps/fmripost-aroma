"""Tests for fmripost_aroma.workflows."""


def test_init_single_subject_wf():
    from fmripost_aroma.workflows.base import init_single_subject_wf

    wf = init_single_subject_wf(subject_id='01')
    assert wf.name == 'sub_01_wf'
