"""Tests for fmripost_aroma.workflows."""


def test_init_ica_aroma_wf():
    from fmripost_aroma.workflows.aroma import init_ica_aroma_wf

    wf = init_ica_aroma_wf(subject_id='01')
    assert wf.name == 'sub_01_wf'
