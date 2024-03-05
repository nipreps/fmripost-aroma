.. include:: links.rst

===========================
Processing pipeline details
===========================
*fMRIPrep* adapts its pipeline depending on what data and metadata are
available and are used as the input.
For example, slice timing correction will be
performed only if the ``SliceTiming`` metadata field is found for the input
dataset.

A (very) high-level view of the simplest pipeline (for a single-band dataset with only
one task, single-run, with no slice-timing information nor fieldmap acquisitions)
is presented below:

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from fmripost_aroma.workflows.tests import mock_config
    from fmripost_aroma.workflows.base import init_single_subject_wf
    with mock_config():
        wf = init_single_subject_wf('01')
