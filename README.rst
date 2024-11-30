#################
fMRIPost-template
#################

A generic fMRIPost workflow.

********
Overview
********

fMRIPost-template is a template repository that can be used to create new fMRIPost workflows.

The workflows and functions in this repository are designed to implement the majority of
general-purpose steps in an fMRIPost pipeline.
Here are a few of the key features:

1.  Configuration files to define expected BIDS derivatives from the preprocessing pipeline.
2.  Functions to collect and organize data from the BIDS derivatives.
3.  The ability to work with the following preprocessing configurations:

    -   fMRIPrep with ``--level full`` and the required output space for the fMRIPost workflow.
        For example, fMRIPost-AROMA requires outputs in ``MNI152NLin6Asym`` space with
        2x2x2 mm voxels.
    -   fMRIPrep with ``--level full`` and boldref-space outputs,
        along with transforms to the required output space for the fMRIPost workflow.
    -   fMRIPrep with ``--level full`` and boldref-space outputs,
        along with transforms to spaces that can be combined with existing transforms to
        required spaces.
        For example, users may apply fMRIPost-AROMA to boldref derivatives with transforms to
        MNI152NLin2009cAsym space.
        In this case, the fMRIPost-AROMA workflow will pull a transform from MNI152NLin2009cAsym
        to MNI152NLin6Asym from TemplateFlow and apply it,
        along with the boldref-to-MNI152NLin2009cAsym transform, to the boldref-space derivatives.

    .. warning::

        Currently, minimal- and resampling-level fMRIPrep derivatives are not supported,
        as fMRIPost workflows typically require confounds that are only generated with
        ``--level full``.

4.  General NiPreps infrastructure for running a BIDS App, such as a config file,
    a command-line interface, and tools to generate HTML reports.
