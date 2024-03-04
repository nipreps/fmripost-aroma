# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""fMRIPost-AROMA workflows to run ICA-AROMA."""
import os

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from fmripost_aroma import config
from fmripost_aroma.interfaces.bids import DerivativesDataSink


def init_ica_aroma_wf(
    *,
    bold_file: str,
    metadata: dict,
    aroma_melodic_dim: int = -200,
    err_on_aroma_warn: bool = False,
    susan_fwhm: float = 6.0,
):
    """Build a workflow that runs `ICA-AROMA`_.

    This workflow wraps `ICA-AROMA`_ to identify and remove motion-related
    independent components from a BOLD time series.

    The following steps are performed:

    #. Remove non-steady state volumes from the bold series.
    #. Smooth data using FSL `susan`, with a kernel width FWHM=6.0mm.
    #. Run FSL `melodic` outside of ICA-AROMA to generate the report
    #. Run ICA-AROMA
    #. Aggregate identified motion components (aggressive) to TSV
    #. Return ``classified_motion_ICs`` and ``melodic_mix`` for user to complete
       non-aggressive denoising in T1w space
    #. Calculate ICA-AROMA-identified noise components
       (columns named ``AROMAAggrCompXX``)

    Additionally, non-aggressive denoising is performed on the BOLD series
    resampled into MNI space.

    There is a current discussion on whether other confounds should be extracted
    before or after denoising `here
    <http://nbviewer.jupyter.org/github/nipreps/fmriprep-notebooks/blob/
    922e436429b879271fa13e76767a6e73443e74d9/issue-817_aroma_confounds.ipynb>`__.

    .. _ICA-AROMA: https://github.com/maartenmennes/ICA-AROMA

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold.confounds import init_ica_aroma_wf

            wf = init_ica_aroma_wf(
                bold_file="fake.nii.gz",
                metadata={"RepetitionTime": 1.0},
            )

    Parameters
    ----------
    bold_file
    metadata : :obj:`dict`
        BIDS metadata for BOLD file
    susan_fwhm : :obj:`float`
        Kernel width (FWHM in mm) for the smoothing step with
        FSL ``susan`` (default: 6.0mm)
    err_on_aroma_warn : :obj:`bool`
        Do not fail on ICA-AROMA errors
    aroma_melodic_dim : :obj:`int`
        Set the dimensionality of the MELODIC ICA decomposition.
        Negative numbers set a maximum on automatic dimensionality estimation.
        Positive numbers set an exact number of components to extract.
        (default: -200, i.e., estimate <=200 components)

    Inputs
    ------
    itk_bold_to_t1
        Affine transform from ``ref_bold_brain`` to T1 space (ITK format)
    anat2std_xfm
        ANTs-compatible affine-and-warp transform file
    name_source
        BOLD series NIfTI file
        Used to recover original information lost during processing
    skip_vols
        number of non steady state volumes
    bold_split
        Individual 3D BOLD volumes, not motion corrected
    bold_mask
        BOLD series mask in template space
    hmc_xforms
        List of affine transforms aligning each volume to ``ref_image`` in ITK format
    movpar_file
        SPM-formatted motion parameters file

    Outputs
    -------
    aroma_confounds
        TSV of confounds identified as noise by ICA-AROMA
    aroma_noise_ics
        CSV of noise components identified by ICA-AROMA
    melodic_mix
        FSL MELODIC mixing matrix
    nonaggr_denoised_file
        BOLD series with non-aggressive ICA-AROMA denoising applied
    """
    from nipype.interfaces import fsl
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.utility import TSV2JSON

    from fmripost_aroma.interfaces.confounds import ICAConfounds
    from fmripost_aroma.interfaces.reportlets import ICAAROMARPT

    workflow = Workflow(name=_get_wf_name(bold_file, "aroma"))
    workflow.__postdesc__ = """\
Automatic removal of motion artifacts using independent component analysis
[ICA-AROMA, @aroma] was performed on the *preprocessed BOLD on MNI space*
time-series after removal of non-steady state volumes and spatial smoothing
with an isotropic, Gaussian kernel of 6mm FWHM (full-width half-maximum).
Corresponding "non-aggresively" denoised runs were produced after such
smoothing.
Additionally, the "aggressive" noise-regressors were collected and placed
in the corresponding confounds file.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_std",
                "bold_mask_std",
                "confounds",
                "name_source",
                "skip_vols",
                "spatial_reference",
            ],
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "aroma_confounds",
                "aroma_noise_ics",
                "melodic_mix",
                "nonaggr_denoised_file",
                "aroma_metadata",
            ],
        ),
        name="outputnode",
    )

    # Convert confounds to FSL motpars file.
    ...

    rm_non_steady_state = pe.Node(
        niu.Function(function=_remove_volumes, output_names=["bold_cut"]),
        name="rm_nonsteady",
    )
    # fmt:off
    workflow.connect([
        (inputnode, rm_non_steady_state, [
            ("skip_vols", "skip_vols"),
            ("bold_std", "bold_file"),
        ]),
    ])
    # fmt:on

    calc_median_val = pe.Node(
        fsl.ImageStats(op_string="-k %s -p 50"),
        name="calc_median_val",
    )
    calc_bold_mean = pe.Node(
        fsl.MeanImage(),
        name="calc_bold_mean",
    )

    getusans = pe.Node(
        niu.Function(function=_getusans_func, output_names=["usans"]),
        name="getusans",
        mem_gb=0.01,
    )

    smooth = pe.Node(
        fsl.SUSAN(
            fwhm=susan_fwhm,
            output_type="NIFTI" if config.execution.low_mem else "NIFTI_GZ",
        ),
        name="smooth",
    )

    # melodic node
    melodic = pe.Node(
        fsl.MELODIC(
            no_bet=True,
            tr_sec=float(metadata["RepetitionTime"]),
            mm_thresh=0.5,
            out_stats=True,
            dim=aroma_melodic_dim,
        ),
        name="melodic",
    )

    # ica_aroma node
    ica_aroma = pe.Node(
        ICAAROMARPT(
            denoise_type="nonaggr",
            generate_report=True,
            TR=metadata["RepetitionTime"],
            args="-np",
        ),
        name="ica_aroma",
    )

    add_non_steady_state = pe.Node(
        niu.Function(function=_add_volumes, output_names=["bold_add"]),
        name="add_nonsteady",
    )

    # extract the confound ICs from the results
    ica_aroma_confound_extraction = pe.Node(
        ICAConfounds(err_on_aroma_warn=err_on_aroma_warn),
        name="ica_aroma_confound_extraction",
    )

    ica_aroma_metadata_fmt = pe.Node(
        TSV2JSON(
            index_column="IC",
            output=None,
            enforce_case=True,
            additional_metadata={
                "Method": {
                    "Name": "ICA-AROMA",
                    "Version": os.getenv("AROMA_VERSION", "n/a"),
                },
            },
        ),
        name="ica_aroma_metadata_fmt",
    )

    ds_report_ica_aroma = pe.Node(
        DerivativesDataSink(desc="aroma", datatype="figures", dismiss_entities=("echo",)),
        name="ds_report_ica_aroma",
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ica_aroma, [("movpar_file", "motion_parameters")]),
        (inputnode, calc_median_val, [("bold_mask_std", "mask_file")]),
        (rm_non_steady_state, calc_median_val, [("bold_cut", "in_file")]),
        (rm_non_steady_state, calc_bold_mean, [("bold_cut", "in_file")]),
        (calc_bold_mean, getusans, [("out_file", "image")]),
        (calc_median_val, getusans, [("out_stat", "thresh")]),
        # Connect input nodes to complete smoothing
        (rm_non_steady_state, smooth, [("bold_cut", "in_file")]),
        (getusans, smooth, [("usans", "usans")]),
        (calc_median_val, smooth, [(("out_stat", _getbtthresh), "brightness_threshold")]),
        # connect smooth to melodic
        (smooth, melodic, [("smoothed_file", "in_files")]),
        (inputnode, melodic, [("bold_mask_std", "mask")]),
        # connect nodes to ICA-AROMA
        (smooth, ica_aroma, [("smoothed_file", "in_file")]),
        (inputnode, ica_aroma, [
            ("bold_mask_std", "report_mask"),
            ("bold_mask_std", "mask")]),
        (melodic, ica_aroma, [("out_dir", "melodic_dir")]),
        # generate tsvs from ICA-AROMA
        (ica_aroma, ica_aroma_confound_extraction, [("out_dir", "in_directory")]),
        (inputnode, ica_aroma_confound_extraction, [("skip_vols", "skip_vols")]),
        (ica_aroma_confound_extraction, ica_aroma_metadata_fmt, [("aroma_metadata", "in_file")]),
        # output for processing and reporting
        (ica_aroma_confound_extraction, outputnode, [
            ("aroma_confounds", "aroma_confounds"),
            ("aroma_noise_ics", "aroma_noise_ics"),
            ("melodic_mix", "melodic_mix"),
        ]),
        (ica_aroma_metadata_fmt, outputnode, [("output", "aroma_metadata")]),
        (ica_aroma, add_non_steady_state, [("nonaggr_denoised_file", "bold_cut_file")]),
        (inputnode, add_non_steady_state, [
            ("bold_std", "bold_file"),
            ("skip_vols", "skip_vols"),
        ]),
        (add_non_steady_state, outputnode, [("bold_add", "nonaggr_denoised_file")]),
        (ica_aroma, ds_report_ica_aroma, [("out_report", "in_file")]),
    ])
    # fmt:on
    return workflow


def _getbtthresh(medianval):
    return 0.75 * medianval


def _getusans_func(image, thresh):
    return [tuple([image, thresh])]


def _remove_volumes(bold_file, skip_vols):
    """Remove skip_vols from bold_file."""
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    if skip_vols == 0:
        return bold_file

    out = fname_presuffix(bold_file, suffix="_cut")
    bold_img = nb.load(bold_file)
    bold_img.__class__(
        bold_img.dataobj[..., skip_vols:], bold_img.affine, bold_img.header
    ).to_filename(out)
    return out


def _add_volumes(bold_file, bold_cut_file, skip_vols):
    """Prepend skip_vols from bold_file onto bold_cut_file."""
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    if skip_vols == 0:
        return bold_cut_file

    bold_img = nb.load(bold_file)
    bold_cut_img = nb.load(bold_cut_file)

    bold_data = np.concatenate((bold_img.dataobj[..., :skip_vols], bold_cut_img.dataobj), axis=3)

    out = fname_presuffix(bold_cut_file, suffix="_addnonsteady")
    bold_img.__class__(bold_data, bold_img.affine, bold_img.header).to_filename(out)
    return out


def _get_wf_name(bold_fname, prefix):
    """
    Derive the workflow name for supplied BOLD file.

    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_bold.nii.gz", "aroma")
    'aroma_task_nback_wf'
    >>> _get_wf_name(
    ...     "/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz",
    ...     "preproc",
    ... )
    'preproc_task_nback_run_01_echo_1_wf'

    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = "_".join(fname.split("_")[1:-1])
    return f"{prefix}_{fname_nosub.replace('-', '_')}_wf"
