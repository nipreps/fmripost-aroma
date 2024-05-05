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
from fmripost_aroma.interfaces.aroma import AROMAClassifier


def init_ica_aroma_wf(
    *,
    bold_file: str,
    metadata: dict,
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
    <http://nbviewer.jupyter.org/github/nipreps/fmripost_aroma-notebooks/blob/
    922e436429b879271fa13e76767a6e73443e74d9/issue-817_aroma_confounds.ipynb>`__.

    .. _ICA-AROMA: https://github.com/maartenmennes/ICA-AROMA

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_aroma.workflows.bold.confounds import init_ica_aroma_wf

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

    Inputs
    ------
    name_source
        BOLD series NIfTI file
        Used to recover original information lost during processing
    skip_vols
        number of non steady state volumes
    bold_mask
        BOLD series mask in template space
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

    # TODO: Convert confounds to FSL motpars file.
    ...

    rm_non_steady_state = pe.Node(
        niu.Function(function=_remove_volumes, output_names=["bold_cut"]),
        name="rm_nonsteady",
    )
    workflow.connect([
        (inputnode, rm_non_steady_state, [
            ("skip_vols", "skip_vols"),
            ("bold_std", "bold_file"),
        ]),
    ])  # fmt:skip

    calc_median_val = pe.Node(
        fsl.ImageStats(op_string="-k %s -p 50"),
        name="calc_median_val",
    )
    workflow.connect([
        (inputnode, calc_median_val, [("bold_mask_std", "mask_file")]),
        (rm_non_steady_state, calc_median_val, [("bold_cut", "in_file")]),
    ])  # fmt:skip

    calc_bold_mean = pe.Node(
        fsl.MeanImage(),
        name="calc_bold_mean",
    )
    workflow.connect([(rm_non_steady_state, calc_bold_mean, [("bold_cut", "in_file")])])

    getusans = pe.Node(
        niu.Function(function=_getusans_func, output_names=["usans"]),
        name="getusans",
        mem_gb=0.01,
    )
    workflow.connect([
        (calc_median_val, getusans, [("out_stat", "thresh")]),
        (calc_bold_mean, getusans, [("out_file", "image")]),
    ])  # fmt:skip

    smooth = pe.Node(
        fsl.SUSAN(
            fwhm=susan_fwhm,
            output_type="NIFTI" if config.execution.low_mem else "NIFTI_GZ",
        ),
        name="smooth",
    )
    workflow.connect([
        (rm_non_steady_state, smooth, [("bold_cut", "in_file")]),
        (getusans, smooth, [("usans", "usans")]),
        (calc_median_val, smooth, [(("out_stat", _getbtthresh), "brightness_threshold")]),
    ])  # fmt:skip

    # melodic node
    melodic = pe.Node(
        fsl.MELODIC(
            no_bet=True,
            tr_sec=float(metadata["RepetitionTime"]),
            mm_thresh=0.5,
            out_stats=True,
            dim=config.workflow.melodic_dim,
        ),
        name="melodic",
    )
    workflow.connect([
        (inputnode, melodic, [("bold_mask_std", "mask")]),
        (smooth, melodic, [("smoothed_file", "in_files")]),
    ])  # fmt:skip

    select_melodic_files = pe.Node(
        niu.Function(
            function=_select_melodic_files,
            input_names=["melodic_dir"],
            output_names=["mixing", "component_maps"],
        ),
        name="select_melodic_files",
    )
    workflow.connect([(melodic, select_melodic_files, [("out_dir", "melodic_dir")])])

    # Run the ICA-AROMA classifier
    ica_aroma = pe.Node(AROMAClassifier(TR=metadata["RepetitionTime"]))
    workflow.connect([
        (inputnode, ica_aroma, [("movpar_file", "motion_parameters")]),
        (select_melodic_files, ica_aroma, [
            ("mixing", "mixing"),
            ("component_maps", "component_maps"),
        ]),
    ])  # fmt:skip

    # Generate the ICA-AROMA report
    # What steps does this entail?
    aroma_rpt = pe.Node(
        ICAAROMARPT(TR=metadata["RepetitionTime"]),
        name="aroma_rpt",
    )
    workflow.connect([
        (inputnode, aroma_rpt, [("bold_mask_std", "report_mask")]),
        (smooth, aroma_rpt, [("smoothed_file", "in_file")]),
        (melodic, aroma_rpt, [("out_dir", "melodic_dir")]),
        (ica_aroma, aroma_rpt, [("aroma_noise_ics", "aroma_noise_ics")]),
    ])  # fmt:skip

    add_non_steady_state = pe.Node(
        niu.Function(function=_add_volumes, output_names=["bold_add"]),
        name="add_non_steady_state",
    )
    workflow.connect([
        (inputnode, add_non_steady_state, [
            ("bold_std", "bold_file"),
            ("skip_vols", "skip_vols"),
        ]),
        (aroma_rpt, add_non_steady_state, [("nonaggr_denoised_file", "bold_cut_file")]),
        (add_non_steady_state, outputnode, [("bold_add", "nonaggr_denoised_file")]),
    ])  # fmt:skip

    # extract the confound ICs from the results
    ica_aroma_confound_extraction = pe.Node(
        ICAConfounds(
            err_on_aroma_warn=config.workflow.err_on_warn,
            orthogonalize=config.workflow.orthogonalize,
        ),
        name="ica_aroma_confound_extraction",
    )
    workflow.connect([
        (inputnode, ica_aroma_confound_extraction, [("skip_vols", "skip_vols")]),
        (melodic, ica_aroma_confound_extraction, [("out_dir", "melodic_dir")]),
        (ica_aroma, ica_aroma_confound_extraction, [
            ("aroma_features", "aroma_features"),
            ("aroma_noise_ics", "aroma_noise_ics"),
            ("aroma_metadata", "aroma_metadata"),
        ]),
        (ica_aroma_confound_extraction, outputnode, [
            ("aroma_confounds", "aroma_confounds"),
            ("aroma_noise_ics", "aroma_noise_ics"),
            ("melodic_mix", "melodic_mix"),
        ]),
    ])  # fmt:skip

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
    workflow.connect([
        (ica_aroma_confound_extraction, ica_aroma_metadata_fmt, [("aroma_metadata", "in_file")]),
        (ica_aroma_metadata_fmt, outputnode, [("output", "aroma_metadata")]),
    ])  # fmt:skip

    ds_report_ica_aroma = pe.Node(
        DerivativesDataSink(desc="aroma", datatype="figures", dismiss_entities=("echo",)),
        name="ds_report_ica_aroma",
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([(aroma_rpt, ds_report_ica_aroma, [("out_report", "in_file")])])

    return workflow


def init_denoise_wf(bold_file):
    """Build a workflow that denoises a BOLD series using AROMA confounds.

    This workflow performs the denoising in the requested output space(s).
    It doesn't currently work on CIFTIs.
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_aroma.interfaces.confounds import ICADenoise

    workflow = Workflow(name=_get_wf_name(bold_file, "denoise"))

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "bold_mask_std",
                "confounds",
                "name_source",
                "skip_vols",
                "spatial_reference",
            ],
        ),
        name="inputnode",
    )

    rm_non_steady_state = pe.Node(
        niu.Function(function=_remove_volumes, output_names=["bold_cut"]),
        name="rm_nonsteady",
    )
    workflow.connect([
        (inputnode, rm_non_steady_state, [
            ("skip_vols", "skip_vols"),
            ("bold_file", "bold_file"),
        ]),
    ])  # fmt:skip

    for denoise_method in config.workflow.denoise_method:
        denoise = pe.Node(
            ICADenoise(method=denoise_method),
            name=f"denoise_{denoise_method}",
        )
        workflow.connect([
            (inputnode, denoise, [
                ("confounds", "confounds_file"),
                ("skip_vols", "skip_vols"),
                ("bold_mask_std", "mask_file"),
            ]),
            (rm_non_steady_state, denoise, [("bold_cut", "bold_file")]),
        ])  # fmt:skip

        add_non_steady_state = pe.Node(
            niu.Function(function=_add_volumes, output_names=["bold_add"]),
            name="add_non_steady_state",
        )
        workflow.connect([
            (inputnode, add_non_steady_state, [
                ("bold_file", "bold_file"),
                ("skip_vols", "skip_vols"),
            ]),
            (denoise, add_non_steady_state, [("denoised_file", "bold_cut_file")]),
        ])  # fmt:skip

        ds_denoised = pe.Node(
            DerivativesDataSink(
                desc=f"{denoise_method}Denoised",
                datatype="func",
                dismiss_entities=("echo",),
            ),
            name=f"ds_denoised_{denoise_method}",
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([(add_non_steady_state, ds_denoised, [("bold_add", "denoised_file")])])

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


def _select_melodic_files(melodic_dir):
    """Select the mixing and component maps from the Melodic output."""
    import os

    mixing = os.path.join(melodic_dir, "melodic_mix")
    assert os.path.isfile(mixing), f"Missing MELODIC mixing matrix: {mixing}"
    component_maps = os.path.join(melodic_dir, "melodic_IC.nii.gz")
    assert os.path.isfile(component_maps), f"Missing MELODIC ICs: {component_maps}"
    return mixing, component_maps
