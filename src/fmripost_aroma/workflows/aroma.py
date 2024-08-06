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

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from fmripost_aroma import config
from fmripost_aroma.interfaces.aroma import AROMAClassifier
from fmripost_aroma.interfaces.bids import DerivativesDataSink
from fmripost_aroma.utils.utils import _get_wf_name


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
    #. Aggregate components and classifications to TSVs

    There is a current discussion on whether other confounds should be extracted
    before or after denoising `here
    <http://nbviewer.jupyter.org/github/nipreps/fmriprep-notebooks/blob/
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
                susan_fwhm=6.0,
            )

    Parameters
    ----------
    bold_file
        BOLD series used as name source for derivatives
    metadata : :obj:`dict`
        BIDS metadata for BOLD file
    susan_fwhm : :obj:`float`
        Kernel width (FWHM in mm) for the smoothing step with
        FSL ``susan`` (default: 6.0mm)

    Inputs
    ------
    bold_std
        BOLD series in template space
    bold_mask_std
        BOLD series mask in template space
    confounds
        fMRIPrep-formatted confounds file, which must include the following columns:
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z".
    skip_vols
        number of non steady state volumes

    Outputs
    -------
    mixing
        FSL MELODIC mixing matrix
    aroma_features
        TSV of feature values used to classify components in ``mixing``.
    features_metadata
        Dictionary describing the ICA-AROMA run
    aroma_confounds
        TSV of confounds identified as noise by ICA-AROMA
    """

    from nipype.interfaces import fsl
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_aroma.interfaces.confounds import ICAConfounds
    from fmripost_aroma.interfaces.nilearn import MeanImage, MedianValue
    from fmripost_aroma.interfaces.reportlets import ICAAROMARPT, ICAAROMAMetricsRPT
    from fmripost_aroma.utils.utils import _convert_to_tsv

    workflow = Workflow(name=_get_wf_name(bold_file, 'aroma'))
    workflow.__postdesc__ = f"""\
Automatic removal of motion artifacts using independent component analysis
[ICA-AROMA, @ica_aroma] was performed on the *preprocessed BOLD on MNI152NLin6Asym space*
time-series after removal of non-steady state volumes and spatial smoothing
with a nonlinear filter that preserves underlying structure [SUSAN, @susan],
using a FWHM of {susan_fwhm} mm.
Additionally, the component time-series classified as "noise" were collected and placed
in the corresponding confounds file.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_std',
                'bold_mask_std',
                'confounds',
                'skip_vols',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'mixing',
                'aroma_features',
                'features_metadata',
                'aroma_confounds',
            ],
        ),
        name='outputnode',
    )

    rm_non_steady_state = pe.Node(
        niu.Function(function=_remove_volumes, output_names=['bold_cut']),
        name='rm_nonsteady',
    )
    workflow.connect([
        (inputnode, rm_non_steady_state, [
            ('skip_vols', 'skip_vols'),
            ('bold_std', 'bold_file'),
        ]),
    ])  # fmt:skip

    calc_median_val = pe.Node(
        MedianValue(),
        name='calc_median_val',
    )
    workflow.connect([
        (inputnode, calc_median_val, [('bold_mask_std', 'mask_file')]),
        (rm_non_steady_state, calc_median_val, [('bold_cut', 'bold_file')]),
    ])  # fmt:skip

    calc_bold_mean = pe.Node(
        MeanImage(),
        name='calc_bold_mean',
    )
    workflow.connect([
        (inputnode, calc_bold_mean, [('bold_mask_std', 'mask_file')]),
        (rm_non_steady_state, calc_bold_mean, [('bold_cut', 'bold_file')]),
    ])  # fmt:skip

    getusans = pe.Node(
        niu.Function(function=_getusans_func, output_names=['usans']),
        name='getusans',
        mem_gb=0.01,
    )
    workflow.connect([
        (calc_median_val, getusans, [('median_value', 'thresh')]),
        (calc_bold_mean, getusans, [('out_file', 'image')]),
    ])  # fmt:skip

    smooth = pe.Node(
        fsl.SUSAN(
            fwhm=susan_fwhm,
            output_type='NIFTI' if config.execution.low_mem else 'NIFTI_GZ',
        ),
        name='smooth',
    )
    workflow.connect([
        (rm_non_steady_state, smooth, [('bold_cut', 'in_file')]),
        (getusans, smooth, [('usans', 'usans')]),
        (calc_median_val, smooth, [(('median_value', _getbtthresh), 'brightness_threshold')]),
    ])  # fmt:skip

    # ICA with MELODIC
    melodic = pe.Node(
        fsl.MELODIC(
            no_bet=True,
            tr_sec=float(metadata['RepetitionTime']),
            mm_thresh=0.5,
            out_stats=True,
            dim=config.workflow.melodic_dim,
        ),
        name='melodic',
    )
    workflow.connect([
        (inputnode, melodic, [('bold_mask_std', 'mask')]),
        (smooth, melodic, [('smoothed_file', 'in_files')]),
    ])  # fmt:skip

    select_melodic_files = pe.Node(
        niu.Function(
            function=_select_melodic_files,
            input_names=['melodic_dir'],
            output_names=['mixing', 'component_maps', 'component_stats'],
        ),
        name='select_melodic_files',
    )
    workflow.connect([(melodic, select_melodic_files, [('out_dir', 'melodic_dir')])])

    # Run the ICA-AROMA classifier
    ica_aroma = pe.Node(
        AROMAClassifier(TR=metadata['RepetitionTime']),
        name='ica_aroma',
    )
    workflow.connect([
        (inputnode, ica_aroma, [
            ('confounds', 'motpars'),
            ('skip_vols', 'skip_vols'),
        ]),
        (select_melodic_files, ica_aroma, [
            ('mixing', 'mixing'),
            ('component_maps', 'component_maps'),
            ('component_stats', 'component_stats'),
        ]),
        (ica_aroma, outputnode, [
            ('aroma_features', 'aroma_features'),
            ('aroma_metadata', 'features_metadata'),
        ]),
    ])  # fmt:skip

    # Generate reportlets
    aroma_rpt = pe.Node(
        ICAAROMARPT(),
        name='aroma_rpt',
    )
    workflow.connect([
        (inputnode, aroma_rpt, [('bold_mask_std', 'report_mask')]),
        (smooth, aroma_rpt, [('smoothed_file', 'in_file')]),
        (melodic, aroma_rpt, [('out_dir', 'melodic_dir')]),
        (ica_aroma, aroma_rpt, [('aroma_noise_ics', 'aroma_noise_ics')]),
    ])  # fmt:skip

    ds_report_ica_aroma = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmripost_aroma_dir,
            source_file=bold_file,
            datatype='figures',
            desc='aroma',
            dismiss_entities=('echo', 'den', 'res'),
        ),
        name='ds_report_ica_aroma',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([(aroma_rpt, ds_report_ica_aroma, [('out_report', 'in_file')])])

    # extract the confound ICs from the results
    ica_aroma_confound_extraction = pe.Node(
        ICAConfounds(
            err_on_aroma_warn=config.workflow.err_on_warn,
        ),
        name='ica_aroma_confound_extraction',
    )
    workflow.connect([
        (inputnode, ica_aroma_confound_extraction, [('skip_vols', 'skip_vols')]),
        (select_melodic_files, ica_aroma_confound_extraction, [('mixing', 'mixing')]),
        (ica_aroma, ica_aroma_confound_extraction, [('aroma_features', 'aroma_features')]),
        (ica_aroma_confound_extraction, outputnode, [
            ('aroma_confounds', 'aroma_confounds'),
            ('mixing', 'mixing'),
        ]),
    ])  # fmt:skip

    ds_components = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmripost_aroma_dir,
            source_file=bold_file,
            datatype='func',
            desc='melodic',
            suffix='components',
            extension='nii.gz',
            dismiss_entities=('echo', 'den', 'res'),
        ),
        name='ds_components',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([(select_melodic_files, ds_components, [('component_maps', 'in_file')])])

    convert_to_tsv = pe.Node(
        niu.Function(function=_convert_to_tsv, output_names=['out_file']),
        name='convert_to_tsv',
    )
    workflow.connect([(select_melodic_files, convert_to_tsv, [('mixing', 'in_file')])])

    ds_mixing = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmripost_aroma_dir,
            source_file=bold_file,
            datatype='func',
            desc='melodic',
            suffix='mixing',
            extension='tsv',
            dismiss_entities=('echo', 'den', 'res'),
        ),
        name='ds_mixing',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([(convert_to_tsv, ds_mixing, [('out_file', 'in_file')])])

    ds_aroma_features = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmripost_aroma_dir,
            source_file=bold_file,
            datatype='func',
            desc='aroma',
            suffix='metrics',
            extension='tsv',
            dismiss_entities=('echo', 'den', 'res'),
        ),
        name='ds_aroma_features',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([
        (ica_aroma, ds_aroma_features, [
            ('aroma_features', 'in_file'),
            ('aroma_metadata', 'meta_dict'),
        ]),
    ])  # fmt:skip

    ds_aroma_confounds = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmripost_aroma_dir,
            source_file=bold_file,
            datatype='func',
            desc='melodic',
            suffix='timeseries',
            extension='tsv',
            dismiss_entities=('echo', 'den', 'res'),
        ),
        name='ds_aroma_confounds',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([
        (ica_aroma_confound_extraction, ds_aroma_confounds, [('aroma_confounds', 'in_file')]),
    ])  # fmt:skip

    metrics_rpt = pe.Node(
        ICAAROMAMetricsRPT(),
        name='metrics_rpt',
    )
    workflow.connect([(ica_aroma, metrics_rpt, [('aroma_features', 'aroma_features')])])

    ds_report_metrics = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmripost_aroma_dir,
            source_file=bold_file,
            datatype='figures',
            desc='metrics',
            dismiss_entities=('echo', 'den', 'res'),
        ),
        name='ds_report_metrics',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([(metrics_rpt, ds_report_metrics, [('out_report', 'in_file')])])

    return workflow


def init_denoise_wf(bold_file):
    """Build a workflow that denoises a BOLD series using AROMA confounds.

    This workflow performs the denoising in the requested output space(s).
    It doesn't currently work on CIFTIs.
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_aroma.interfaces.confounds import ICADenoise

    workflow = Workflow(name=_get_wf_name(bold_file, 'denoise'))

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_file',
                'bold_mask',
                'mixing',
                'classifications',
                'skip_vols',
                'space',
                'cohort',
                'resolution',
            ],
        ),
        name='inputnode',
    )

    rm_non_steady_state = pe.Node(
        niu.Function(function=_remove_volumes, output_names=['bold_cut']),
        name='rm_nonsteady',
    )
    workflow.connect([
        (inputnode, rm_non_steady_state, [
            ('skip_vols', 'skip_vols'),
            ('bold_file', 'bold_file'),
        ]),
    ])  # fmt:skip

    for denoise_method in config.workflow.denoise_method:
        denoise = pe.Node(
            ICADenoise(method=denoise_method),
            name=f'denoise_{denoise_method}',
        )
        workflow.connect([
            (inputnode, denoise, [
                ('mixing', 'mixing'),
                ('classifications', 'metrics'),
                ('bold_mask', 'mask_file'),
                ('skip_vols', 'skip_vols'),
            ]),
            (rm_non_steady_state, denoise, [('bold_cut', 'bold_file')]),
        ])  # fmt:skip

        add_non_steady_state = pe.Node(
            niu.Function(function=_add_volumes, output_names=['bold_add']),
            name=f'add_non_steady_state_{denoise_method}',
        )
        workflow.connect([
            (inputnode, add_non_steady_state, [
                ('bold_file', 'bold_file'),
                ('skip_vols', 'skip_vols'),
            ]),
            (denoise, add_non_steady_state, [('denoised_file', 'bold_cut_file')]),
        ])  # fmt:skip

        ds_denoised = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.fmripost_aroma_dir,
                source_file=bold_file,
                compress=True,
                datatype='func',
                desc=f'{denoise_method}Denoised',
                suffix='bold',
            ),
            name=f'ds_denoised_{denoise_method}',
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, ds_denoised, [
                ('space', 'space'),
                ('cohort', 'cohort'),
                ('resolution', 'resolution'),
            ]),
            (add_non_steady_state, ds_denoised, [('bold_add', 'in_file')]),
        ])  # fmt:skip

    return workflow


def _getbtthresh(medianval):
    return 0.75 * medianval


def _getusans_func(image, thresh):
    return [(image, thresh)]


def _remove_volumes(bold_file, skip_vols):
    """Remove skip_vols from bold_file."""
    import os

    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    if skip_vols == 0:
        return bold_file

    out = fname_presuffix(bold_file, prefix='cut_', newpath=os.getcwd())
    bold_img = nb.load(bold_file)
    bold_img.__class__(
        bold_img.dataobj[..., skip_vols:], bold_img.affine, bold_img.header
    ).to_filename(out)
    return out


def _add_volumes(bold_file, bold_cut_file, skip_vols):
    """Prepend skip_vols from bold_file onto bold_cut_file."""
    import os

    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    if skip_vols == 0:
        return bold_cut_file

    bold_img = nb.load(bold_file)
    bold_cut_img = nb.load(bold_cut_file)

    bold_data = np.concatenate((bold_img.dataobj[..., :skip_vols], bold_cut_img.dataobj), axis=3)

    out = fname_presuffix(bold_cut_file, prefix='addnonsteady_', newpath=os.getcwd())
    bold_img.__class__(bold_data, bold_img.affine, bold_img.header).to_filename(out)
    return out


def _select_melodic_files(melodic_dir):
    """Select the mixing and component maps from the Melodic output."""
    import os

    mixing = os.path.join(melodic_dir, 'melodic_mix')
    if not os.path.isfile(mixing):
        raise FileNotFoundError(f'Missing MELODIC mixing matrix: {mixing}')

    component_maps = os.path.join(melodic_dir, 'melodic_IC.nii.gz')
    if not os.path.isfile(component_maps):
        raise FileNotFoundError(f'Missing MELODIC ICs: {component_maps}')

    component_stats = os.path.join(melodic_dir, 'melodic_ICstats')
    if not os.path.isfile(component_stats):
        raise FileNotFoundError(f'Missing MELODIC IC stats: {component_stats}')

    return mixing, component_maps, component_stats
