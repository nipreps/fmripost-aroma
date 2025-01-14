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
"""
fMRIPost AROMA workflows
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fmripost_aroma_wf
.. autofunction:: init_single_subject_wf

"""

import os
import sys
from collections import defaultdict
from copy import deepcopy

import yaml
from nipype.pipeline import engine as pe
from packaging.version import Version

from fmripost_aroma import config
from fmripost_aroma.utils.utils import _get_wf_name, update_dict


def init_fmripost_aroma_wf():
    """Build *fMRIPost-AROMA*'s pipeline.

    This workflow organizes the execution of fMRIPost-AROMA,
    with a sub-workflow for each subject.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_aroma.workflows.tests import mock_config
            from fmripost_aroma.workflows.base import init_fmripost_aroma_wf

            with mock_config():
                wf = init_fmripost_aroma_wf()

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    ver = Version(config.environment.version)

    fmripost_aroma_wf = Workflow(name=f'fmripost_aroma_{ver.major}_{ver.minor}_wf')
    fmripost_aroma_wf.base_dir = config.execution.work_dir

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        single_subject_wf.config['execution']['crashdump_dir'] = str(
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        fmripost_aroma_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / 'fmripost_aroma.toml')

    return fmripost_aroma_wf


def init_single_subject_wf(subject_id: str):
    """Organize the postprocessing pipeline for a single subject.

    It collects and reports information about the subject,
    and prepares sub-workflows to postprocess each BOLD series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_aroma.workflows.tests import mock_config
            from fmripost_aroma.workflows.base import init_single_subject_wf

            with mock_config():
                wf = init_single_subject_wf('01')

    Parameters
    ----------
    subject_id : :obj:`str`
        Subject label for this single-subject workflow.

    Notes
    -----
    1.  Load fMRIPost-AROMA config file.
    2.  Collect fMRIPrep derivatives.
        -   BOLD file in native space.
        -   Two main possibilities:
            1.  bids_dir is a raw BIDS dataset and preprocessing derivatives
                are provided through ``--derivatives``.
                In this scenario, we only need minimal derivatives.
            2.  bids_dir is a derivatives dataset and we need to collect compliant
                derivatives to get the data into the right space.
    3.  Loop over runs.
    4.  Collect each run's associated files.
        -   Transform(s) to MNI152NLin6Asym
        -   Confounds file
        -   ICA-AROMA uses its own standard-space edge, CSF, and brain masks,
            so we don't need to worry about those.
    5.  Use ``resampler`` to warp BOLD to MNI152NLin6Asym-2mm.
    6.  Convert motion parameters from confounds file to FSL format.
    7.  Run ICA-AROMA.
    8.  Warp BOLD to requested output spaces and denoise with ICA-AROMA.

    """
    from bids.utils import listify
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSInfo
    from niworkflows.interfaces.nilearn import NILEARN_VERSION

    from fmripost_aroma.interfaces.bids import DerivativesDataSink
    from fmripost_aroma.interfaces.reportlets import AboutSummary, SubjectSummary
    from fmripost_aroma.utils.bids import collect_derivatives

    spaces = config.workflow.spaces

    workflow = Workflow(name=f'sub_{subject_id}_wf')
    workflow.__desc__ = f"""
Results included in this manuscript come from postprocessing
performed using *fMRIPost-AROMA* {config.environment.version} (@ica_aroma),
which is based on *Nipype* {config.environment.nipype_version}
(@nipype1; @nipype2; RRID:SCR_002502).

"""
    workflow.__postdesc__ = f"""

Many internal operations of *fMRIPost-AROMA* use
*Nilearn* {NILEARN_VERSION} [@nilearn, RRID:SCR_001362].
For more details of the pipeline, see [the section corresponding
to workflows in *fMRIPost-AROMA*'s documentation]\
(https://fmripost_aroma.readthedocs.io/en/latest/workflows.html \
"FMRIPrep's documentation").


### Copyright Waiver

The above boilerplate text was automatically generated by fMRIPost-AROMA
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0]\
(https://creativecommons.org/publicdomain/zero/1.0/) license.

### References

"""
    entities = config.execution.bids_filters or {}
    entities['subject'] = subject_id

    if config.execution.derivatives:
        # Raw dataset + derivatives dataset
        config.loggers.workflow.info('Raw+derivatives workflow mode enabled')
        # Just build a list of BOLD files right now
        subject_data = collect_derivatives(
            raw_dataset=config.execution.layout,
            derivatives_dataset=None,
            entities=entities,
            fieldmap_id=None,
            allow_multiple=True,
            spaces=None,
        )
        subject_data['bold'] = listify(subject_data['bold_raw'])
    else:
        # Derivatives dataset only
        config.loggers.workflow.info('Derivatives-only workflow mode enabled')
        # Just build a list of BOLD files right now
        subject_data = collect_derivatives(
            raw_dataset=None,
            derivatives_dataset=config.execution.layout,
            entities=entities,
            fieldmap_id=None,
            allow_multiple=True,
            spaces=None,
        )
        # Patch standard-space BOLD files into 'bold' key
        subject_data['bold'] = listify(subject_data['bold_mni152nlin6asym'])

    # Make sure we always go through these two checks
    if not subject_data['bold']:
        task_id = config.execution.task_id
        raise RuntimeError(
            f'No BOLD images found for participant {subject_id} and '
            f'task {task_id if task_id else "<all>"}. '
            'All workflows require BOLD images. '
            f'Please check your BIDS filters: {config.execution.bids_filters}.'
        )

    config.loggers.workflow.info(
        f'Collected subject data:\n{yaml.dump(subject_data, default_flow_style=False, indent=4)}',
    )

    bids_info = pe.Node(
        BIDSInfo(
            bids_dir=config.execution.bids_dir,
            bids_validate=False,
            in_file=subject_data['bold'][0],
        ),
        name='bids_info',
    )

    summary = pe.Node(
        SubjectSummary(
            bold=subject_data['bold'],
            std_spaces=spaces.get_spaces(nonstandard=False),
            nstd_spaces=spaces.get_spaces(standard=False),
        ),
        name='summary',
        run_without_submitting=True,
    )
    workflow.connect([(bids_info, summary, [('subject', 'subject_id')])])

    about = pe.Node(
        AboutSummary(version=config.environment.version, command=' '.join(sys.argv)),
        name='about',
        run_without_submitting=True,
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            source_file=subject_data['bold'][0],
            base_directory=config.execution.output_dir,
            desc='summary',
            datatype='figures',
        ),
        name='ds_report_summary',
        run_without_submitting=True,
    )
    workflow.connect([(summary, ds_report_summary, [('out_report', 'in_file')])])

    ds_report_about = pe.Node(
        DerivativesDataSink(
            source_file=subject_data['bold'][0],
            base_directory=config.execution.output_dir,
            desc='about',
            datatype='figures',
        ),
        name='ds_report_about',
        run_without_submitting=True,
    )
    workflow.connect([(about, ds_report_about, [('out_report', 'in_file')])])

    # Append the functional section to the existing anatomical excerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = f"""
Functional data postprocessing

: For each of the {len(subject_data['bold'])} BOLD runs found per subject
(across all tasks and sessions), the following postprocessing was performed.
"""
    workflow.__desc__ += func_pre_desc

    for bold_file in subject_data['bold']:
        single_run_wf = init_single_run_wf(bold_file)
        workflow.add_nodes([single_run_wf])

    return clean_datasinks(workflow)


def init_single_run_wf(bold_file):
    """Set up a single-run workflow for fMRIPost-AROMA."""
    from fmriprep.utils.misc import estimate_bold_mem_usage
    from nipype.interfaces import utility as niu
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_aroma.utils.bids import collect_derivatives, extract_entities
    from fmripost_aroma.workflows.aroma import init_denoise_wf, init_ica_aroma_wf
    from fmripost_aroma.workflows.outputs import init_func_fit_reports_wf

    spaces = config.workflow.spaces
    omp_nthreads = config.nipype.omp_nthreads

    workflow = Workflow(name=_get_wf_name(bold_file, 'single_run'))
    workflow.__desc__ = ''

    bold_metadata = config.execution.layout.get_metadata(bold_file)
    mem_gb = estimate_bold_mem_usage(bold_file)[1]

    entities = extract_entities(bold_file)

    # Attempt to extract the associated fmap ID
    fmapid = None
    all_fmapids = config.execution.layout.get_fmapids(
        subject=entities['subject'],
        session=entities.get('session', None),
    )
    if all_fmapids:
        fmap_file = config.execution.layout.get_nearest(
            bold_file,
            to=all_fmapids,
            suffix='xfm',
            extension='.txt',
            strict=False,
            **{'from': 'boldref'},
        )
        fmapid = config.execution.layout.get_file(fmap_file).entities['to']

    functional_cache = defaultdict(list, {})
    if config.execution.derivatives:
        # Collect native-space derivatives and transforms
        functional_cache = collect_derivatives(
            raw_dataset=config.execution.layout,
            derivatives_dataset=None,
            entities=entities,
            fieldmap_id=fmapid,
            allow_multiple=False,
            spaces=None,
        )
        for deriv_dir in config.execution.derivatives.values():
            functional_cache = update_dict(
                functional_cache,
                collect_derivatives(
                    raw_dataset=None,
                    derivatives_dataset=deriv_dir,
                    entities=entities,
                    fieldmap_id=fmapid,
                    allow_multiple=False,
                    spaces=spaces,
                ),
            )

        if not functional_cache['bold_confounds']:
            if config.workflow.dummy_scans is None:
                raise ValueError(
                    'No confounds detected. '
                    'Automatical dummy scan detection cannot be performed. '
                    'Please set the `--dummy-scans` flag explicitly.'
                )

            # TODO: Calculate motion parameters from motion correction transform
            raise ValueError('Motion parameters cannot be extracted from transforms yet.')

    else:
        # Collect MNI152NLin6Asym:res-2 derivatives
        # Only derivatives dataset was passed in, so we expected standard-space derivatives
        functional_cache.update(
            collect_derivatives(
                raw_dataset=None,
                derivatives_dataset=config.execution.layout,
                entities=entities,
                fieldmap_id=fmapid,
                allow_multiple=False,
                spaces=spaces,
            ),
        )

    config.loggers.workflow.info(
        (
            f'Collected run data for {os.path.basename(bold_file)}:\n'
            f'{yaml.dump(functional_cache, default_flow_style=False, indent=4)}'
        ),
    )

    if config.workflow.dummy_scans is not None:
        skip_vols = config.workflow.dummy_scans
    else:
        if not functional_cache['bold_confounds']:
            raise ValueError(
                'No confounds detected. '
                'Automatical dummy scan detection cannot be performed. '
                'Please set the `--dummy-scans` flag explicitly.'
            )
        skip_vols = get_nss(functional_cache['bold_confounds'])

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_raw',
                'bold_confounds',
                'bold_mni152nlin6asym',
                'motion_xfm',
                'boldref2fmap_xfm',
                'boldref2anat_xfm',
                'anat2std_xfm',
                'bold_ref_file',
                'fmap',
                'bold_mask_native',
            ],
        ),
        name='inputnode',
    )
    inputnode.inputs.bold_raw = functional_cache['bold_raw']
    inputnode.inputs.bold_confounds = functional_cache['bold_confounds']
    inputnode.inputs.bold_mni152nlin6asym = functional_cache['bold_mni152nlin6asym']
    inputnode.inputs.bold_mask_native = functional_cache['bold_mask_native']
    # Transforms
    inputnode.inputs.bold_hmc = functional_cache['bold_hmc']
    inputnode.inputs.boldref2fmap = functional_cache['boldref2fmap']
    inputnode.inputs.boldref2anat = functional_cache['boldref2anat']
    inputnode.inputs.anat2mni152nlin6asym = functional_cache['anat2mni152nlin6asym']
    # Field maps
    inputnode.inputs.fmap = functional_cache['fmap']

    # Run ICA-AROMA
    ica_aroma_wf = init_ica_aroma_wf(bold_file=bold_file, metadata=bold_metadata, mem_gb=mem_gb)
    ica_aroma_wf.inputs.inputnode.skip_vols = skip_vols
    workflow.connect([(inputnode, ica_aroma_wf, [('bold_confounds', 'inputnode.confounds')])])

    mni6_buffer = pe.Node(niu.IdentityInterface(fields=['bold', 'bold_mask']), name='mni6_buffer')

    if ('bold_mni152nlin6asym' not in functional_cache) and ('bold_raw' in functional_cache):
        # Resample to MNI152NLin6Asym:res-2, for ICA-AROMA classification
        from fmriprep.workflows.bold.stc import init_bold_stc_wf
        from niworkflows.interfaces.header import ValidateImage
        from templateflow.api import get as get_template

        from fmripost_aroma.interfaces.misc import ApplyTransforms
        from fmripost_aroma.workflows.resample import init_bold_volumetric_resample_wf

        workflow.__desc__ += """\
Raw BOLD series were resampled to MNI152NLin6Asym:res-2, for ICA-AROMA classification.
"""

        validate_bold = pe.Node(
            ValidateImage(),
            name='validate_bold',
        )
        workflow.connect([(inputnode, validate_bold, [('bold_raw', 'in_file')])])

        stc_buffer = pe.Node(
            niu.IdentityInterface(fields=['bold_file']),
            name='stc_buffer',
        )
        run_stc = ('SliceTiming' in bold_metadata) and 'slicetiming' not in config.workflow.ignore
        if run_stc:
            bold_stc_wf = init_bold_stc_wf(
                mem_gb=mem_gb,
                metadata=bold_metadata,
                name='bold_stc_wf',
            )
            bold_stc_wf.inputs.inputnode.skip_vols = skip_vols
            workflow.connect([
                (validate_bold, bold_stc_wf, [('out_file', 'inputnode.bold_file')]),
                (bold_stc_wf, stc_buffer, [('outputnode.stc_file', 'bold_file')]),
            ])  # fmt:skip
        else:
            workflow.connect([(validate_bold, stc_buffer, [('out_file', 'bold_file')])])

        mni6_mask = str(
            get_template(
                'MNI152NLin6Asym',
                resolution=2,
                desc='brain',
                suffix='mask',
                extension=['.nii', '.nii.gz'],
            )
        )

        bold_MNI6_wf = init_bold_volumetric_resample_wf(
            metadata=bold_metadata,
            fieldmap_id=fmapid,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            jacobian='fmap-jacobian' not in config.workflow.ignore,
            name='bold_MNI6_wf',
        )
        bold_MNI6_wf.inputs.inputnode.resolution = '02'
        bold_MNI6_wf.inputs.inputnode.target_mask = mni6_mask
        bold_MNI6_wf.inputs.inputnode.target_ref_file = mni6_mask

        # Pass transforms and warped fieldmap to bold_MNI6_wf
        workflow.connect([
            (inputnode, bold_MNI6_wf, [
                ('bold_hmc', 'inputnode.motion_xfm'),
                ('boldref2fmap', 'inputnode.boldref2fmap_xfm'),
                ('boldref2anat', 'inputnode.boldref2anat_xfm'),
                ('anat2mni152nlin6asym', 'inputnode.anat2std_xfm'),
                # use mask as boldref?
                ('bold_mask_native', 'inputnode.bold_ref_file'),
            ]),
            # Resample BOLD to MNI152NLin6Asym, may duplicate bold_std_wf above
            (stc_buffer, bold_MNI6_wf, [('bold_file', 'inputnode.bold_file')]),
            (bold_MNI6_wf, mni6_buffer, [('outputnode.bold_file', 'bold')]),
        ])  # fmt:skip

        # Warp the mask as well
        mask_to_mni6 = pe.Node(
            ApplyTransforms(
                interpolation='GenericLabel',
                reference_image=mni6_mask,
                transforms=[
                    functional_cache['anat2mni152nlin6asym'],
                    functional_cache['boldref2anat'],
                ],
            ),
            name='mask_to_mni6',
        )
        workflow.connect([
            (inputnode, mask_to_mni6, [('bold_mask_native', 'input_image')]),
            (mask_to_mni6, mni6_buffer, [('output_image', 'bold_mask')]),
        ])  # fmt:skip

    elif 'bold_mni152nlin6asym' in functional_cache:
        workflow.__desc__ += """\
Preprocessed BOLD series in MNI152NLin6Asym:res-2 space were collected for ICA-AROMA
classification.
"""
        workflow.connect([
            (inputnode, mni6_buffer, [
                ('bold_mni152nlin6asym', 'bold'),
                ('bold_mask_native', 'bold_mask'),
            ]),
        ])  # fmt:skip

    else:
        raise ValueError('No valid BOLD series found for ICA-AROMA classification.')

    workflow.connect([
        (mni6_buffer, ica_aroma_wf, [
            ('bold', 'inputnode.bold_std'),
            ('bold_mask', 'inputnode.bold_mask_std'),
        ]),
    ])  # fmt:skip

    # Generate reportlets
    func_fit_reports_wf = init_func_fit_reports_wf(output_dir=config.execution.output_dir)
    func_fit_reports_wf.inputs.inputnode.source_file = bold_file
    workflow.connect([
        (inputnode, func_fit_reports_wf, [
            ('anat2mni152nlin6asym', 'inputnode.anat2std_xfm'),
            ('anat_dseg', 'inputnode.anat_dseg'),
        ]),
        (mni6_buffer, func_fit_reports_wf, [('bold', 'inputnode.bold_mni6')]),
    ])  # fmt:skip

    if config.workflow.denoise_method:
        # Now denoise the output-space BOLD data using ICA-AROMA
        denoise_wf = init_denoise_wf(bold_file=bold_file, metadata=bold_metadata)
        denoise_wf.inputs.inputnode.skip_vols = skip_vols
        denoise_wf.inputs.inputnode.space = 'MNI152NLin6Asym'
        denoise_wf.inputs.inputnode.res = '2'

        workflow.connect([
            (inputnode, denoise_wf, [('bold_confounds', 'inputnode.confounds_file')]),
            (mni6_buffer, denoise_wf, [
                ('bold', 'inputnode.bold_file'),
                ('bold_mask', 'inputnode.bold_mask'),
            ]),
            (ica_aroma_wf, denoise_wf, [
                ('outputnode.mixing', 'inputnode.mixing'),
                ('outputnode.aroma_features', 'inputnode.classifications'),
            ]),
        ])  # fmt:skip

    # Fill-in datasinks seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).inputs.base_directory = config.execution.output_dir
            workflow.get_node(node).inputs.source_file = bold_file

    return workflow


def _prefix(subid):
    return subid if subid.startswith('sub-') else f'sub-{subid}'


def clean_datasinks(workflow: pe.Workflow) -> pe.Workflow:
    """Overwrite ``out_path_base`` of smriprep's DataSinks."""
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = ''
    return workflow


def get_nss(confounds_file):
    """Get number of non-steady state volumes."""
    import numpy as np
    import pandas as pd

    df = pd.read_table(confounds_file)

    nss_cols = [c for c in df.columns if c.startswith('non_steady_state_outlier')]

    dummy_scans = 0
    if nss_cols:
        initial_volumes_df = df[nss_cols]
        dummy_scans = np.any(initial_volumes_df.to_numpy(), axis=1)
        dummy_scans = np.where(dummy_scans)[0]

        # reasonably assumes all NSS volumes are contiguous
        dummy_scans = int(dummy_scans[-1] + 1)

    return dummy_scans
