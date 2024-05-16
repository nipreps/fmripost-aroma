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

import sys
from copy import deepcopy

from nipype.pipeline import engine as pe
from packaging.version import Version

from fmripost_aroma import config
from fmripost_aroma.interfaces.bids import DerivativesDataSink
from fmripost_aroma.interfaces.reportlets import AboutSummary, SubjectSummary
from fmripost_aroma.utils.utils import _get_wf_name, update_dict
from fmripost_aroma.workflows.resampling import init_resample_volumetric_wf


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
            config.execution.fmripost_aroma_dir
            / f'sub-{subject_id}'
            / 'log'
            / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        fmripost_aroma_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.fmripost_aroma_dir
            / f'sub-{subject_id}'
            / 'log'
            / config.execution.run_uuid
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
    from niworkflows.interfaces.bids import BIDSDataGrabber, BIDSInfo
    from niworkflows.interfaces.nilearn import NILEARN_VERSION
    from niworkflows.utils.misc import fix_multi_T1w_source_name
    from niworkflows.utils.spaces import Reference

    from fmripost_aroma.utils.bids import collect_derivatives, extract_entities
    from fmripost_aroma.workflows.aroma import init_denoise_wf, init_ica_aroma_wf

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

    if config.execution.derivatives:
        # Raw dataset + derivatives dataset
        config.loggers.workflow.info('Raw+derivatives workflow mode enabled')
        subject_data = collect_derivatives(
            raw_dataset=config.execution.layout,
            derivatives_dataset=None,
            entities=config.execution.bids_filters,
            fieldmap_id=None,
            allow_multiple=True,
        )
        subject_data['bold'] = listify(subject_data['bold_raw'])
    else:
        # Derivatives dataset only
        config.loggers.workflow.info('Derivatives-only workflow mode enabled')
        subject_data = collect_derivatives(
            raw_dataset=None,
            derivatives_dataset=config.execution.layout,
            entities=config.execution.bids_filters,
            fieldmap_id=None,
            allow_multiple=True,
        )
        # Patch standard-space BOLD files into 'bold' key
        subject_data['bold'] = listify(subject_data['bold_std'])

    # Make sure we always go through these two checks
    if not subject_data['bold']:
        task_id = config.execution.task_id
        raise RuntimeError(
            f"No BOLD images found for participant {subject_id} and "
            f"task {task_id if task_id else '<all>'}. "
            "All workflows require BOLD images. "
            f"Please check your BIDS filters: {config.execution.bids_filters}."
        )

    bidssrc = pe.Node(
        BIDSDataGrabber(
            subject_data=subject_data,
            subject_id=subject_id,
            anat_only=False,
        ),
        name='bidssrc',
    )

    bids_info = pe.Node(
        BIDSInfo(bids_dir=config.execution.bids_dir, bids_validate=False),
        name='bids_info',
    )

    summary = pe.Node(
        SubjectSummary(
            std_spaces=['MNI152NLin6Asym'],
            nstd_spaces=[],
        ),
        name='summary',
        run_without_submitting=True,
    )

    about = pe.Node(
        AboutSummary(version=config.environment.version, command=' '.join(sys.argv)),
        name='about',
        run_without_submitting=True,
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmripost_aroma_dir,
            desc='summary',
            datatype='figures',
            dismiss_entities=('echo',),
        ),
        name='ds_report_summary',
        run_without_submitting=True,
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmripost_aroma_dir,
            desc='about',
            datatype='figures',
            dismiss_entities=('echo',),
        ),
        name='ds_report_about',
        run_without_submitting=True,
    )

    workflow.connect([
        (bidssrc, bids_info, [(('t1w', fix_multi_T1w_source_name), 'in_file')]),
        # Reporting connections
        (bidssrc, summary, [('t1w', 't1w'), ('t2w', 't2w'), ('bold', 'bold')]),
        (bids_info, summary, [('subject', 'subject_id')]),
        (bidssrc, ds_report_summary, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (bidssrc, ds_report_about, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (about, ds_report_about, [('out_report', 'in_file')]),
    ])  # fmt:skip

    # Append the functional section to the existing anatomical excerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = f"""
Functional data postprocessing

: For each of the {len(subject_data['bold'])} BOLD runs found per subject
(across all tasks and sessions), the following postprocessing was performed.
"""

    for bold_file in subject_data['bold']:
        bold_metadata = config.execution.layout.get_metadata(bold_file)
        ica_aroma_wf = init_ica_aroma_wf(bold_file=bold_file, metadata=bold_metadata)
        ica_aroma_wf.__desc__ = func_pre_desc + (ica_aroma_wf.__desc__ or '')

        entities = extract_entities(bold_file)

        functional_cache = {}
        if config.execution.derivatives:
            # Collect native-space derivatives and transforms
            for deriv_dir in config.execution.derivatives.values():
                functional_cache = update_dict(
                    functional_cache,
                    collect_derivatives(
                        raw_dataset=None,
                        derivatives_dataset=deriv_dir,
                        entities=entities,
                        fieldmap_id=None,
                        allow_multiple=False,
                    ),
                )

            if not functional_cache['confounds']:
                if config.workflow.dummy_scans is None:
                    raise ValueError(
                        'No confounds detected. '
                        'Automatical dummy scan detection cannot be performed. '
                        'Please set the `--dummy-scans` flag explicitly.'
                    )

                # TODO: Calculate motion parameters from motion correction transform
                raise ValueError('Motion parameters cannot be extracted from transforms yet.')

            # Resample to MNI152NLin6Asym:res-2, for ICA-AROMA classification
            resample_raw_wf = init_resample_volumetric_wf(
                bold_file=bold_file,
                precomputed=functional_cache,
                space=Reference.from_string('MNI152NLin6Asym:res-2')[0],
                name=_get_wf_name(bold_file, 'resample_raw'),
            )
            resample_raw_wf.inputs.inputnode.bold_file = bold_file
            workflow.connect([
                (resample_raw_wf, ica_aroma_wf, [
                    ('outputnode.bold_std', 'inputnode.bold_std'),
                    ('outputnode.bold_mask_std', 'inputnode.bold_mask_std'),
                ]),
            ])  # fmt:skip
        else:
            # Collect MNI152NLin6Asym:res-2 derivatives
            # Only derivatives dataset was passed in, so we expected standard-space derivatives
            functional_cache.update(
                collect_derivatives(
                    raw_dataset=None,
                    derivatives_dataset=config.execution.layout,
                    entities=entities,
                    fieldmap_id=None,
                    allow_multiple=False,
                ),
            )
            ica_aroma_wf.inputs.inputnode.bold_std = functional_cache['bold_std']
            ica_aroma_wf.inputs.inputnode.bold_mask_std = functional_cache['bold_mask_std']
            workflow.add_nodes([ica_aroma_wf])

        if config.workflow.dummy_scans is not None:
            skip_vols = config.workflow.dummy_scans
        else:
            skip_vols = get_nss(functional_cache['confounds'])

        ica_aroma_wf.inputs.inputnode.confounds = functional_cache['confounds']
        ica_aroma_wf.inputs.inputnode.skip_vols = skip_vols

        if config.workflow.denoise_method:
            for space in spaces:
                # Warp each BOLD run to requested output spaces
                resample_to_space_wf = init_resample_volumetric_wf(
                    bold_file=bold_file,
                    functional_cache=functional_cache,
                    space=space,
                    name=_get_wf_name(bold_file, f'resample_{space}'),
                )

                # Now denoise the output-space BOLD data using ICA-AROMA
                denoise_wf = init_denoise_wf(bold_file=bold_file)
                denoise_wf.inputs.inputnode.bold_mask = functional_cache['bold_mask']
                denoise_wf.inputs.inputnode.skip_vols = functional_cache['skip_vols']
                workflow.connect([
                    (resample_to_space_wf, denoise_wf, [
                        ('bold_std', 'inputnode.bold_file'),
                        ('bold_mask_std', 'inputnode.bold_mask'),
                        ('spatial_reference', 'inputnode.spatial_reference'),
                    ]),
                    (ica_aroma_wf, denoise_wf, [
                        ('outputnode.aroma_confounds', 'inputnode.confounds'),
                    ]),
                ])  # fmt:skip

    return clean_datasinks(workflow)


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

    nss_cols = [c for c in df.columns if c.startswith("non_steady_state_outlier")]

    dummy_scans = 0
    if nss_cols:
        initial_volumes_df = df[nss_cols]
        dummy_scans = np.any(initial_volumes_df.to_numpy(), axis=1)
        dummy_scans = np.where(dummy_scans)[0]

        # reasonably assumes all NSS volumes are contiguous
        dummy_scans = int(dummy_scans[-1] + 1)

    return dummy_scans
