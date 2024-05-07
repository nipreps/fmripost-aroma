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
import warnings
from copy import deepcopy

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from packaging.version import Version

from fmripost_aroma import config
from fmripost_aroma.interfaces.bids import DerivativesDataSink
from fmripost_aroma.interfaces.reportlets import AboutSummary, SubjectSummary


def init_fmripost_aroma_wf():
    """Build *fMRIPost-AROMA*'s pipeline.

    This workflow organizes the execution of FMRIPREP, with a sub-workflow for
    each subject.

    If FreeSurfer's ``recon-all`` is to be run, a corresponding folder is created
    and populated with any needed template subjects under the derivatives folder.

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
    from niworkflows.interfaces.bids import BIDSFreeSurferDir

    ver = Version(config.environment.version)

    fmripost_aroma_wf = Workflow(name=f'fmripost_aroma_{ver.major}_{ver.minor}_wf')
    fmripost_aroma_wf.base_dir = config.execution.work_dir

    freesurfer = config.workflow.run_reconall
    if freesurfer:
        fsdir = pe.Node(
            BIDSFreeSurferDir(
                derivatives=config.execution.output_dir,
                freesurfer_home=os.getenv('FREESURFER_HOME'),
                spaces=config.workflow.spaces.get_fs_spaces(),
                minimum_fs_version='7.0.0',
            ),
            name=f"fsdir_run_{config.execution.run_uuid.replace('-', '_')}",
            run_without_submitting=True,
        )
        if config.execution.fs_subjects_dir is not None:
            fsdir.inputs.subjects_dir = str(config.execution.fs_subjects_dir.absolute())

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

        if freesurfer:
            fmripost_aroma_wf.connect(
                fsdir,
                'subjects_dir',
                single_subject_wf,
                'inputnode.subjects_dir',
            )
        else:
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

    It collects and reports information about the subject, and prepares
    sub-workflows to perform anatomical and functional preprocessing.
    Anatomical preprocessing is performed in a single workflow, regardless of
    the number of sessions.
    Functional preprocessing is performed using a separate workflow for each
    individual BOLD series.

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

    Inputs
    ------
    subjects_dir : :obj:`str`
        FreeSurfer's ``$SUBJECTS_DIR``.

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
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSDataGrabber, BIDSInfo
    from niworkflows.interfaces.nilearn import NILEARN_VERSION
    from niworkflows.utils.misc import fix_multi_T1w_source_name

    from fmripost_aroma.utils.bids import collect_derivatives
    from fmripost_aroma.workflows.aroma import init_ica_aroma_wf

    workflow = Workflow(name=f'sub_{subject_id}_wf')
    workflow.__desc__ = f"""
Results included in this manuscript come from preprocessing
performed using *fMRIPost-AROMA* {config.environment.version}
(@fmriprep1; @fmriprep2; RRID:SCR_016216),
which is based on *Nipype* {config.environment.nipype_version}
(@nipype1; @nipype2; RRID:SCR_002502).

"""
    workflow.__postdesc__ = f"""

Many internal operations of *fMRIPost-AROMA* use
*Nilearn* {NILEARN_VERSION} [@nilearn, RRID:SCR_001362],
mostly within the functional processing workflow.
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

    subject_data = collect_derivatives(
        config.execution.layout,
        subject_id,
        task=config.execution.task_id,
        bids_filters=config.execution.bids_filters,
    )

    if 'flair' in config.workflow.ignore:
        subject_data['flair'] = []
    if 't2w' in config.workflow.ignore:
        subject_data['t2w'] = []

    anat_only = config.workflow.anat_only
    # Make sure we always go through these two checks
    if not anat_only and not subject_data['bold']:
        task_id = config.execution.task_id
        raise RuntimeError(
            f"No BOLD images found for participant {subject_id} and "
            f"task {task_id if task_id else '<all>'}. "
            "All workflows require BOLD images."
        )

    if subject_data['roi']:
        warnings.warn(
            f"Lesion mask {subject_data['roi']} found. "
            "Future versions of fMRIPost-AROMA will use alternative conventions. "
            "Please refer to the documentation before upgrading.",
            FutureWarning,
        )

    inputnode = pe.Node(niu.IdentityInterface(fields=['subjects_dir']), name='inputnode')

    bidssrc = pe.Node(
        BIDSDataGrabber(
            subject_data=subject_data,
            anat_only=config.workflow.anat_only,
            subject_id=subject_id,
        ),
        name='bidssrc',
    )

    bids_info = pe.Node(
        BIDSInfo(bids_dir=config.execution.bids_dir, bids_validate=False), name='bids_info'
    )

    summary = pe.Node(
        SubjectSummary(
            std_spaces=['MNI152NLin6Asym'],
            nstd_spaces=None,
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
        (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
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
        functional_cache = {}
        if config.execution.derivatives:
            # Collect native-space derivatives and warp them to MNI152NLin6Asym
            from fmripost_aroma.utils.bids import collect_derivatives, extract_entities

            entities = extract_entities(bold_file)

            for deriv_dir in config.execution.derivatives:
                functional_cache.update(
                    collect_derivatives(
                        derivatives_dir=deriv_dir,
                        entities=entities,
                    )
                )
        else:
            # Collect standard-space derivatives
            from fmripost_aroma.utils.bids import collect_derivatives

            ...

        ica_aroma_wf = init_ica_aroma_wf(
            bold_file=bold_file,
            precomputed=functional_cache,
        )
        ica_aroma_wf.__desc__ = func_pre_desc + (ica_aroma_wf.__desc__ or '')

        # fmt:off
        workflow.connect([
            (inputnode, ica_aroma_wf, [
                ('bold_std', 'inputnode.bold_std'),
                ('bold_mask_std', 'inputnode.bold_mask_std'),
                ('movpar_file', 'inputnode.movpar_file'),
                ('name_source', 'inputnode.name_source'),
                ('skip_vols', 'inputnode.skip_vols'),
                ('spatial_reference', 'inputnode.spatial_reference'),
            ]),
        ])
        # fmt:on

    return clean_datasinks(workflow)


def _prefix(subid):
    return subid if subid.startswith('sub-') else f'sub-{subid}'


def clean_datasinks(workflow: pe.Workflow) -> pe.Workflow:
    """Overwrite ``out_path_base`` of smriprep's DataSinks."""
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = ''
    return workflow
