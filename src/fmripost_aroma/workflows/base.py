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

import bids
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.utils.connections import listify
from packaging.version import Version

from fmripost_aroma import config
from fmripost_aroma.interfaces.bids import DerivativesDataSink
from fmripost_aroma.interfaces.reports import AboutSummary, SubjectSummary
from fmripost_aroma.workflows.aroma import init_ica_aroma_wf


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

    fmriprep_wf = Workflow(name=f'fmriprep_{ver.major}_{ver.minor}_wf')
    fmriprep_wf.base_dir = config.execution.work_dir

    freesurfer = config.workflow.run_reconall
    if freesurfer:
        fsdir = pe.Node(
            BIDSFreeSurferDir(
                derivatives=config.execution.output_dir,
                freesurfer_home=os.getenv('FREESURFER_HOME'),
                spaces=config.workflow.spaces.get_fs_spaces(),
                minimum_fs_version="7.0.0",
            ),
            name=f"fsdir_run_{config.execution.run_uuid.replace('-', '_')}",
            run_without_submitting=True,
        )
        if config.execution.fs_subjects_dir is not None:
            fsdir.inputs.subjects_dir = str(config.execution.fs_subjects_dir.absolute())

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        single_subject_wf.config['execution']['crashdump_dir'] = str(
            config.execution.fmriprep_dir / f"sub-{subject_id}" / "log" / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        if freesurfer:
            fmriprep_wf.connect(fsdir, 'subjects_dir', single_subject_wf, 'inputnode.subjects_dir')
        else:
            fmriprep_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.fmriprep_dir / f"sub-{subject_id}" / 'log' / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / 'fmripost_aroma.toml')

    return fmriprep_wf


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

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSDataGrabber, BIDSInfo
    from niworkflows.interfaces.nilearn import NILEARN_VERSION
    from niworkflows.utils.bids import collect_data
    from niworkflows.utils.misc import fix_multi_T1w_source_name
    from niworkflows.utils.spaces import Reference

    from fmripost_aroma.workflows.bold.base import init_bold_wf

    workflow = Workflow(name=f'sub_{subject_id}_wf')
    workflow.__desc__ = """
Results included in this manuscript come from preprocessing
performed using *fMRIPrep* {fmriprep_ver}
(@fmriprep1; @fmriprep2; RRID:SCR_016216),
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

""".format(
        fmriprep_ver=config.environment.version, nipype_ver=config.environment.nipype_version
    )
    workflow.__postdesc__ = """

Many internal operations of *fMRIPrep* use
*Nilearn* {nilearn_ver} [@nilearn, RRID:SCR_001362],
mostly within the functional processing workflow.
For more details of the pipeline, see [the section corresponding
to workflows in *fMRIPrep*'s documentation]\
(https://fmripost_aroma.readthedocs.io/en/latest/workflows.html \
"FMRIPrep's documentation").


### Copyright Waiver

The above boilerplate text was automatically generated by fMRIPrep
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0]\
(https://creativecommons.org/publicdomain/zero/1.0/) license.

### References

""".format(
        nilearn_ver=NILEARN_VERSION
    )

    subject_data = collect_data(
        config.execution.layout,
        subject_id,
        task=config.execution.task_id,
        echo=config.execution.echo_idx,
        bids_filters=config.execution.bids_filters,
    )[0]

    if 'flair' in config.workflow.ignore:
        subject_data['flair'] = []
    if 't2w' in config.workflow.ignore:
        subject_data['t2w'] = []

    anat_only = config.workflow.anat_only
    # Make sure we always go through these two checks
    if not anat_only and not subject_data['bold']:
        task_id = config.execution.task_id
        raise RuntimeError(
            "No BOLD images found for participant {} and task {}. "
            "All workflows require BOLD images.".format(
                subject_id, task_id if task_id else '<all>'
            )
        )

    if subject_data['roi']:
        warnings.warn(
            f"Lesion mask {subject_data['roi']} found. "
            "Future versions of fMRIPrep will use alternative conventions. "
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
            std_spaces=spaces.get_spaces(nonstandard=False),
            nstd_spaces=spaces.get_spaces(standard=False),
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
            base_directory=config.execution.fmriprep_dir,
            desc='summary',
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name='ds_report_summary',
        run_without_submitting=True,
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.fmriprep_dir,
            desc='about',
            datatype="figures",
            dismiss_entities=("echo",),
        ),
        name='ds_report_about',
        run_without_submitting=True,
    )

    # fmt:off
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
    ])
    # fmt:on

    # Append the functional section to the existing anatomical excerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = """
Functional data preprocessing

: For each of the {num_bold} BOLD runs found per subject (across all
tasks and sessions), the following preprocessing was performed.
""".format(
        num_bold=len(subject_data['bold'])
    )

    for bold_series in subject_data['bold']:
        bold_series = sorted(listify(bold_series))
        bold_file = bold_series[0]

        functional_cache = {}
        if config.execution.derivatives:
            from fmripost_aroma.utils.bids import collect_derivatives, extract_entities

            entities = extract_entities(bold_series)

            for deriv_dir in config.execution.derivatives:
                functional_cache.update(
                    collect_derivatives(
                        derivatives_dir=deriv_dir,
                        entities=entities,
                    )
                )

        ica_aroma_wf = init_ica_aroma_wf(
            bold_file=bold_file,
            precomputed=functional_cache,
        )
        ica_aroma_wf.__desc__ = func_pre_desc + (ica_aroma_wf.__desc__ or "")

        # fmt:off
        workflow.connect([
            (inputnode, ica_aroma_wf, [
                ('bold_std', 'inputnode.bold_std'),
                ("bold_mask_std", "inputnode.bold_mask_std"),
                ("movpar_file", "inputnode.movpar_file"),
                ("name_source", "inputnode.name_source"),
                ("skip_vols", "inputnode.skip_vols"),
                ("spatial_reference", "inputnode.spatial_reference"),
            ]),
        ])
        # fmt:on

    return clean_datasinks(workflow)


def _prefix(subid):
    return subid if subid.startswith('sub-') else f'sub-{subid}'


def clean_datasinks(workflow: pe.Workflow) -> pe.Workflow:
    # Overwrite ``out_path_base`` of smriprep's DataSinks
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = ""
    return workflow
