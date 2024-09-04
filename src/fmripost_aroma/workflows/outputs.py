# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
"""Writing out derivative files."""

from __future__ import annotations

from fmriprep.utils.bids import dismiss_echo
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.utils.images import dseg_label

from fmripost_aroma.config import DEFAULT_MEMORY_MIN_GB
from fmripost_aroma.interfaces.bids import DerivativesDataSink
from fmripost_aroma.interfaces.misc import ApplyTransforms


def init_func_fit_reports_wf(
    *,
    output_dir: str,
    name='func_fit_reports_wf',
) -> pe.Workflow:
    """Set up a battery of datasinks to store reports in the right location.

    Parameters
    ----------
    freesurfer : :obj:`bool`
        FreeSurfer was enabled
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: func_fit_reports_wf)

    Inputs
    ------
    source_file
        Input BOLD images
    """
    from nireports.interfaces.reporting.base import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )
    from templateflow.api import get as get_template

    from fmripost_aroma.interfaces.nilearn import MeanImage

    workflow = pe.Workflow(name=name)

    inputfields = [
        'source_file',
        'bold_mni6',
        'anat_dseg',
        'anat2std_xfm',
    ]
    inputnode = pe.Node(niu.IdentityInterface(fields=inputfields), name='inputnode')

    # Average the BOLD image over time
    calculate_mean_bold = pe.Node(
        MeanImage(),
        name='calculate_mean_bold',
        mem_gb=1,
    )
    workflow.connect([(inputnode, calculate_mean_bold, [('bold_mni6', 'bold_file')])])

    # Warp the tissue segmentation to MNI
    dseg_to_mni6 = pe.Node(
        ApplyTransforms(interpolation='GenericLabel'),
        name='dseg_to_mni6',
        mem_gb=1,
    )
    workflow.connect([
        (inputnode, dseg_to_mni6, [
            ('anat_dseg', 'input_image'),
            ('anat2std_xfm', 'transforms'),
            ('bold_mni6', 'reference_image'),
        ]),
    ])  # fmt:skip

    mni6_wm = pe.Node(
        niu.Function(function=dseg_label),
        name='mni6_wm',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    mni6_wm.inputs.label = 2  # BIDS default is WM=2
    workflow.connect([(dseg_to_mni6, mni6_wm, [('output_image', 'in_seg')])])

    # EPI-MNI registration
    epi_mni_report = pe.Node(
        SimpleBeforeAfter(
            after=str(
                get_template(
                    'MNI152NLin6Asym',
                    resolution=2,
                    desc='brain',
                    suffix='T1w',
                    extension=['.nii', '.nii.gz'],
                )
            ),
            before_label='EPI',
            after_label='MNI152NLin6Asym',
            dismiss_affine=True,
        ),
        name='epi_mni_report',
        mem_gb=0.1,
    )
    workflow.connect([
        (calculate_mean_bold, epi_mni_report, [('out_file', 'before')]),
        (mni6_wm, epi_mni_report, [('out', 'wm_seg')]),
    ])  # fmt:skip

    ds_epi_mni_report = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='normalization',
            suffix='bold',
            datatype='figures',
            dismiss_entities=dismiss_echo(),
        ),
        name='ds_epi_mni_report',
    )
    workflow.connect([
        (inputnode, ds_epi_mni_report, [('source_file', 'source_file')]),
        (epi_mni_report, ds_epi_mni_report, [('out_report', 'in_file')]),
    ])  # fmt:skip

    return workflow
