# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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
"""Fixtures for the CircleCI tests."""

import base64
import importlib.resources
import os
import re

import pytest


# Set up the commandline options as fixtures
@pytest.fixture(scope='session')
def data_dir():
    """Grab data directory."""
    test_data = importlib.resources.files('fmripost_template.tests') / 'data'
    with importlib.resources.as_file(test_data) as data:
        yield data


@pytest.fixture(scope='session', autouse=True)
def _fslicense(tmp_path_factory):
    """Set the FreeSurfer license as an environment variable."""
    working_dir = tmp_path_factory.mktemp('fslicense')
    FS_LICENSE = os.path.join(working_dir, 'license.txt')
    LICENSE_CODE = (
        'bWF0dGhldy5jaWVzbGFrQHBzeWNoLnVjc2IuZWR1CjIwNzA2CipDZmVWZEg1VVQ4clkKRlNCWVouVWtlVElDdwo='
    )
    with open(FS_LICENSE, 'w') as f:
        f.write(base64.b64decode(LICENSE_CODE).decode())

    os.putenv('FS_LICENSE', FS_LICENSE)
    return


@pytest.fixture(scope='session')
def base_config():
    from fmripost_template.tests.tests import mock_config

    return mock_config


@pytest.fixture(scope='session')
def base_ignore_list():
    """Create the standard ignore list used by fMRIPost-template."""
    return [
        'code',
        'stimuli',
        'sourcedata',
        'models',
        re.compile(r'^\.'),
        re.compile(r'sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(beh|dwi|eeg|ieeg|meg|perf)'),
    ]


@pytest.fixture(scope='session')
def minimal_ignore_list(base_ignore_list):
    """Create an ignore list that ignores full derivative files from ds000005."""
    base_ignore_list = base_ignore_list[:]
    files_to_ignore = [
        'sub-01/anat/sub-01_desc-brain_mask.json',
        'sub-01/anat/sub-01_desc-brain_mask.nii.gz',
        'sub-01/anat/sub-01_desc-preproc_T1w.json',
        'sub-01/anat/sub-01_desc-preproc_T1w.nii.gz',
        'sub-01/anat/sub-01_desc-ribbon_mask.json',
        'sub-01/anat/sub-01_desc-ribbon_mask.nii.gz',
        'sub-01/anat/sub-01_dseg.nii.gz',
        # 'sub-01/anat/sub-01_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5',
        # 'sub-01/anat/sub-01_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5',
        # 'sub-01/anat/sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
        # 'sub-01/anat/sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5',
        # 'sub-01/anat/sub-01_from-T1w_to-fsnative_mode-image_xfm.txt',
        # 'sub-01/anat/sub-01_from-fsnative_to-T1w_mode-image_xfm.txt',
        'sub-01/anat/sub-01_hemi-L_desc-reg_sphere.surf.gii',
        'sub-01/anat/sub-01_hemi-L_midthickness.surf.gii',
        'sub-01/anat/sub-01_hemi-L_pial.surf.gii',
        'sub-01/anat/sub-01_hemi-L_space-fsLR_desc-msmsulc_sphere.surf.gii',
        'sub-01/anat/sub-01_hemi-L_space-fsLR_desc-reg_sphere.surf.gii',
        'sub-01/anat/sub-01_hemi-L_sphere.surf.gii',
        'sub-01/anat/sub-01_hemi-L_sulc.shape.gii',
        'sub-01/anat/sub-01_hemi-L_thickness.shape.gii',
        'sub-01/anat/sub-01_hemi-L_white.surf.gii',
        'sub-01/anat/sub-01_hemi-R_desc-reg_sphere.surf.gii',
        'sub-01/anat/sub-01_hemi-R_midthickness.surf.gii',
        'sub-01/anat/sub-01_hemi-R_pial.surf.gii',
        'sub-01/anat/sub-01_hemi-R_space-fsLR_desc-msmsulc_sphere.surf.gii',
        'sub-01/anat/sub-01_hemi-R_space-fsLR_desc-reg_sphere.surf.gii',
        'sub-01/anat/sub-01_hemi-R_sphere.surf.gii',
        'sub-01/anat/sub-01_hemi-R_sulc.shape.gii',
        'sub-01/anat/sub-01_hemi-R_thickness.shape.gii',
        'sub-01/anat/sub-01_hemi-R_white.surf.gii',
        'sub-01/anat/sub-01_label-CSF_probseg.nii.gz',
        'sub-01/anat/sub-01_label-GM_probseg.nii.gz',
        'sub-01/anat/sub-01_label-WM_probseg.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin2009cAsym_desc-brain_mask.json',
        'sub-01/anat/sub-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.json',
        'sub-01/anat/sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin2009cAsym_dseg.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin2009cAsym_label-WM_probseg.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin6Asym_res-2_desc-brain_mask.json',
        'sub-01/anat/sub-01_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.json',
        'sub-01/anat/sub-01_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin6Asym_res-2_dseg.json',
        'sub-01/anat/sub-01_space-MNI152NLin6Asym_res-2_dseg.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin6Asym_res-2_label-CSF_probseg.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin6Asym_res-2_label-GM_probseg.nii.gz',
        'sub-01/anat/sub-01_space-MNI152NLin6Asym_res-2_label-WM_probseg.nii.gz',
        'sub-01/figures/',
        'sub-01/func/sub-01_task-mixedgamblestask_run-01_desc-confounds_timeseries.json',
        'sub-01/func/sub-01_task-mixedgamblestask_run-01_desc-confounds_timeseries.tsv',
        # 'sub-01/func/sub-01_task-mixedgamblestask_run-01_desc-coreg_boldref.json',
        # 'sub-01/func/sub-01_task-mixedgamblestask_run-01_desc-coreg_boldref.nii.gz',
        # 'sub-01/func/sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.json',
        # 'sub-01/func/sub-01_task-mixedgamblestask_run-01_desc-hmc_boldref.nii.gz',
        # (
        #     'sub-01/func/sub-01_task-mixedgamblestask_run-01_from-boldref_to-T1w_mode-image_'
        #     'desc-coreg_xfm.json'
        # ),
        # (
        #     'sub-01/func/sub-01_task-mixedgamblestask_run-01_from-boldref_to-T1w_mode-image_'
        #     'desc-coreg_xfm.txt'
        # ),
        # (
        #     'sub-01/func/sub-01_task-mixedgamblestask_run-01_from-orig_to-boldref_mode-image_'
        #     'desc-hmc_xfm.json'
        # ),
        # (
        #     'sub-01/func/sub-01_task-mixedgamblestask_run-01_from-orig_to-boldref_mode-image_'
        #     'desc-hmc_xfm.txt'
        # ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-01_space-MNI152NLin2009cAsym_'
            'boldref.nii.gz'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-01_space-MNI152NLin2009cAsym_'
            'desc-brain_mask.nii.gz'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-01_space-MNI152NLin2009cAsym_'
            'desc-preproc_bold.json'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-01_space-MNI152NLin2009cAsym_'
            'desc-preproc_bold.nii.gz'
        ),
        'sub-01/func/sub-01_task-mixedgamblestask_run-01_space-MNI152NLin6Asym_res-2_boldref.json',
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-01_space-MNI152NLin6Asym_'
            'res-2_boldref.nii.gz'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-01_space-MNI152NLin6Asym_'
            'res-2_desc-brain_mask.json'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-01_space-MNI152NLin6Asym_'
            'res-2_desc-brain_mask.nii.gz'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-01_space-MNI152NLin6Asym_'
            'res-2_desc-preproc_bold.json'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-01_space-MNI152NLin6Asym_'
            'res-2_desc-preproc_bold.nii.gz'
        ),
        'sub-01/func/sub-01_task-mixedgamblestask_run-02_desc-confounds_timeseries.json',
        'sub-01/func/sub-01_task-mixedgamblestask_run-02_desc-confounds_timeseries.tsv',
        # 'sub-01/func/sub-01_task-mixedgamblestask_run-02_desc-coreg_boldref.json',
        # 'sub-01/func/sub-01_task-mixedgamblestask_run-02_desc-coreg_boldref.nii.gz',
        # 'sub-01/func/sub-01_task-mixedgamblestask_run-02_desc-hmc_boldref.json',
        # 'sub-01/func/sub-01_task-mixedgamblestask_run-02_desc-hmc_boldref.nii.gz',
        # (
        #     'sub-01/func/sub-01_task-mixedgamblestask_run-02_from-boldref_to-T1w_mode-image_'
        #     'desc-coreg_xfm.json'
        # ),
        # (
        #     'sub-01/func/sub-01_task-mixedgamblestask_run-02_from-boldref_to-T1w_mode-image_'
        #     'desc-coreg_xfm.txt'
        # ),
        # (
        #     'sub-01/func/sub-01_task-mixedgamblestask_run-02_from-orig_to-boldref_mode-image_'
        #     'desc-hmc_xfm.json'
        # ),
        # (
        #     'sub-01/func/sub-01_task-mixedgamblestask_run-02_from-orig_to-boldref_mode-image_'
        #     'desc-hmc_xfm.txt'
        # ),
        'sub-01/func/sub-01_task-mixedgamblestask_run-02_space-MNI152NLin2009cAsym_boldref.nii.gz',
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-02_space-MNI152NLin2009cAsym_'
            'desc-brain_mask.nii.gz'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-02_space-MNI152NLin2009cAsym_'
            'desc-preproc_bold.json'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-02_space-MNI152NLin2009cAsym_'
            'desc-preproc_bold.nii.gz'
        ),
        'sub-01/func/sub-01_task-mixedgamblestask_run-02_space-MNI152NLin6Asym_res-2_boldref.json',
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-02_space-MNI152NLin6Asym_'
            'res-2_boldref.nii.gz'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-02_space-MNI152NLin6Asym_'
            'res-2_desc-brain_mask.json'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-02_space-MNI152NLin6Asym_'
            'res-2_desc-brain_mask.nii.gz'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-02_space-MNI152NLin6Asym_'
            'res-2_desc-preproc_bold.json'
        ),
        (
            'sub-01/func/sub-01_task-mixedgamblestask_run-02_space-MNI152NLin6Asym_'
            'res-2_desc-preproc_bold.nii.gz'
        ),
    ]
    regex = '|'.join(files_to_ignore)
    base_ignore_list.append(re.compile(regex))
    return base_ignore_list


@pytest.fixture(scope='session')
def full_ignore_list(base_ignore_list):
    """Create an ignore list that ignores minimal derivative files from ds000005.

    The 'full' run didn't include func-space outputs, so there's no func-space brain mask.
    """
    base_ignore_list = base_ignore_list[:]
    files_to_ignore = [
        'sub-01/func/sub-01_task-mixedgamblestask_run-01_desc-brain_mask.nii.gz',
    ]
    regex = '|'.join(files_to_ignore)
    base_ignore_list.append(re.compile(regex))
    return base_ignore_list
