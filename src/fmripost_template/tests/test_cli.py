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
"""Command-line interface tests.

The tests in this file run the full fMRIPost-template workflow on test data
and check the outputs against a list of expected files.
"""

import os
import sys
from unittest.mock import patch

import pytest

from fmripost_template.cli import run
from fmripost_template.cli.parser import parse_args
from fmripost_template.cli.workflow import build_boilerplate, build_workflow
from fmripost_template.reports.core import generate_reports
from fmripost_template.tests.utils import (
    check_generated_files,
    download_test_data,
    get_test_data_path,
)
from fmripost_template.utils.bids import write_derivative_description


@pytest.mark.integration
def test_ds000001(data_dir, output_dir, working_dir):
    """Run fMRIPost-template on ds000001 fMRIPrep derivatives."""
    test_name = 'test_ds000001'

    fmriprep_dir = download_test_data('ds000001', data_dir)
    out_dir = os.path.join(output_dir, test_name)

    parameters = [
        fmriprep_dir,
        out_dir,
        'participant',
        '--work-dir',
        working_dir,
    ]
    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
    )


def _run_and_generate(test_name, parameters, test_main=True):
    from fmripost_template import config

    parameters.append('--clean-workdir')
    parameters.append('--stop-on-first-crash')
    parameters.append('--notrack')
    parameters.append('-v')

    if test_main:
        # This runs, but for some reason doesn't count toward coverage.
        argv = ['fmripost_template'] + parameters
        with patch.object(sys, 'argv', argv):
            with pytest.raises(SystemExit) as e:
                run.main()

            assert e.value.code == 0
    else:
        # XXX: I want to drop this option and use the main function,
        # but the main function doesn't track coverage correctly.
        parse_args(parameters)
        config_file = config.execution.work_dir / f'config-{config.execution.run_uuid}.toml'
        config.loggers.cli.warning(f'Saving config file to {config_file}')
        config.to_filename(config_file)

        retval = build_workflow(config_file, retval={})
        wf = retval['workflow']
        wf.run()
        write_derivative_description(
            input_dir=config.execution.bids_dir,
            output_dir=config.execution.output_dir,
            dataset_links={},
        )

        build_boilerplate(str(config_file), wf)
        session_list = (
            config.execution.bids_filters.get('bold', {}).get('session')
            if config.execution.bids_filters
            else None
        )
        generate_reports(
            subject_list=config.execution.participant_label,
            output_dir=config.execution.output_dir,
            run_uuid=config.execution.run_uuid,
            session_list=session_list,
        )

    output_list_file = os.path.join(get_test_data_path(), f'{test_name}_outputs.txt')
    check_generated_files(config.execution.output_dir, output_list_file)
