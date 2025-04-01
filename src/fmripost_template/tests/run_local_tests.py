#!/usr/bin/env python3
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
"""Run tests locally by calling Docker."""

import argparse
import os
import subprocess


def _get_parser():
    """Parse command line inputs for tests.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-k',
        dest='test_regex',
        metavar='PATTERN',
        type=str,
        help='Test pattern.',
        required=False,
        default=None,
    )
    parser.add_argument(
        '-m',
        dest='test_mark',
        metavar='LABEL',
        type=str,
        help='Test mark label.',
        required=False,
        default=None,
    )
    return parser


def run_command(command, env=None):
    """Run a given shell command with certain environment variables set.

    Keep this out of the real fmripost_template code so that devs don't need
    to install fMRIPost-template to run tests.
    """
    merged_env = os.environ
    if env:
        merged_env.update(env)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        env=merged_env,
    )
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break

    if process.returncode != 0:
        raise RuntimeError(
            f'Non zero return code: {process.returncode}\n{command}\n\n{process.stdout.read()}'
        )


def run_tests(test_regex, test_mark):
    """Run the tests."""
    local_patch = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mounted_code = '/usr/local/miniconda/lib/python3.10/site-packages/fmripost_template'
    run_str = 'docker run --rm -ti '
    run_str += f'-v {local_patch}:{mounted_code} '
    run_str += '--entrypoint pytest '
    run_str += 'pennlinc/fmripost_template:unstable '
    run_str += (
        f'{mounted_code}/fmripost_template '
        f'--data_dir={mounted_code}/fmripost_template/tests/test_data '
        f'--output_dir={mounted_code}/fmripost_template/tests/pytests/out '
        f'--working_dir={mounted_code}/fmripost_template/tests/pytests/work '
    )
    if test_regex:
        run_str += f'-k {test_regex} '
    elif test_mark:
        run_str += f'-rP -o log_cli=true -m {test_mark} '

    run_command(run_str)


def _main(argv=None):
    """Run the tests."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    run_tests(**kwargs)


if __name__ == '__main__':
    _main()
