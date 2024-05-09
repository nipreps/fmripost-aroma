"""Fixtures for the CircleCI tests."""

import base64
import os

import pytest


def pytest_addoption(parser):
    """Collect pytest parameters for running tests."""
    parser.addoption(
        '--data_dir',
        action='store',
        default='/home/runner/work/fmripost-aroma/fmripost-aroma/src/fmripost_aroma/tests/data',
    )
    parser.addoption(
        '--output_dir',
        action='store',
        default='/home/runner/work/pytests/out',
    )
    parser.addoption(
        '--working_dir',
        action='store',
        default='/home/runner/work/pytests/work',
    )


# Set up the commandline options as fixtures
@pytest.fixture(scope='session')
def data_dir(request):
    """Grab data directory."""
    return request.config.getoption('--data_dir')


@pytest.fixture(scope='session')
def working_dir(request):
    """Grab working directory."""
    workdir = request.config.getoption('--working_dir')
    os.makedirs(workdir, exist_ok=True)
    return workdir


@pytest.fixture(scope='session')
def output_dir(request):
    """Grab output directory."""
    outdir = request.config.getoption('--output_dir')
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture(scope='session', autouse=True)
def _fslicense(working_dir):
    """Set the FreeSurfer license as an environment variable."""
    FS_LICENSE = os.path.join(working_dir, 'license.txt')
    os.environ['FS_LICENSE'] = FS_LICENSE
    LICENSE_CODE = (
        'bWF0dGhldy5jaWVzbGFrQHBzeWNoLnVjc2IuZWR1CjIwNzA2CipDZmVWZEg1VVQ4clkKRlNCWVouVWtlVElDdwo='
    )
    with open(FS_LICENSE, 'w') as f:
        f.write(base64.b64decode(LICENSE_CODE).decode())


@pytest.fixture(scope='session')
def base_config():
    from fmripost_aroma.tests.tests import mock_config

    return mock_config


@pytest.fixture(scope='session')
def base_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Test parser')
    return parser
