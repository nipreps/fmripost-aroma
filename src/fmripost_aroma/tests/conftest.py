"""Fixtures for the CircleCI tests."""

import base64
import importlib.resources
import os

import pytest


# Set up the commandline options as fixtures
@pytest.fixture(scope='session')
def data_dir():
    """Grab data directory."""
    test_data = importlib.resources.files('fmripost_aroma.tests') / 'data'
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
    from fmripost_aroma.tests.tests import mock_config

    return mock_config


@pytest.fixture(scope='session')
def base_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Test parser')
    return parser
