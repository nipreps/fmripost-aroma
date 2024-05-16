"""Utility functions for tests."""

import os
from pathlib import Path


def get_test_data_path():
    """Return the path to test datasets, terminated with separator.

    Test-related data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return Path(__file__).resolve().parent.parent.parent.parent / 'tests' / 'data')
