"""Command-line interface tests."""

import json
import os
import sys
from unittest.mock import patch

import pytest

from fmripost_aroma.cli import run
from fmripost_aroma.cli.parser import parse_args
from fmripost_aroma.cli.workflow import build_boilerplate, build_workflow
from fmripost_aroma.reports.core import generate_reports
from fmripost_aroma.tests.utils import (
    check_generated_files,
    download_test_data,
    get_test_data_path,
)
from fmripost_aroma.utils.bids import write_derivative_description


@pytest.mark.integration
@pytest.mark.ds005115_deriv_only
def test_ds005115_deriv_only(data_dir, output_dir, working_dir):
    """Run fMRIPost-AROMA on ds005115 fMRIPrep derivatives with MNI152NLin6Asym-space data."""
    test_name = 'test_ds005115_deriv_only'

    fmriprep_dir = download_test_data('ds005115_deriv_mni6', data_dir)
    out_dir = os.path.join(output_dir, test_name)

    with open(os.path.join(out_dir, 'filter.json'), 'w') as f:
        json.dump({'task': ['mixedgamblestask']}, f)

    parameters = [
        fmriprep_dir,
        out_dir,
        'participant',
        '--work-dir',
        working_dir,
        '--denoising-method',
        'aggr',
        'nonaggr',
        'orthaggr',
        '--bids-filter-file',
        os.path.join(out_dir, 'filter.json'),
    ]
    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
    )


@pytest.mark.integration
@pytest.mark.ds005115_deriv_and_raw
def test_ds005115_deriv_and_raw(data_dir, output_dir, working_dir):
    """Run fMRIPost-AROMA on ds005115 raw BIDS + fMRIPrep derivatives w/o MNI152NLin6Asym data."""
    test_name = 'test_ds005115_deriv_and_raw'

    raw_dir = download_test_data('ds005115_raw', data_dir)
    fmriprep_dir = download_test_data('ds005115_deriv_no_mni6', data_dir)
    out_dir = os.path.join(output_dir, test_name)

    parameters = [
        raw_dir,
        out_dir,
        'participant',
        '--derivatives',
        f'fmriprep={fmriprep_dir}',
        '--work-dir',
        working_dir,
        '--denoising-method',
        'aggr',
        'nonaggr',
        'orthaggr',
    ]
    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
    )


@pytest.mark.integration
@pytest.mark.ds005115_resampling_and_raw
def test_ds005115_resampling_and_raw(data_dir, output_dir, working_dir):
    """Run fMRIPost-AROMA on ds005115 raw BIDS + resampling-level fMRIPrep derivatives."""
    test_name = 'test_ds005115_resampling_and_raw'

    raw_dir = download_test_data('ds005115_raw', data_dir)
    fmriprep_dir = download_test_data('ds005115_resampling', data_dir)
    out_dir = os.path.join(output_dir, test_name)

    parameters = [
        raw_dir,
        out_dir,
        'participant',
        '--derivatives',
        f'fmriprep={fmriprep_dir}',
        '--work-dir',
        working_dir,
        '--denoising-method',
        'aggr',
        'nonaggr',
        'orthaggr',
    ]
    _run_and_generate(
        test_name=test_name,
        parameters=parameters,
    )


def _run_and_generate(test_name, parameters, test_main=True):
    from fmripost_aroma import config

    parameters.append('--clean-workdir')
    parameters.append('--stop-on-first-crash')
    parameters.append('--notrack')
    parameters.append('-v')

    if test_main:
        # This runs, but for some reason doesn't count toward coverage.
        argv = ['fmripost_aroma'] + parameters
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
            config.execution.bids_filters.get('session') if config.execution.bids_filters else None
        )
        generate_reports(
            subject_list=config.execution.participant_label,
            output_dir=config.execution.output_dir,
            run_uuid=config.execution.run_uuid,
            session_list=session_list,
        )

    output_list_file = os.path.join(get_test_data_path(), f'{test_name}_outputs.txt')
    check_generated_files(config.execution.output_dir, output_list_file)
