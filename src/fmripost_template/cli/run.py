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
"""fMRI post-processing template workflow."""

from fmripost_template import config

EXITCODE: int = -1


def main():
    """Entry point."""

    import gc
    import sys
    from multiprocessing import Manager, Process
    from os import EX_SOFTWARE
    from pathlib import Path

    from fmripost_template.cli.parser import parse_args
    from fmripost_template.cli.workflow import build_workflow
    from fmripost_template.utils.bids import write_bidsignore, write_derivative_description

    parse_args()

    # Code Carbon
    if config.execution.track_carbon:
        from codecarbon import OfflineEmissionsTracker

        country_iso_code = config.execution.country_code
        config.loggers.workflow.log(25, 'CodeCarbon tracker started ...')
        config.loggers.workflow.log(25, f'Using country_iso_code: {country_iso_code}')
        config.loggers.workflow.log(25, f'Saving logs at: {config.execution.log_dir}')

        tracker = OfflineEmissionsTracker(
            output_dir=config.execution.log_dir, country_iso_code=country_iso_code
        )
        tracker.start()

    if 'pdb' in config.execution.debug:
        from fmripost_template.utils.debug import setup_exceptionhook

        setup_exceptionhook()
        config.nipype.plugin = 'Linear'

    sentry_sdk = None
    if not config.execution.notrack and not config.execution.debug:
        import atexit

        import sentry_sdk

        from fmripost_template.utils.telemetry import sentry_setup, setup_migas

        sentry_setup()
        setup_migas(init_ping=True)
        atexit.register(migas_exit)

    # CRITICAL Save the config to a file. This is necessary because the execution graph
    # is built as a separate process to keep the memory footprint low. The most
    # straightforward way to communicate with the child process is via the filesystem.
    config_file = config.execution.work_dir / config.execution.run_uuid / 'config.toml'
    config_file.parent.mkdir(exist_ok=True, parents=True)
    config.to_filename(config_file)

    # CRITICAL Call build_workflow(config_file, retval) in a subprocess.
    # Because Python on Linux does not ever free virtual memory (VM), running the
    # workflow construction jailed within a process preempts excessive VM buildup.
    if 'pdb' not in config.execution.debug:
        with Manager() as mgr:
            retval = mgr.dict()
            p = Process(target=build_workflow, args=(str(config_file), retval))
            p.start()
            p.join()
            retval = dict(retval.items())  # Convert to base dictionary

            if p.exitcode:
                retval['return_code'] = p.exitcode

    else:
        retval = build_workflow(str(config_file), {})

    global EXITCODE
    EXITCODE = retval.get('return_code', 0)
    fmripost_template_wf = retval.get('workflow', None)

    # CRITICAL Load the config from the file. This is necessary because the ``build_workflow``
    # function executed constrained in a process may change the config (and thus the global
    # state of fMRIPost-template).
    config.load(config_file)

    if config.execution.reports_only:
        sys.exit(int(EXITCODE > 0))

    if fmripost_template_wf and config.execution.write_graph:
        fmripost_template_wf.write_graph(graph2use='colored', format='svg', simple_form=True)

    EXITCODE = EXITCODE or (fmripost_template_wf is None) * EX_SOFTWARE
    if EXITCODE != 0:
        sys.exit(EXITCODE)

    # Generate boilerplate
    with Manager() as mgr:
        from fmripost_template.cli.workflow import build_boilerplate

        p = Process(target=build_boilerplate, args=(str(config_file), fmripost_template_wf))
        p.start()
        p.join()

    if config.execution.boilerplate_only:
        sys.exit(int(EXITCODE > 0))

    # Clean up master process before running workflow, which may create forks
    gc.collect()

    # Sentry tracking
    if sentry_sdk is not None:
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag('run_uuid', config.execution.run_uuid)
            scope.set_tag('npart', len(config.execution.participant_label))
        sentry_sdk.add_breadcrumb(message='fMRIPost-template started', level='info')
        sentry_sdk.capture_message('fMRIPost-template started', level='info')

    config.loggers.workflow.log(
        15,
        '\n'.join(
            ['fMRIPost-template config:'] + [f'\t\t{s}' for s in config.dumps().splitlines()]
        ),
    )
    config.loggers.workflow.log(25, 'fMRIPost-template started!')
    errno = 1  # Default is error exit unless otherwise set
    try:
        fmripost_template_wf.run(**config.nipype.get_plugin())
    except Exception as e:
        if not config.execution.notrack:
            from fmripost_template.utils.telemetry import process_crashfile

            crashfolders = [
                config.execution.output_dir / f'sub-{s}' / 'log' / config.execution.run_uuid
                for s in config.execution.participant_label
            ]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob('crash*.*'):
                    process_crashfile(crashfile)

            if sentry_sdk is not None and 'Workflow did not execute cleanly' not in str(e):
                sentry_sdk.capture_exception(e)

        config.loggers.workflow.critical('fMRIPost-template failed: %s', e)
        raise

    else:
        config.loggers.workflow.log(25, 'fMRIPost-template finished successfully!')
        if sentry_sdk is not None:
            success_message = 'fMRIPost-template finished without errors'
            sentry_sdk.add_breadcrumb(message=success_message, level='info')
            sentry_sdk.capture_message(success_message, level='info')

        # Bother users with the boilerplate only iff the workflow went okay.
        boiler_file = config.execution.output_dir / 'logs' / 'CITATION.md'
        if boiler_file.exists():
            if config.environment.exec_env in (
                'singularity',
                'docker',
                'fmripost_template-docker',
            ):
                boiler_file = Path('<OUTPUT_PATH>') / boiler_file.relative_to(
                    config.execution.output_dir
                )

            config.loggers.workflow.log(
                25,
                'Works derived from this fMRIPost-template execution should include the '
                f'boilerplate text found in {boiler_file}.',
            )

        if config.workflow.run_reconall:
            from niworkflows.utils.misc import _copy_any
            from templateflow import api

            dseg_tsv = str(api.get('fsaverage', suffix='dseg', extension=['.tsv']))
            _copy_any(dseg_tsv, str(config.execution.output_dir / 'desc-aseg_dseg.tsv'))
            _copy_any(dseg_tsv, str(config.execution.output_dir / 'desc-aparcaseg_dseg.tsv'))
        errno = 0
    finally:
        # Code Carbon
        if config.execution.track_carbon:
            emissions: float = tracker.stop()
            config.loggers.workflow.log(25, 'CodeCarbon tracker has stopped.')
            config.loggers.workflow.log(25, f'Saving logs at: {config.execution.log_dir}')
            config.loggers.workflow.log(25, f'Carbon emissions: {emissions} kg')

        from fmripost_template.reports.core import generate_reports

        # Generate reports phase
        failed_reports = generate_reports(
            subject_list=config.execution.participant_label,
            output_dir=config.execution.output_dir,
            run_uuid=config.execution.run_uuid,
        )
        write_derivative_description(
            input_dir=config.execution.bids_dir,
            output_dir=config.execution.output_dir,
            dataset_links=config.execution.dataset_links,
        )
        write_bidsignore(config.execution.output_dir)

        if sentry_sdk is not None and failed_reports:
            sentry_sdk.capture_message(
                f'Report generation failed for {failed_reports} subjects',
                level='error',
            )
        sys.exit(int((errno + len(failed_reports)) > 0))


def migas_exit() -> None:
    """Exit migas.

    Send a final crumb to the migas server signaling if the run successfully completed
    This function should be registered with `atexit` to run at termination.
    """
    import sys

    from fmripost_template.utils.telemetry import send_breadcrumb

    global EXITCODE
    migas_kwargs = {'status': 'C'}
    # `sys` will not have these attributes unless an error has been handled
    if hasattr(sys, 'last_type'):
        migas_kwargs = {
            'status': 'F',
            'status_desc': 'Finished with error(s)',
            'error_type': sys.last_type,
            'error_desc': sys.last_value,
        }
    elif EXITCODE != 0:
        migas_kwargs.update(
            {
                'status': 'F',
                'status_desc': f'Completed with exitcode {EXITCODE}',
            }
        )
    else:
        migas_kwargs['status_desc'] = 'Success'

    send_breadcrumb(**migas_kwargs)


if __name__ == '__main__':
    raise RuntimeError(
        'fmripost_template/cli/run.py should not be run directly;\n'
        'Please `pip install` fmripost_template and use the `fmripost_template` command'
    )
