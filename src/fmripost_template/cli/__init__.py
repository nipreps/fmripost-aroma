# SPDX-FileCopyrightText: 2023-present Chris Markiewicz <effigies@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import click

from .. import __version__


@click.group(context_settings={'help_option_names': ['-h', '--help']}, invoke_without_command=True)
@click.version_option(version=__version__, prog_name='fmripost-template')
@click.pass_context
def fmripost_template(ctx: click.Context):
    click.echo('Hello world!')
