# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
"""ReportCapableInterfaces for segmentation tools."""

import os
import re
import time
from collections import Counter

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    File,
    InputMultiObject,
    SimpleInterface,
    Str,
    TraitedSpec,
    isdefined,
    traits,
)

SUBJECT_TEMPLATE = """\
\t<ul class="elem-desc">
\t\t<li>Subject ID: {subject_id}</li>
\t\t<li>Functional series: {n_bold:d}</li>
{tasks}
\t\t<li>Standard output spaces: {std_spaces}</li>
\t\t<li>Non-standard output spaces: {nstd_spaces}</li>
\t</ul>
"""

FUNCTIONAL_TEMPLATE = """\
\t\t<details open>
\t\t<summary>Summary</summary>
\t\t<ul class="elem-desc">
\t\t\t<li>Original orientation: {ornt}</li>
\t\t\t<li>Repetition time (TR): {tr:.03g}s</li>
\t\t\t<li>Phase-encoding (PE) direction: {pedir}</li>
\t\t\t<li>Slice timing correction: {stc}</li>
\t\t\t<li>Susceptibility distortion correction: {sdc}</li>
\t\t\t<li>Registration: {registration}</li>
\t\t\t<li>Non-steady-state volumes: {dummy_scan_desc}</li>
\t\t</ul>
\t\t</details>
"""

ABOUT_TEMPLATE = """\t<ul>
\t\t<li>fMRIPost-AROMA version: {version}</li>
\t\t<li>fMRIPost-AROMA command: <code>{command}</code></li>
\t\t<li>Date preprocessed: {date}</li>
\t</ul>
</div>
"""


class SummaryOutputSpec(TraitedSpec):
    out_report = File(exists=True, desc='HTML segment containing summary')


class SummaryInterface(SimpleInterface):
    output_spec = SummaryOutputSpec

    def _run_interface(self, runtime):
        segment = self._generate_segment()
        fname = os.path.join(runtime.cwd, 'report.html')
        with open(fname, 'w') as fobj:
            fobj.write(segment)
        self._results['out_report'] = fname
        return runtime

    def _generate_segment(self):
        raise NotImplementedError


class SubjectSummaryInputSpec(BaseInterfaceInputSpec):
    subject_id = Str(desc='Subject ID')
    bold = InputMultiObject(
        traits.Either(File(exists=True), traits.List(File(exists=True))),
        desc='BOLD functional series',
    )
    std_spaces = traits.List(Str, desc='list of standard spaces')
    nstd_spaces = traits.List(Str, desc='list of non-standard spaces')


class SubjectSummary(SummaryInterface):
    input_spec = SubjectSummaryInputSpec
    output_spec = SummaryOutputSpec

    def _generate_segment(self):
        BIDS_NAME = re.compile(
            r'^(.*\/)?'
            '(?P<subject_id>sub-[a-zA-Z0-9]+)'
            '(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
            '(_(?P<task_id>task-[a-zA-Z0-9]+))?'
            '(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
            '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?'
            '(_(?P<run_id>run-[a-zA-Z0-9]+))?'
        )

        # Add list of tasks with number of runs
        bold_series = self.inputs.bold if isdefined(self.inputs.bold) else []
        bold_series = [s[0] if isinstance(s, list) else s for s in bold_series]

        counts = Counter(
            BIDS_NAME.search(series).groupdict()['task_id'][5:] for series in bold_series
        )

        tasks = ''
        if counts:
            header = '\t\t<ul class="elem-desc">'
            footer = '\t\t</ul>'
            lines = [
                '\t\t\t<li>Task: {task_id} ({n_runs:d} run{s})</li>'.format(
                    task_id=task_id, n_runs=n_runs, s='' if n_runs == 1 else 's'
                )
                for task_id, n_runs in sorted(counts.items())
            ]
            tasks = '\n'.join([header] + lines + [footer])

        return SUBJECT_TEMPLATE.format(
            subject_id=self.inputs.subject_id,
            n_bold=len(bold_series),
            tasks=tasks,
            std_spaces=', '.join(self.inputs.std_spaces),
            nstd_spaces=', '.join(self.inputs.nstd_spaces),
        )


class AboutSummaryInputSpec(BaseInterfaceInputSpec):
    version = Str(desc='FMRIPREP version')
    command = Str(desc='FMRIPREP command')
    # Date not included - update timestamp only if version or command changes


class AboutSummary(SummaryInterface):
    input_spec = AboutSummaryInputSpec

    def _generate_segment(self):
        return ABOUT_TEMPLATE.format(
            version=self.inputs.version,
            command=self.inputs.command,
            date=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        )


class _ICAAROMAInputSpecRPT(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='BOLD series input to ICA-AROMA',
    )
    melodic_dir = Directory(
        exists=True,
        mandatory=True,
        desc='MELODIC directory containing the ICA outputs',
    )
    aroma_noise_ics = File(
        exists=True,
        desc='Noise components estimated by ICA-AROMA, in a comma-separated values file.',
    )
    out_report = File(
        'ica_aroma_reportlet.svg',
        usedefault=True,
        desc='Filename for the visual report generated by Nipype.',
    )
    report_mask = File(
        exists=True,
        mandatory=True,
        desc=(
            'Mask used to draw the outline on the reportlet. '
            'If not set the mask will be derived from the data.'
        ),
    )
    compress_report = traits.Bool(
        True,
        usedefault=True,
        desc='Whether to compress the reportlet with SVGO.',
    )


class _ICAAROMAOutputSpecRPT(TraitedSpec):
    out_report = File(
        exists=True,
        desc='Filename for the visual report generated by Nipype.',
    )


class ICAAROMARPT(SimpleInterface):
    """Create a reportlet for ICA-AROMA outputs."""

    input_spec = _ICAAROMAInputSpecRPT
    output_spec = _ICAAROMAOutputSpecRPT

    def _run_interface(self, runtime):
        from niworkflows.viz.utils import plot_melodic_components

        out_file = os.path.abspath(self.inputs.out_report)

        plot_melodic_components(
            melodic_dir=self.inputs.melodic_dir,
            in_file=self.inputs.in_file,
            out_file=out_file,
            compress=self.inputs.compress_report,
            report_mask=self.inputs.report_mask,
            noise_components_file=self.inputs.aroma_noise_ics,
        )
        self._results['out_report'] = out_file
        return runtime


class _ICAAROMAMetricsInputSpecRPT(BaseInterfaceInputSpec):
    aroma_features = File(
        exists=True,
        mandatory=True,
        desc='ICA-AROMA metrics',
    )
    out_report = File(
        'metrics_reportlet.svg',
        usedefault=True,
        desc='Filename for the visual report generated by Nipype.',
    )


class _ICAAROMAMetricsOutputSpecRPT(TraitedSpec):
    out_report = File(
        exists=True,
        desc='Filename for the visual report generated by Nipype.',
    )


class ICAAROMAMetricsRPT(SimpleInterface):
    """Create a reportlet for ICA-AROMA outputs."""

    input_spec = _ICAAROMAMetricsInputSpecRPT
    output_spec = _ICAAROMAMetricsOutputSpecRPT

    def _run_interface(self, runtime):
        import pandas as pd
        import seaborn as sns

        out_file = os.path.abspath(self.inputs.out_report)

        df = pd.read_table(self.inputs.aroma_features)

        sns.set_theme(style='ticks')

        g = sns.pairplot(
            df,
            hue='classification',
            vars=['edge_fract', 'csf_fract', 'max_RP_corr', 'HFC'],
            palette={'rejected': 'red', 'accepted': 'blue'},
            corner=True,
        )
        g.savefig(out_file)
        self._results['out_report'] = out_file
        return runtime
