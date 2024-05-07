"""Workflows to resample data."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe


def init_resample_raw_wf(bold_file, functional_cache):
    """Resample raw BOLD data to MNI152NLin6Asym:res-2mm space."""
    from fmriprep.workflows.bold.stc import init_bold_stc_wf
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_aroma.interfaces.resampler import Resampler

    workflow = Workflow(name='resample_raw_wf')

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'mask_file']),
        name='inputnode',
    )
    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.mask_file = functional_cache['bold_mask']

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_std', 'bold_mask_std']),
        name='outputnode',
    )

    stc_wf = init_bold_stc_wf(name='resample_stc_wf')
    workflow.connect([
        (inputnode, stc_wf, [
            ('bold_file', 'inputnode.bold_file'),
            ('mask_file', 'inputnode.mask_file'),
        ]),
    ])  # fmt:skip

    resample_bold = pe.Node(
        Resampler(space='MNI152NLin6Asym', resolution='2'),
        name='resample_bold',
    )
    workflow.connect([
        (stc_wf, resample_bold, [('outputnode.bold_file', 'bold_file')]),
        (resample_bold, outputnode, [('output_file', 'bold_std')]),
    ])  # fmt:skip

    resample_bold_mask = pe.Node(
        Resampler(space='MNI152NLin6Asym', resolution='2'),
        name='resample_bold_mask',
    )
    workflow.connect([
        (inputnode, resample_bold_mask, [('mask_file', 'bold_file')]),
        (resample_bold_mask, outputnode, [('output_file', 'bold_mask_std')]),
    ])  # fmt:skip

    return workflow
