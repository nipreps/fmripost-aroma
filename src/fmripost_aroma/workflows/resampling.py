"""Workflows to resample data."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe


def init_resample_volumetric_wf(
    bold_file,
    metadata,
    functional_cache,
    run_stc,
    name='resample_volumetric_wf',
):
    """Resample raw BOLD data to requested volumetric space.

    Parameters
    ----------
    bold_file : str
        Path to raw BOLD file.
    functional_cache : dict
        Dictionary with paths to functional data.
    run_stc : bool
        Whether to run STC.
    name : str
        Workflow name.
    """
    from fmriprep.workflows.bold.stc import init_bold_stc_wf
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_file',
                'mask_file',
            ],
        ),
        name='inputnode',
    )
    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.mask_file = functional_cache['bold_mask_native']

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_std', 'bold_mask_std']),
        name='outputnode',
    )

    stc_buffer = pe.Node(
        niu.IdentityInterface(fields=['bold_file']),
        name='stc_buffer',
    )
    if run_stc:
        stc_wf = init_bold_stc_wf(
            mem_gb={'filesize': 1},
            metadata=metadata,
            name='resample_stc_wf',
        )
        workflow.connect([
            (inputnode, stc_wf, [
                ('bold_file', 'inputnode.bold_file'),
                ('mask_file', 'inputnode.mask_file'),
            ]),
            (stc_wf, stc_buffer, [('outputnode.bold_file', 'bold_file')]),
        ])  # fmt:skip
    else:
        workflow.connect([(inputnode, stc_buffer, [('bold_file', 'bold_file')])])

    boldref2target = pe.Node(niu.Merge(2), name='boldref2target', run_without_submitting=True)
    bold2target = pe.Node(niu.Merge(2), name='bold2target', run_without_submitting=True)
    resample = pe.Node(
        ResampleSeries(jacobian=jacobian),
        name='resample',
        n_procs=omp_nthreads,
        mem_gb=mem_gb['resampled'],
    )

    workflow.connect([
        (inputnode, gen_ref, [
            ('bold_ref_file', 'moving_image'),
            ('target_ref_file', 'fixed_image'),
            ('target_mask', 'fov_mask'),
            (('resolution', _is_native), 'keep_native'),
        ]),
        (inputnode, boldref2target, [
            ('boldref2anat_xfm', 'in1'),
            ('anat2std_xfm', 'in2'),
        ]),
        (inputnode, bold2target, [('motion_xfm', 'in1')]),
        (inputnode, resample, [('bold_file', 'in_file')]),
        (gen_ref, resample, [('out_file', 'ref_file')]),
        (boldref2target, bold2target, [('out', 'in2')]),
        (bold2target, resample, [('out', 'transforms')]),
        (gen_ref, outputnode, [('out_file', 'resampling_reference')]),
        (resample, outputnode, [('out_file', 'bold_file')]),
    ])  # fmt:skip

    xfms = [
        functional_cache['orig_to_boldref_xfm'],
        functional_cache['boldref_to_anat_xfm'],
        functional_cache['anat_to_mni6_xfm'],
    ]

    resample_bold = pe.Node(
        Resampler(),
        name='resample_bold',
    )
    workflow.connect([
        (inputnode, resample_bold, [
            ('space', 'space'),
            ('res', 'res'),
        ]),
        (stc_buffer, resample_bold, [('outputnode.bold_file', 'bold_file')]),
        (resample_bold, outputnode, [('output_file', 'bold_std')]),
    ])  # fmt:skip

    resample_bold_mask = pe.Node(
        Resampler(),
        name='resample_bold_mask',
    )
    workflow.connect([
        (inputnode, resample_bold_mask, [
            ('space', 'space'),
            ('res', 'res'),
        ]),
        (inputnode, resample_bold_mask, [('mask_file', 'bold_file')]),
        (resample_bold_mask, outputnode, [('output_file', 'bold_mask_std')]),
    ])  # fmt:skip

    return workflow
