"""Nilearn interfaces."""

import os

import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.interfaces.nilearn import NilearnBaseInterface


class _MeanImageInputSpec(BaseInterfaceInputSpec):
    bold_file = File(
        exists=True,
        mandatory=True,
        desc='A 4D BOLD file to process.',
    )
    mask_file = File(
        exists=True,
        mandatory=False,
        desc='A binary brain mask.',
    )
    out_file = File(
        'mean.nii.gz',
        usedefault=True,
        exists=False,
        desc='The name of the mean file to write out. mean.nii.gz by default.',
    )


class _MeanImageOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='Mean output file.',
    )


class MeanImage(NilearnBaseInterface, SimpleInterface):
    """MeanImage images."""

    input_spec = _MeanImageInputSpec
    output_spec = _MeanImageOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        from nilearn.masking import apply_mask, unmask
        from nipype.interfaces.base import isdefined

        if isdefined(self.inputs.mask_file):
            data = apply_mask(self.inputs.bold_file, self.inputs.mask_file)
            mean_data = data.mean(axis=0)
            mean_img = unmask(mean_data, self.inputs.mask_file)
        else:
            in_img = nb.load(self.inputs.bold_file)
            mean_data = in_img.get_fdata().mean(axis=3)
            mean_img = nb.Nifti1Image(mean_data, in_img.affine, in_img.header)

        self._results['out_file'] = os.path.join(runtime.cwd, self.inputs.out_file)
        mean_img.to_filename(self._results['out_file'])

        return runtime


class _MedianValueInputSpec(BaseInterfaceInputSpec):
    bold_file = File(
        exists=True,
        mandatory=True,
        desc='A 4D BOLD file to process.',
    )
    mask_file = File(
        exists=True,
        mandatory=True,
        desc='A binary brain mask.',
    )


class _MedianValueOutputSpec(TraitedSpec):
    median_value = traits.Float()


class MedianValue(NilearnBaseInterface, SimpleInterface):
    """MedianImage images."""

    input_spec = _MedianValueInputSpec
    output_spec = _MedianValueOutputSpec

    def _run_interface(self, runtime):
        from nilearn.masking import apply_mask

        data = apply_mask(self.inputs.bold_file, self.inputs.mask_file)
        self._results['median_value'] = np.median(data)

        return runtime
