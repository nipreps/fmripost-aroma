"""Interfaces for resampling."""

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _ResamplerInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True, desc="BOLD file to resample.")
    derivs_path = traits.Directory(
        exists=True,
        desc="Path to derivatives.",
    )
    output_dir = traits.Directory(
        exists=True,
        desc="Output directory.",
    )
    space = traits.Str(
        "MNI152NLin6Asym",
        usedefault=True,
        desc="Output space.",
    )
    resolution = traits.Str(
        "2",
        usedefault=True,
        desc="Output resolution.",
    )


class _ResamplerOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc="Resampled BOLD file.")


class Resampler(SimpleInterface):
    """Extract timeseries and compute connectivity matrices.

    Write out time series using Nilearn's NiftiLabelMasker
    Then write out functional correlation matrix of
    timeseries using numpy.
    """

    input_spec = _ResamplerInputSpec
    output_spec = _ResamplerOutputSpec

    def _run_interface(self, runtime):
        from fmripost_aroma.utils import resampler

        output_file = resampler.main(
            bold_file=self.inputs.bold_file,
            derivs_path=self.inputs.derivs_path,
            output_dir=self.inputs.output_dir,
            space=self.inputs.space,
            resolution=self.inputs.resolution,
        )

        self._results["output_file"] = output_file

        return runtime
