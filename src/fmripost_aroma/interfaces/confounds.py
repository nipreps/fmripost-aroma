"""Handling confounds."""

import os
import re
import shutil

import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from fmripost_aroma import config


class _ICAConfoundsInputSpec(BaseInterfaceInputSpec):
    in_directory = Directory(
        mandatory=True,
        desc='directory where ICA derivatives are found',
    )
    skip_vols = traits.Int(desc='number of non steady state volumes identified')
    err_on_aroma_warn = traits.Bool(False, usedefault=True, desc='raise error if aroma fails')


class _ICAConfoundsOutputSpec(TraitedSpec):
    aroma_confounds = traits.Either(
        None, File(exists=True, desc='output confounds file extracted from ICA-AROMA')
    )
    aroma_noise_ics = File(exists=True, desc='ICA-AROMA noise components')
    melodic_mix = File(exists=True, desc='melodic mix file')
    aroma_metadata = File(exists=True, desc='tabulated ICA-AROMA metadata')


class ICAConfounds(SimpleInterface):
    """Extract confounds from ICA-AROMA result directory."""

    input_spec = _ICAConfoundsInputSpec
    output_spec = _ICAConfoundsOutputSpec

    def _run_interface(self, runtime):
        (aroma_confounds, motion_ics_out, melodic_mix_out, aroma_metadata) = _get_ica_confounds(
            self.inputs.in_directory, self.inputs.skip_vols, newpath=runtime.cwd
        )

        if self.inputs.err_on_aroma_warn and aroma_confounds is None:
            raise RuntimeError('ICA-AROMA failed')

        aroma_confounds = self._results['aroma_confounds'] = aroma_confounds

        self._results['aroma_noise_ics'] = motion_ics_out
        self._results['melodic_mix'] = melodic_mix_out
        self._results['aroma_metadata'] = aroma_metadata
        return runtime


def _get_ica_confounds(ica_out_dir, skip_vols, newpath=None):
    """Extract confounds from ICA-AROMA result directory."""
    if newpath is None:
        newpath = os.getcwd()

    # load the txt files from ICA-AROMA
    melodic_mix = os.path.join(ica_out_dir, 'melodic.ica/melodic_mix')
    motion_ics = os.path.join(ica_out_dir, 'classified_motion_ICs.txt')
    aroma_metadata = os.path.join(ica_out_dir, 'classification_overview.txt')
    aroma_icstats = os.path.join(ica_out_dir, 'melodic.ica/melodic_ICstats')

    # Change names of motion_ics and melodic_mix for output
    melodic_mix_out = os.path.join(newpath, 'MELODICmix.tsv')
    motion_ics_out = os.path.join(newpath, 'AROMAnoiseICs.csv')
    aroma_metadata_out = os.path.join(newpath, 'classification_overview.tsv')

    # copy metion_ics file to derivatives name
    shutil.copyfile(motion_ics, motion_ics_out)

    # -1 since python lists start at index 0
    motion_ic_indices = np.loadtxt(motion_ics, dtype=int, delimiter=',', ndmin=1) - 1
    melodic_mix_arr = np.loadtxt(melodic_mix, ndmin=2)

    # pad melodic_mix_arr with rows of zeros corresponding to number non steadystate volumes
    if skip_vols > 0:
        zeros = np.zeros([skip_vols, melodic_mix_arr.shape[1]])
        melodic_mix_arr = np.vstack([zeros, melodic_mix_arr])

    # save melodic_mix_arr
    np.savetxt(melodic_mix_out, melodic_mix_arr, delimiter='\t')

    # process the metadata so that the IC column entries match the BIDS name of
    # the regressor
    aroma_metadata = pd.read_csv(aroma_metadata, sep='\t')
    aroma_metadata['IC'] = [f'aroma_motion_{name}' for name in aroma_metadata['IC']]
    aroma_metadata.columns = [re.sub(r'[ |\-|\/]', '_', c) for c in aroma_metadata.columns]

    # Add variance statistics to metadata
    aroma_icstats = pd.read_csv(aroma_icstats, header=None, sep='  ')[[0, 1]] / 100
    aroma_icstats.columns = ['model_variance_explained', 'total_variance_explained']
    aroma_metadata = pd.concat([aroma_metadata, aroma_icstats], axis=1)

    aroma_metadata.to_csv(aroma_metadata_out, sep='\t', index=False)

    # Return dummy list of ones if no noise components were found
    if motion_ic_indices.size == 0:
        config.loggers.interfaces.warning('No noise components were classified')
        return None, motion_ics_out, melodic_mix_out, aroma_metadata_out

    # the "good" ics, (e.g., not motion related)
    good_ic_arr = np.delete(melodic_mix_arr, motion_ic_indices, 1).T

    # return dummy lists of zeros if no signal components were found
    if good_ic_arr.size == 0:
        config.loggers.interfaces.warning('No signal components were classified')
        return None, motion_ics_out, melodic_mix_out, aroma_metadata_out

    # transpose melodic_mix_arr so x refers to the correct dimension
    aggr_confounds = np.asarray([melodic_mix_arr.T[x] for x in motion_ic_indices])

    # add one to motion_ic_indices to match melodic report.
    aroma_confounds = os.path.join(newpath, 'AROMAAggrCompAROMAConfounds.tsv')
    pd.DataFrame(
        aggr_confounds.T,
        columns=[f'aroma_motion_{x + 1:02d}' for x in motion_ic_indices],
    ).to_csv(aroma_confounds, sep='\t', index=None)

    return aroma_confounds, motion_ics_out, melodic_mix_out, aroma_metadata_out


class _ICADenoiseInputSpec(BaseInterfaceInputSpec):
    method = traits.Enum("aggr", "nonaggr", "orthaggr", mandatory=True, desc="denoising method")
    bold_file = File(exists=True, mandatory=True, desc="input file to denoise")
    mask_file = File(exists=True, mandatory=True, desc="mask file")
    confounds = File(exists=True, mandatory=True, desc="confounds file")


class _ICADenoiseOutputSpec(TraitedSpec):
    denoised_file = File(exists=True, desc="denoised output file")


class ICADenoise(SimpleInterface):
    """Denoise data using ICA components."""

    input_spec = _ICADenoiseInputSpec
    output_spec = _ICADenoiseOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        import pandas as pd
        from nilearn.maskers import NiftiMasker
        from nilearn.masking import apply_mask, unmask

        method = self.inputs.method
        bold_file = self.inputs.bold_file
        confounds_file = self.inputs.confounds
        metrics_file = self.inputs.metrics_file

        confounds_df = pd.read_table(confounds_file)

        # Split up component time series into accepted and rejected components
        metrics_df = pd.read_table(metrics_file)
        rejected_columns = metrics_df.loc[metrics_df["classification"] == "rejected", "Component"]
        accepted_columns = metrics_df.loc[metrics_df["classification"] == "accepted", "Component"]
        rejected_components = confounds_df[rejected_columns].to_numpy()
        accepted_components = confounds_df[accepted_columns].to_numpy()

        if method == "aggr":
            # Denoise the data with the motion components
            masker = NiftiMasker(
                mask_img=self.inputs.mask_file,
                standardize_confounds=True,
                standardize=False,
                smoothing_fwhm=None,
                detrend=False,
                low_pass=None,
                high_pass=None,
                t_r=None,  # This shouldn't be necessary since we aren't bandpass filtering
                reports=False,
            )

            # Denoise the data by fitting and transforming the data file using the masker
            denoised_img_2d = masker.fit_transform(bold_file, confounds=rejected_components)

            # Transform denoised data back into 4D space
            denoised_img = masker.inverse_transform(denoised_img_2d)
        elif method == "orthaggr":
            # Regress the good components out of the bad time series to get "pure evil" regressors
            betas = np.linalg.lstsq(accepted_components, rejected_components, rcond=None)[0]
            pred_bad_timeseries = np.dot(accepted_components, betas)
            orth_bad_timeseries = rejected_components - pred_bad_timeseries

            # Once you have these "pure evil" components, you can denoise the data
            masker = NiftiMasker(
                mask_img=self.inputs.mask_file,
                standardize_confounds=True,
                standardize=False,
                smoothing_fwhm=None,
                detrend=False,
                low_pass=None,
                high_pass=None,
                t_r=None,  # This shouldn't be necessary since we aren't bandpass filtering
                reports=False,
            )

            # Denoise the data by fitting and transforming the data file using the masker
            denoised_img_2d = masker.fit_transform(bold_file, confounds=orth_bad_timeseries)

            # Transform denoised data back into 4D space
            denoised_img = masker.inverse_transform(denoised_img_2d)
        else:
            # Apply the mask to the data image to get a 2d array
            data = apply_mask(bold_file, self.inputs.mask_file)

            # Fit GLM to accepted components and rejected components
            # (after adding a constant term)
            regressors = np.hstack(
                (
                    rejected_components,
                    accepted_components,
                    np.ones((confounds_df.shape[0], 1)),
                ),
            )
            betas = np.linalg.lstsq(regressors, data, rcond=None)[0][:-1]

            # Denoise the data using the betas from just the bad components
            confounds_idx = np.arange(rejected_components.shape[1])
            pred_data = np.dot(rejected_components, betas[confounds_idx, :])
            data_denoised = data - pred_data

            # Save to file
            denoised_img = unmask(data_denoised, self.inputs.mask_file)

        self._results["denoised_file"] = os.path.abspath("denoised.nii.gz")
        denoised_img.to_filename(self._results["denoised_file"])

        return runtime
