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
