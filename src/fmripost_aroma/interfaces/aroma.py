"""Handling confounds."""

import os

import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from fmripost_aroma import config
from fmripost_aroma.utils import features, utils


class _AROMAClassifierInputSpec(BaseInterfaceInputSpec):
    motpars = File(exists=True, desc='motion parameters or general confounds file')
    mixing = File(exists=True, desc='mixing matrix')
    component_maps = File(exists=True, desc='thresholded z-statistic component maps')
    component_stats = File(exists=True, desc='melodic component statistics file')
    TR = traits.Float(desc='repetition time in seconds')
    skip_vols = traits.Int(desc='number of volumes to skip at the beginning of the timeseries')


class _AROMAClassifierOutputSpec(TraitedSpec):
    aroma_features = File(exists=True, desc='output confounds file extracted from ICA-AROMA')
    aroma_metadata = traits.Dict(desc='metadata for the ICA-AROMA confounds')
    aroma_noise_ics = File(exists=True, desc='output noise components from ICA-AROMA')


class AROMAClassifier(SimpleInterface):
    """Calculate ICA-AROMA features and classify components as signal or noise."""

    input_spec = _AROMAClassifierInputSpec
    output_spec = _AROMAClassifierOutputSpec

    def _run_interface(self, runtime):
        TR = self.inputs.TR
        motion_params = utils.load_motpars(self.inputs.motpars, source='fmriprep')
        motion_params = motion_params[self.inputs.skip_vols:, :]
        mixing = np.loadtxt(self.inputs.mixing)  # T x C
        component_maps = nb.load(self.inputs.component_maps)  # X x Y x Z x C
        if mixing.shape[1] != component_maps.shape[3]:
            raise ValueError(
                f'Number of columns in mixing matrix ({mixing.shape[1]}) does not match '
                f'fourth dimension of component maps file ({component_maps.shape[3]}).'
            )

        if mixing.shape[0] != motion_params.shape[0]:
            raise ValueError(
                f'Number of rows in mixing matrix ({mixing.shape[0]}) does not match '
                f'number of rows in motion parameters ({motion_params.shape[0]}).'
            )

        config.loggers.interface.info('  - extracting the CSF & Edge fraction features')
        metric_metadata = {}
        features_df = pd.DataFrame()
        spatial_df, spatial_metadata = features.feature_spatial(self.inputs.component_maps)
        features_df = pd.concat([features_df, spatial_df], axis=1)
        metric_metadata.update(spatial_metadata)

        config.loggers.interface.info('  - extracting the Maximum RP correlation feature')
        ts_df, ts_metadata = features.feature_time_series(mixing, motion_params)
        features_df = pd.concat([features_df, ts_df], axis=1)
        metric_metadata.update(ts_metadata)

        config.loggers.interface.info('  - extracting the High-frequency content feature')
        # Should probably check that the frequencies match up with MELODIC's outputs
        mixing_fft, _ = utils.get_spectrum(mixing, TR)
        freq_df, freq_metadata = features.feature_frequency(mixing_fft, TR, f_hp=0.01)
        features_df = pd.concat([features_df, freq_df], axis=1)
        metric_metadata.update(freq_metadata)

        config.loggers.interface.info('  - classification')
        clf_df, clf_metadata = utils.classification(features_df)
        features_df = pd.concat([features_df, clf_df], axis=1)
        metric_metadata.update(clf_metadata)

        # Add MELODIC component statistics to the AROMA features
        component_stats = pd.read_csv(self.inputs.component_stats, header=None, sep='  ')[[0, 1]] / 100
        component_stats.columns = ['model_variance_explained', 'total_variance_explained']
        features_df = pd.concat([features_df, component_stats], axis=1)
        metric_metadata.update(
            {
                'model_variance_explained': {
                    'Description': 'Model variance explained by the component',
                },
                'total_variance_explained': {
                    'Description': 'Total variance explained by the component',
                },
            },
        )

        features_file = os.path.abspath('aroma_features.tsv')
        features_df.to_csv(features_file, sep='\t', index=False)

        # Noise components
        noise_ics = clf_df[clf_df['classification'] == 'rejected'].index.values + 1
        noise_ics_file = os.path.abspath('aroma_noise_ics.csv')
        with open(noise_ics_file, 'w') as f:
            f.write(','.join(map(str, noise_ics)))

        self._results['aroma_features'] = features_file
        self._results['aroma_metadata'] = metric_metadata
        self._results['aroma_noise_ics'] = noise_ics_file
        return runtime
