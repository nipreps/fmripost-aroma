"""Functions to calculate ICA-AROMA features for component classification."""

import logging

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn import image, masking

from fmripost_aroma import data as load_data
from fmripost_aroma.utils import utils

LGR = logging.getLogger(__name__)


def feature_time_series(mixing: np.ndarray, motpars: np.ndarray):
    """Extract maximum motion parameter correlation scores from components.

    This function determines the maximum robust correlation of each component
    time series with a model of 72 realignment parameters.

    Parameters
    ----------
    mixing : numpy.ndarray of shape (T, C)
        Mixing matrix in shape T (time) by C (component).
    motpars : array_like
        Motion parameters are (time x 6), with the first three columns being
        rotation parameters (in radians) and the final three being translation
        parameters (in mm).

    Returns
    -------
    max_RP_corr : array_like
        Array of the maximum RP correlation feature scores for the components
        of the melodic_mix file.
    metric_metadata : None or dict
        If the ``metric_metadata`` input was None, then None will be returned.
        Otherwise, this will be a dictionary containing existing information,
        as well as new metadata for the ``max_RP_corr`` metric.
    """
    metric_metadata = {
        "max_RP_corr": {
            "LongName": "Maximum motion parameter correlation",
            "Description": (
                "The maximum correlation coefficient between each component and "
                "a set of 36 regressors derived from the motion parameters. "
                "The derived regressors are the raw six motion parameters (6), "
                "their derivatives (6), "
                "the parameters and their derivatives time-shifted one TR forward (12), and "
                "the parameters and their derivatives time-shifted one TR backward (12). "
                "The correlations are performed on a series of 1000 permutations, "
                "in which 90 percent of the volumes are selected from both the "
                "component time series and the motion parameters. "
                "The correlation is performed between each permuted component time series and "
                "each permuted regressor in the motion parameter model, "
                "as well as the squared versions of both. "
                "The maximum correlation coefficient from each permutation is retained and these "
                "correlation coefficients are averaged across permutations for the final metric."
            ),
            "Units": "arbitrary",
        },
    }

    rp6 = motpars.copy()
    if (rp6.ndim != 2) or (rp6.shape[1] != 6):
        raise ValueError(f"Motion parameters must of shape (n_trs, 6), not {rp6.shape}")

    if rp6.shape[0] != mixing.shape[0]:
        raise ValueError(
            f"Number of rows in mixing matrix ({mixing.shape[0]}) does not match "
            f"number of rows in motion parameters ({rp6.shape[0]})."
        )

    # Determine the derivatives of the RPs (add zeros at time-point zero)
    _, n_motpars = rp6.shape
    rp6_der = np.vstack((np.zeros(n_motpars), np.diff(rp6, axis=0)))

    # Create an RP-model including the RPs and its derivatives
    rp12 = np.hstack((rp6, rp6_der))

    # add the fw and bw shifted versions
    rp12_1fw = np.vstack((np.zeros(2 * n_motpars), rp12[:-1]))
    rp12_1bw = np.vstack((rp12[1:], np.zeros(2 * n_motpars)))
    rp_model = np.hstack((rp12, rp12_1fw, rp12_1bw))

    # Determine the maximum correlation between RPs and IC time-series
    n_splits = 1000
    n_volumes, n_components = mixing.shape
    n_rows_to_choose = int(round(0.9 * n_volumes))

    # Max correlations for multiple splits of the dataset (for a robust estimate)
    max_correlations = np.empty((n_splits, n_components))
    for i_split in range(n_splits):
        # Select a random subset of 90% of the dataset rows (*without* replacement)
        chosen_rows = np.random.choice(a=range(n_volumes), size=n_rows_to_choose, replace=False)

        # Combined correlations between RP and IC time-series, squared and non squared
        correl_nonsquared = utils.cross_correlation(mixing[chosen_rows], rp_model[chosen_rows])
        correl_squared = utils.cross_correlation(
            mixing[chosen_rows] ** 2, rp_model[chosen_rows] ** 2
        )
        correl_both = np.hstack((correl_squared, correl_nonsquared))

        # Maximum absolute temporal correlation for every IC
        max_correlations[i_split] = np.abs(correl_both).max(axis=1)

    # Feature score is the mean of the maximum correlation over all the random splits.
    # Avoid propagating occasional nans that arise in artificial test cases
    max_RP_corr = np.nanmean(max_correlations, axis=0)
    metric_df = pd.DataFrame(data=max_RP_corr, columns=["max_RP_corr"])
    return metric_df, metric_metadata


def feature_frequency(mixing_fft: np.ndarray, TR: float, f_hp: float = 0.01):
    """Extract the high-frequency content feature scores.

    This function determines the frequency, as fraction of the Nyquist
    frequency, at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    Parameters
    ----------
    mixing_fft : numpy.ndarray of shape (F, C)
        Stored array is (frequency x component), with frequencies
        ranging from 0 Hz to Nyquist frequency.
    TR : float
        TR (in seconds) of the fMRI data
    f_hp: float, optional
        High-pass cutoff frequency in spectrum computations.

    Returns
    -------
    HFC : array_like
        Array of the HFC ('High-frequency content') feature scores
        for the components of the melodic_FTmix file
    metric_metadata : None or dict
        If the ``metric_metadata`` input was None, then None will be returned.
        Otherwise, this will be a dictionary containing existing information,
        as well as new metadata for the ``HFC`` metric.
    """
    metric_metadata = {
        "HFC": {
            "LongName": "High-frequency content",
            "Description": (
                "The proportion of the power spectrum for each component that falls above "
                f"{f_hp} Hz."
            ),
            "Units": "arbitrary",
        },
    }

    # Determine sample frequency
    Fs = 1 / TR

    # Determine Nyquist-frequency
    Ny = Fs / 2

    n_frequencies = mixing_fft.shape[0]

    # Determine which frequencies are associated with every row in the
    # melodic_FTmix file (assuming the rows range from 0Hz to Nyquist)
    frequencies = Ny * np.arange(1, n_frequencies + 1) / n_frequencies

    # Only include frequencies higher than f_hp Hz
    included_freqs_idx = np.squeeze(np.array(np.where(frequencies > f_hp)))
    mixing_fft = mixing_fft[included_freqs_idx, :]
    frequencies = frequencies[included_freqs_idx]

    # Set frequency range to [0-1]
    frequencies_normalized = (frequencies - f_hp) / (Ny - f_hp)

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fract = np.cumsum(mixing_fft, axis=0) / np.sum(mixing_fft, axis=0)

    # Determine the index of the frequency with the fractional cumulative sum closest to 0.5
    cutoff_idx = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)

    # Now get the fractions associated with those indices index, these are the final feature scores
    high_frequency_content = frequencies_normalized[cutoff_idx]
    metric_df = pd.DataFrame(data=high_frequency_content, columns=["HFC"])

    return metric_df, metric_metadata


def feature_spatial(component_maps):
    """Extract the spatial feature scores.

    For each IC it determines the fraction of the mixture modeled thresholded
    Z-maps respectively located within the CSF or at the brain edges,
    using predefined standardized masks.

    Parameters
    ----------
    component_maps : str or niimg_like
        Full path of the nii.gz file containing mixture-modeled thresholded
        (p<0.5) Z-maps, registered to the MNI152 2mm template

    Returns
    -------
    edge_fract : array_like
        Array of the edge fraction feature scores for the components of the
        component_maps file
    csf_fract : array_like
        Array of the CSF fraction feature scores for the components of the
        component_maps file
    metric_metadata : None or dict
        If the ``metric_metadata`` input was None, then None will be returned.
        Otherwise, this will be a dictionary containing existing information,
        as well as new metadata for the ``edge_fract`` and ``csf_fract``
        metrics.
    """
    metric_metadata = {
        "edge_fract": {
            "LongName": "Edge content fraction",
            "Description": (
                "The fraction of thresholded component z-values at the edge of the brain. "
                "This is calculated by "
                "(1) taking the absolute value of the thresholded Z map for each component, "
                "(2) summing z-statistics from the whole brain, "
                "(3) summing z-statistics from outside of the brain, "
                "(4) summing z-statistics from voxels in CSF compartments, "
                "(5) summing z-statistics from voxels at the edge of the brain, "
                "(6) adding the sums from outside of the brain and the edge of the brain, "
                "(7) subtracting the CSF sum from the total brain sum, and "
                "(8) dividing the out-of-brain+edge-of-brain sum by the whole brain (minus CSF) "
                "sum."
            ),
            "Units": "arbitrary",
        },
        "csf_fract": {
            "LongName": "CSF content fraction",
            "Description": (
                "The fraction of thresholded component z-values in the brain's cerebrospinal "
                "fluid. "
                "This is calculated by "
                "(1) taking the absolute value of the thresholded Z map for each component, "
                "(2) summing z-statistics from the whole brain, "
                "(3) summing z-statistics from voxels in CSF compartments, and "
                "(4) dividing the CSF z-statistic sum by the whole brain z-statistic sum."
            ),
            "Units": "arbitrary",
        },
    }

    # Get the number of ICs
    components_img = nb.load(component_maps)
    n_components = components_img.shape[3]

    csf_mask = load_data("mask_csf.nii.gz")
    edge_mask = load_data("mask_edge.nii.gz")
    out_mask = load_data("mask_out.nii.gz")

    # Loop over ICs
    metric_df = pd.DataFrame(columns=["edge_fract", "csf_fract"], data=np.zeros((n_components, 2)))
    for i_comp in range(n_components):
        # Extract IC from the merged melodic_IC_thr2MNI2mm file
        component_img = image.index_img(component_maps, i_comp)

        # Change to absolute Z-values
        component_img = image.math_img("np.abs(img)", img=component_img)

        # Get sum of Z-values within the total Z-map
        component_data = component_img.get_fdata()
        tot_sum = np.sum(component_data)

        if tot_sum == 0:
            LGR.info(f"\t- The spatial map of component {i_comp + 1} is empty. Please check!")

        # Get sum of Z-values of the voxels located within the CSF
        csf_data = masking.apply_mask(component_img, csf_mask)
        csf_sum = np.sum(csf_data)

        # Get sum of Z-values of the voxels located within the Edge
        edge_data = masking.apply_mask(component_img, edge_mask)
        edge_sum = np.sum(edge_data)

        # Get sum of Z-values of the voxels located outside the brain
        out_data = masking.apply_mask(component_img, out_mask)
        out_sum = np.sum(out_data)

        # Determine edge and CSF fraction
        if tot_sum != 0:
            metric_df.loc[i_comp, "edge_fract"] = (out_sum + edge_sum) / (tot_sum - csf_sum)
            metric_df.loc[i_comp, "csf_fract"] = csf_sum / tot_sum
        else:
            metric_df.loc[i_comp, "edge_fract"] = 0
            metric_df.loc[i_comp, "csf_fract"] = 0

    return metric_df, metric_metadata
