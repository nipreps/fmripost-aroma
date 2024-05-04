"""Functions to calculate ICA-AROMA features for component classification."""

import logging
import os

import numpy as np
from nilearn import image, masking
from nilearn._utils import load_niimg

from aroma import utils

LGR = logging.getLogger(__name__)


def feature_time_series(mel_mix, mc, metric_metadata=None):
    """Extract maximum motion parameter correlation scores from components.

    This function determines the maximum robust correlation of each component
    time series with a model of 72 realignment parameters.

    Parameters
    ----------
    mel_mix : numpy.ndarray of shape (T, C)
        Mixing matrix in shape T (time) by C (component).
    mc : str or array_like
        Full path of the text file containing the realignment parameters.
        Motion parameters are (time x 6), with the first three columns being
        rotation parameters (in radians) and the final three being translation
        parameters (in mm).
    metric_metadata : None or dict, optional
        A dictionary containing metadata about the AROMA metrics.
        If provided, metadata for the ``max_RP_corr`` metric will be added.
        Otherwise, no operations will be performed on this parameter.

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
    if isinstance(metric_metadata, dict):
        metric_metadata["max_RP_corr"] = {
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
        }

    if isinstance(mc, str):
        rp6 = utils.load_motpars(mc, source="auto")
    else:
        rp6 = mc

    if (rp6.ndim != 2) or (rp6.shape[1] != 6):
        raise ValueError(f"Motion parameters must of shape (n_trs, 6), not {rp6.shape}")

    if rp6.shape[0] != mel_mix.shape[0]:
        raise ValueError(
            f"Number of rows in mixing matrix ({mel_mix.shape[0]}) does not match "
            f"number of rows in motion parameters ({rp6.shape[0]})."
        )

    # Determine the derivatives of the RPs (add zeros at time-point zero)
    _, nparams = rp6.shape
    rp6_der = np.vstack((np.zeros(nparams), np.diff(rp6, axis=0)))

    # Create an RP-model including the RPs and its derivatives
    rp12 = np.hstack((rp6, rp6_der))

    # add the fw and bw shifted versions
    rp12_1fw = np.vstack((np.zeros(2 * nparams), rp12[:-1]))
    rp12_1bw = np.vstack((rp12[1:], np.zeros(2 * nparams)))
    rp_model = np.hstack((rp12, rp12_1fw, rp12_1bw))

    # Determine the maximum correlation between RPs and IC time-series
    nsplits = 1000
    nmixrows, nmixcols = mel_mix.shape
    nrows_to_choose = int(round(0.9 * nmixrows))

    # Max correlations for multiple splits of the dataset (for a robust
    # estimate)
    max_correls = np.empty((nsplits, nmixcols))
    for i in range(nsplits):
        # Select a random subset of 90% of the dataset rows
        # (*without* replacement)
        chosen_rows = np.random.choice(a=range(nmixrows), size=nrows_to_choose, replace=False)

        # Combined correlations between RP and IC time-series, squared and
        # non squared
        correl_nonsquared = utils.cross_correlation(mel_mix[chosen_rows], rp_model[chosen_rows])
        correl_squared = utils.cross_correlation(
            mel_mix[chosen_rows] ** 2, rp_model[chosen_rows] ** 2
        )
        correl_both = np.hstack((correl_squared, correl_nonsquared))

        # Maximum absolute temporal correlation for every IC
        max_correls[i] = np.abs(correl_both).max(axis=1)

    # Feature score is the mean of the maximum correlation over all the random
    # splits
    # Avoid propagating occasional nans that arise in artificial test cases
    max_RP_corr = np.nanmean(max_correls, axis=0)
    return max_RP_corr, metric_metadata


def feature_frequency(mel_FT_mix: np.ndarray, TR: float, metric_metadata=None, f_hp: float = 0.01):
    """Extract the high-frequency content feature scores.

    This function determines the frequency, as fraction of the Nyquist
    frequency, at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    Parameters
    ----------
    mel_FT_mix : numpy.ndarray of shape (F, C)
        Stored array is (frequency x component), with frequencies
        ranging from 0 Hz to Nyquist frequency.
    TR : float
        TR (in seconds) of the fMRI data
    metric_metadata : None or dict, optional
        A dictionary containing metadata about the AROMA metrics.
        If provided, metadata for the ``HFC`` metric will be added.
        Otherwise, no operations will be performed on this parameter.
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
    if isinstance(metric_metadata, dict):
        metric_metadata["HFC"] = {
            "LongName": "High-frequency content",
            "Description": (
                "The proportion of the power spectrum for each component that falls above 0.01 Hz."
            ),
            "Units": "arbitrary",
        }

    # Determine sample frequency
    Fs = 1 / TR

    # Determine Nyquist-frequency
    Ny = Fs / 2

    n_frequencies = mel_FT_mix.shape[0]

    # Determine which frequencies are associated with every row in the
    # melodic_FTmix file (assuming the rows range from 0Hz to Nyquist)
    f = Ny * np.arange(1, n_frequencies + 1) / n_frequencies

    # Only include frequencies higher than f_hp Hz
    fincl = np.squeeze(np.array(np.where(f > f_hp)))
    mel_FT_mix = mel_FT_mix[fincl, :]
    f = f[fincl]

    # Set frequency range to [0-1]
    f_norm = (f - f_hp) / (Ny - f_hp)

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fract = np.cumsum(mel_FT_mix, axis=0) / np.sum(mel_FT_mix, axis=0)

    # Determine the index of the frequency with the fractional cumulative sum
    # closest to 0.5
    idx_cutoff = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)

    # Now get the fractions associated with those indices index, these are the
    # final feature scores
    HFC = f_norm[idx_cutoff]

    # Return feature score
    return HFC, metric_metadata


def feature_spatial(mel_IC, metric_metadata=None):
    """Extract the spatial feature scores.

    For each IC it determines the fraction of the mixture modeled thresholded
    Z-maps respectively located within the CSF or at the brain edges,
    using predefined standardized masks.

    Parameters
    ----------
    mel_IC : str or niimg_like
        Full path of the nii.gz file containing mixture-modeled thresholded
        (p<0.5) Z-maps, registered to the MNI152 2mm template
    metric_metadata : None or dict, optional
        A dictionary containing metadata about the AROMA metrics.
        If provided, metadata for the ``edge_fract`` and ``csf_fract`` metrics
        will be added.
        Otherwise, no operations will be performed on this parameter.

    Returns
    -------
    edge_fract : array_like
        Array of the edge fraction feature scores for the components of the
        mel_IC file
    csf_fract : array_like
        Array of the CSF fraction feature scores for the components of the
        mel_IC file
    metric_metadata : None or dict
        If the ``metric_metadata`` input was None, then None will be returned.
        Otherwise, this will be a dictionary containing existing information,
        as well as new metadata for the ``edge_fract`` and ``csf_fract``
        metrics.
    """
    if isinstance(metric_metadata, dict):
        metric_metadata["edge_fract"] = {
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
        }
        metric_metadata["csf_fract"] = {
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
        }

    # Get the number of ICs
    mel_IC_img = load_niimg(mel_IC)
    num_ICs = mel_IC_img.shape[3]

    masks_dir = utils.get_resource_path()
    csf_mask = os.path.join(masks_dir, "mask_csf.nii.gz")
    edge_mask = os.path.join(masks_dir, "mask_edge.nii.gz")
    out_mask = os.path.join(masks_dir, "mask_out.nii.gz")

    # Loop over ICs
    edge_fract = np.zeros(num_ICs)
    csf_fract = np.zeros(num_ICs)
    for i in range(num_ICs):
        # Extract IC from the merged melodic_IC_thr2MNI2mm file
        temp_IC = image.index_img(mel_IC, i)

        # Change to absolute Z-values
        temp_IC = image.math_img("np.abs(img)", img=temp_IC)

        # Get sum of Z-values within the total Z-map
        temp_IC_data = temp_IC.get_fdata()
        tot_sum = np.sum(temp_IC_data)

        if tot_sum == 0:
            LGR.info(
                "\t- The spatial map of component {} is empty. " "Please check!".format(i + 1)
            )

        # Get sum of Z-values of the voxels located within the CSF
        csf_data = masking.apply_mask(temp_IC, csf_mask)
        csf_sum = np.sum(csf_data)

        # Get sum of Z-values of the voxels located within the Edge
        edge_data = masking.apply_mask(temp_IC, edge_mask)
        edge_sum = np.sum(edge_data)

        # Get sum of Z-values of the voxels located outside the brain
        out_data = masking.apply_mask(temp_IC, out_mask)
        out_sum = np.sum(out_data)

        # Determine edge and CSF fraction
        if tot_sum != 0:
            edge_fract[i] = (out_sum + edge_sum) / (tot_sum - csf_sum)
            csf_fract[i] = csf_sum / tot_sum
        else:
            edge_fract[i] = 0
            csf_fract[i] = 0

    # Return feature scores
    return edge_fract, csf_fract, metric_metadata
