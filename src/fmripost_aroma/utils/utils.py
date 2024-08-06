"""Utility functions for ICA-AROMA."""

import json
import logging
import os.path as op
import shutil

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn import masking
from nilearn._utils import load_niimg

# Define criteria needed for classification (thresholds and
# hyperplane-parameters)
THR_CSF = 0.10
THR_HFC = 0.35
HYPERPLANE = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

LGR = logging.getLogger(__name__)


def cross_correlation(a, b):
    """Perform cross-correlations between columns of two matrices.

    Parameters
    ----------
    a : (M x X) array_like
        First array to cross-correlate
    b : (N x X) array_like
        Second array to cross-correlate

    Returns
    -------
    correlations : (M x N) array_like
        Cross-correlations of columns of a against columns of b.
    """
    if a.ndim != 2:
        raise ValueError(f'Input `a` must be 2D, not {a.ndim}D')

    if b.ndim != 2:
        raise ValueError(f'Input `b` must be 2D, not {b.ndim}D')

    _, ncols_a = a.shape
    # nb variables in columns rather than rows hence transpose
    # extract just the cross terms between cols in a and cols in b
    return np.corrcoef(a.T, b.T)[:ncols_a, ncols_a:]


def classification(features_df: pd.DataFrame):
    """Classify components as motion or non-motion based on four features.

    The four features used for classification are: maximum RP correlation,
    high-frequency content, edge-fraction, and CSF-fraction.

    Parameters
    ----------
    features_df : (C x 4) :obj:`pandas.DataFrame`
        DataFrame with the following columns:
        "edge_fract", "csf_fract", "max_RP_corr", and "HFC".

    Returns
    -------
    clf_df
    clf_metadata
    """
    clf_metadata = {
        'classification': {
            'LongName': 'Component classification',
            'Description': ('Classification from the classification procedure.'),
            'Levels': {
                'accepted': 'A component that is determined not to be associated with motion.',
                'rejected': 'A motion-related component.',
            },
        },
        'rationale': {
            'LongName': 'Rationale for component classification',
            'Description': (
                'The reason for the classification. '
                'In cases where components are classified based on more than one criterion, '
                'they are listed sequentially, separated by semicolons.'
            ),
            'Levels': {
                'CSF': f'The csf_fract value is higher than {THR_CSF}',
                'HFC': f'The HFC value is higher than {THR_HFC}',
                'hyperplane': (
                    'After the max_RP_corr and edge_fract values are projected '
                    'to a hyperplane, the projected point is less than zero.'
                ),
            },
        },
    }

    # Classify the ICs as motion (rejected) or non-motion (accepted)
    clf_df = pd.DataFrame(index=features_df.index, columns=['classification', 'rationale'])
    clf_df['classification'] = 'accepted'
    clf_df['rationale'] = ''

    # CSF
    rej_csf = features_df['csf_fract'] > THR_CSF
    clf_df.loc[rej_csf, 'classification'] = 'rejected'
    clf_df.loc[rej_csf, 'rationale'] += 'CSF;'

    # HFC
    rej_hfc = features_df['HFC'] > THR_HFC
    clf_df.loc[rej_hfc, 'classification'] = 'rejected'
    clf_df.loc[rej_hfc, 'rationale'] += 'HFC;'

    # Hyperplane
    # Project edge & max_RP_corr feature scores to new 1D space
    x = features_df[['max_RP_corr', 'edge_fract']].values
    proj = HYPERPLANE[0] + np.dot(x, HYPERPLANE[1:])
    rej_hyperplane = proj > 0
    clf_df.loc[rej_hyperplane, 'classification'] = 'rejected'
    clf_df.loc[rej_hyperplane, 'rationale'] += 'hyperplane;'

    # Check the classifications
    is_motion = (features_df['csf_fract'] > THR_CSF) | (features_df['HFC'] > THR_HFC) | (proj > 0)
    if not np.array_equal(is_motion, (clf_df['classification'] == 'rejected').values):
        raise ValueError('Classification error: classifications do not match criteria.')

    # Remove trailing semicolons
    clf_df['rationale'] = clf_df['rationale'].str.rstrip(';')

    return clf_df, clf_metadata


def write_metrics(features_df, out_dir, metric_metadata=None):
    """Write out feature/classification information and metadata.

    Parameters
    ----------
    features_df : (C x 5) :obj:`pandas.DataFrame`
        DataFrame with metric values and classifications.
        Must have the following columns: "edge_fract", "csf_fract", "max_RP_corr", "HFC", and
        "classification".
    out_dir : :obj:`str`
        Output directory.
    metric_metadata : :obj:`dict` or None, optional
        Metric metadata in a dictionary.

    Returns
    -------
    motion_ICs : array_like
        Array containing the indices of the components identified as motion components.

    Output
    ------
    AROMAnoiseICs.csv : A text file containing the indices of the
                        components identified as motion components
    desc-AROMA_metrics.tsv
    desc-AROMA_metrics.json
    """
    # Put the indices of motion-classified ICs in a text file (starting with 1)
    motion_ICs = features_df['classification'][features_df['classification'] == 'rejected'].index
    motion_ICs = motion_ICs.values

    with open(op.join(out_dir, 'AROMAnoiseICs.csv'), 'w') as file_obj:
        out_str = ','.join(motion_ICs.astype(str))
        file_obj.write(out_str)

    # Create a summary overview of the classification
    out_file = op.join(out_dir, 'desc-AROMA_metrics.tsv')
    features_df.to_csv(out_file, sep='\t', index_label='IC')

    if isinstance(metric_metadata, dict):
        with open(op.join(out_dir, 'desc-AROMA_metrics.json'), 'w') as file_obj:
            json.dump(metric_metadata, file_obj, sort_keys=True, indent=4)

    return motion_ICs


def denoising(in_file, out_dir, mixing, den_type, den_idx):
    """Remove noise components from fMRI data.

    Parameters
    ----------
    in_file : str
        Full path to the data file (nii.gz) which has to be denoised
    out_dir : str
        Full path of the output directory
    mixing : numpy.ndarray of shape (T, C)
        Mixing matrix.
    den_type : {"aggr", "nonaggr", "both"}
        Type of requested denoising ('aggr': aggressive, 'nonaggr':
        non-aggressive, 'both': both aggressive and non-aggressive
    den_idx : array_like
        Index of the components that should be regressed out

    Output
    ------
    desc-smoothAROMA<den_type>_bold.nii.gz : The denoised fMRI data
    """
    # Check if denoising is needed (i.e. are there motion components?)
    motion_components_found = den_idx.size > 0

    nonaggr_denoised_file = op.join(out_dir, 'desc-smoothAROMAnonaggr_bold.nii.gz')
    aggr_denoised_file = op.join(out_dir, 'desc-smoothAROMAaggr_bold.nii.gz')

    if motion_components_found:
        motion_components = mixing[:, den_idx]

        # Create a fake mask to make it easier to reshape the full data to 2D
        img = load_niimg(in_file)
        full_mask = nb.Nifti1Image(np.ones(img.shape[:3], int), img.affine)
        data = masking.apply_mask(img, full_mask)  # T x S

        # Non-aggressive denoising of the data using fsl_regfilt
        # (partial regression), if requested
        if den_type in ('nonaggr', 'both'):
            # Fit GLM to all components
            betas = np.linalg.lstsq(mixing, data, rcond=None)[0]

            # Denoise the data using the betas from just the bad components.
            pred_data = np.dot(motion_components, betas[den_idx, :])
            data_denoised = data - pred_data

            # Save to file.
            img_denoised = masking.unmask(data_denoised, full_mask)
            img_denoised.to_filename(nonaggr_denoised_file)

        # Aggressive denoising of the data using fsl_regfilt (full regression)
        if den_type in ('aggr', 'both'):
            # Denoise the data with the bad components.
            betas = np.linalg.lstsq(motion_components, data, rcond=None)[0]
            pred_data = np.dot(motion_components, betas)
            data_denoised = data - pred_data

            # Save to file.
            img_denoised = masking.unmask(data_denoised, full_mask)
            img_denoised.to_filename(aggr_denoised_file)
    else:
        LGR.warning(
            '  - None of the components were classified as motion, '
            'so no denoising is applied (the input file is copied '
            'as-is).'
        )
        if den_type in ('nonaggr', 'both'):
            shutil.copyfile(in_file, nonaggr_denoised_file)

        if den_type in ('aggr', 'both'):
            shutil.copyfile(in_file, aggr_denoised_file)


def motpars_fmriprep2fsl(confounds):
    """Convert fMRIPrep motion parameters to FSL format.

    Parameters
    ----------
    confounds : str or pandas.DataFrame
        Confounds data from fMRIPrep.
        Relevant columns have the format "[rot|trans]_[x|y|z]".
        Rotations are in radians.

    Returns
    -------
    motpars_fsl : (T x 6) numpy.ndarray
        Motion parameters in FSL format, with rotations first (in radians) and
        translations second.
    """
    if isinstance(confounds, str) and op.isfile(confounds):
        confounds = pd.read_table(confounds)
    elif not isinstance(confounds, pd.DataFrame):
        raise ValueError('Input must be an existing file or a DataFrame.')

    # Rotations are in radians
    motpars_fsl = confounds[['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']].values
    return motpars_fsl


def motpars_spm2fsl(motpars):
    """Convert SPM format motion parameters to FSL format.

    Parameters
    ----------
    motpars : str or array_like
        SPM-format motion parameters.
        Rotations are in degrees and translations come first.

    Returns
    -------
    motpars_fsl : (T x 6) numpy.ndarray
        Motion parameters in FSL format, with rotations first (in radians) and
        translations second.
    """
    if isinstance(motpars, str) and op.isfile(motpars):
        motpars = np.loadtxt(motpars)
    elif not isinstance(motpars, np.ndarray):
        raise ValueError('Input must be an existing file or a numpy array.')

    if motpars.shape[1] != 6:
        raise ValueError(f'Motion parameters must have exactly 6 columns, not {motpars.shape[1]}.')

    # Split translations from rotations
    trans, rot = motpars[:, :3], motpars[:, 3:]

    # Convert rotations from degrees to radians
    rot *= np.pi / 180.0

    # Place rotations first
    motpars_fsl = np.hstack((rot, trans))
    return motpars_fsl


def motpars_afni2fsl(motpars):
    """Convert AFNI format motion parameters to FSL format.

    Parameters
    ----------
    motpars : str or array_like
        AfNI-format motion parameters in 1D file.
        Rotations are in degrees and translations come first.

    Returns
    -------
    motpars_fsl : (T x 6) numpy.ndarray
        Motion parameters in FSL format, with rotations first (in radians) and
        translations second.
    """
    if isinstance(motpars, str) and op.isfile(motpars):
        motpars = np.loadtxt(motpars)
    elif not isinstance(motpars, np.ndarray):
        raise ValueError('Input must be an existing file or a numpy array.')

    if motpars.shape[1] != 6:
        raise ValueError(f'Motion parameters must have exactly 6 columns, not {motpars.shape[1]}.')

    # Split translations from rotations
    trans, rot = motpars[:, :3], motpars[:, 3:]

    # Convert rotations from degrees to radians
    rot *= np.pi / 180.0

    # Place rotations first
    motpars_fsl = np.hstack((rot, trans))
    return motpars_fsl


def load_motpars(motion_file, source='auto'):
    """Load motion parameters from file.

    Parameters
    ----------
    motion_file : str
        Motion file.
    source : {"auto", "spm", "afni", "fsl", "fmriprep"}, optional
        Source of the motion data.
        If "auto", try to deduce the source based on the name of the file.

    Returns
    -------
    motpars : (T x 6) numpy.ndarray
        Motion parameters in FSL format, with rotations first (in radians) and
        translations second.
    """
    if source == 'auto':
        if op.basename(motion_file).startswith('rp_') and motion_file.endswith('.txt'):
            source = 'spm'
        elif motion_file.endswith('.1D'):
            source = 'afni'
        elif motion_file.endswith('.tsv'):
            source = 'fmriprep'
        elif motion_file.endswith('.txt'):
            source = 'fsl'
        else:
            raise Exception('Motion parameter source could not be determined automatically.')

    if source == 'spm':
        motpars = motpars_spm2fsl(motion_file)
    elif source == 'afni':
        motpars = motpars_afni2fsl(motion_file)
    elif source == 'fsl':
        motpars = np.loadtxt(motion_file)
    elif source == 'fmriprep':
        motpars = motpars_fmriprep2fsl(motion_file)
    else:
        raise ValueError(f'Source "{source}" not supported.')

    return motpars


def get_resource_path():
    """Return the path to general resources.

    Returns the path to general resources, terminated with separator.
    Resources are kept outside package folder in "resources".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.

    Returns
    -------
    resource_path : str
        Absolute path to resources folder.
    """
    return op.abspath(op.join(op.dirname(__file__), 'resources') + op.sep)


def get_spectrum(data: np.array, tr: float):
    """Return the power spectrum and corresponding frequencies of a time series.

    Parameters
    ----------
    data : numpy.ndarray of shape (T, C) or (T,)
        A time series of shape T (time) by C (component),
        on which you would like to perform an fft.
    tr : :obj:`float`
        Repetition time (TR) of the data, in seconds.

    Returns
    -------
    power_spectrum : numpy.ndarray of shape (F, C)
        Power spectrum of the input time series. C is component, F is frequency.
    freqs : numpy.ndarray of shape (F,)
        Frequencies corresponding to the columns of power_spectrum.
    """
    if data.ndim > 2:
        raise ValueError(f'Input `data` must be 1D or 2D, not {data.ndim}D')

    if data.ndim == 1:
        data = data[:, None]

    power_spectrum = np.abs(np.fft.rfft(data, axis=0)) ** 2
    freqs = np.fft.rfftfreq((power_spectrum.shape[0] * 2) - 1, tr)
    idx = np.argsort(freqs)
    return power_spectrum[idx, :], freqs[idx]


def _get_wf_name(bold_fname, prefix):
    """Derive the workflow name for supplied BOLD file.

    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_bold.nii.gz", "aroma")
    'aroma_task_nback_wf'
    >>> _get_wf_name(
    ...     "/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz",
    ...     "preproc",
    ... )
    'preproc_task_nback_run_01_echo_1_wf'

    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = '_'.join(fname.split('_')[1:-1])
    return f"{prefix}_{fname_nosub.replace('-', '_')}_wf"


def update_dict(orig_dict, new_dict):
    """Update dictionary with values from another dictionary.

    Parameters
    ----------
    orig_dict : dict
        Original dictionary.
    new_dict : dict
        Dictionary with new values.

    Returns
    -------
    updated_dict : dict
        Updated dictionary.
    """
    updated_dict = orig_dict.copy()
    for key, value in new_dict.items():
        if (orig_dict.get(key) is not None) and (value is not None):
            print(f'Updating {key} from {orig_dict[key]} to {value}')
            updated_dict[key].update(value)
        elif value is not None:
            updated_dict[key] = value

    return updated_dict


def _convert_to_tsv(in_file):
    """Convert a file to TSV format.

    Parameters
    ----------
    in_file : str
        Input file.

    Returns
    -------
    out_file : str
        Output file.
    """
    import os

    import numpy as np

    out_file = os.path.abspath('out_file.tsv')
    arr = np.loadtxt(in_file)
    np.savetxt(out_file, arr, delimiter='\t')
    return out_file
