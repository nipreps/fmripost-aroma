"""Plotting functions for ICA-AROMA."""

import logging
import os

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.use('Agg')
LGR = logging.getLogger(__name__)


def classification_plot(in_file, out_dir):
    """Generate a figure to show classifications.

    Parameters
    ----------
    in_file : str
        Path to tab-delimited file with feature scores and classification.
    out_dir : str
        Output directory.
    """
    if not os.path.isfile(in_file):
        raise FileNotFoundError(f'Input file does not exist: {in_file}')

    df = pd.read_csv(in_file, sep='\t', index_col='IC')
    motion_components_df = df.loc[df['classification'] == 'rejected']

    # get counts
    n_components = df.shape[0]
    n_motion_components = motion_components_df.shape[0]
    LGR.info(
        f'Found {n_motion_components} head motion-related components in a total of '
        f'{n_components} components.'
    )

    # add dummy components if needed, this is just for making the plots look nice
    if n_motion_components < 3:
        temp_df = pd.DataFrame(
            {
                'classification': ['rejected', 'rejected', 'rejected'],
                'max_RP_corr': [1.0, 1.0, 1.0],
                'edge_fract': [1.0, 1.0, 1.0],
                'HFC': [0.0, 0.0, 0.0],
                'csf_fract': [0.0, 0.0, 0.0],
            }
        )
        df = df.append(temp_df, ignore_index=True)

    if df.loc[df['classification'] == 'accepted'].shape[0] < 3:
        temp_df = pd.DataFrame(
            {
                'classification': ['accepted', 'accepted', 'accepted'],
                'max_RP_corr': [0.0, 0.0, 0.0],
                'edge_fract': [0.0, 0.0, 0.0],
                'HFC': [0.0, 0.0, 0.0],
                'csf_fract': [0.0, 0.0, 0.0],
            }
        )
        df = df.append(temp_df, ignore_index=True)

    # rename columns
    df = df.rename(
        columns={
            'classification': 'Motion',
            'max_RP_corr': 'RP',
            'edge_fract': 'Edge',
            'HFC': 'Freq',
            'csf_fract': 'CSF',
        }
    )
    df['Motion'] = df['Motion'].map({'rejected': 'True', 'accepted': 'False'})

    # Make pretty figure
    # styling
    sns.set_style('white')
    colortrue = '#FFBF17'
    colorfalse = '#69A00A'
    # plot Edge/RP relationship
    # obtain projection line
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])
    a = -hyp[1] / hyp[2]
    xx = np.linspace(0, 1)
    yy = a * xx - hyp[0] / hyp[2]
    # create aux figure
    h = sns.jointplot(
        x='RP',
        y='Edge',
        data=df,
        hue=df['Motion'],
        kind='scatter',
        palette=[colortrue, colorfalse],
        hue_order=['True', 'False'],
        xlim=[0, 1],
        ylim=[0, 1],
    )
    h.set_axis_labels('Maximum RP Correlation', 'Edge Fraction', fontsize=14, labelpad=10)
    h.ax_joint.set_xticks(np.arange(0, 1.2, 0.2))
    h.ax_joint.set_yticks(np.arange(0, 1.2, 0.2))
    h.ax_joint.tick_params(axis='both', labelsize=12)
    h.ax_joint.plot(xx, yy, '.', color='k', markersize=1)
    h.savefig(os.path.join(out_dir, 'aux_fig.png'), bbox_inches='tight', dpi=300)
    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[12, 4])

    # define subplots
    # Edge/RP

    # plot Freq
    sns.boxplot(
        x='Motion',
        y='Freq',
        data=df,
        ax=ax2,
        palette=[colortrue, colorfalse],
        order=['True', 'False'],
    )
    ax2.hlines(0.35, -1, 2, zorder=0, linestyles='dotted', linewidth=0.5)
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Classification', fontsize=14, labelpad=10)
    ax2.set_ylabel('High-Frequency Content', fontsize=14)
    ax2.set_xticklabels(['Motion', 'Other'])
    ax2.tick_params(axis='both', labelsize=12)
    sns.despine(ax=ax2)

    # plot CSF
    sns.boxplot(
        x='Motion',
        y='CSF',
        data=df,
        ax=ax3,
        palette=[colortrue, colorfalse],
        order=['True', 'False'],
    )
    ax3.hlines(0.1, -1, 2, zorder=0, linestyles='dotted', linewidth=0.5)
    ax3.set_ylim([0, 1])
    ax3.set_xlabel('Classification', fontsize=14, labelpad=10)
    ax3.set_ylabel('CSF Fraction', fontsize=14)
    ax3.set_xticklabels(['Motion', 'Other'])
    ax3.tick_params(axis='both', labelsize=12)
    sns.despine(ax=ax3)

    # plot Edge/RP relationship
    # obtain projection line
    aux_img = mpimg.imread(os.path.join(out_dir, 'aux_fig.png'))
    ax1.imshow(aux_img)
    ax1.axis('off')
    fig.tight_layout()

    # bring tickmarks back
    for myax in fig.get_axes():
        myax.tick_params(which='major', direction='in', length=3)

    # add figure title
    fig.suptitle('Component Assessment', fontsize=20, y=1.08)

    # outtakes
    plt.savefig(
        os.path.join(out_dir, 'ICA_AROMA_component_assessment.svg'),
        bbox_inches='tight',
        dpi=300,
    )
