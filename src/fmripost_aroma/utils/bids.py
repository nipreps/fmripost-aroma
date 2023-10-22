"""BIDS-related utility functions."""


def collect_data(
    layout,
    subject_id,
    task_id=None,
    bids_filters=None,
):
    """Collect preprocessing derivatives."""
    query = {
        "bold": {
            "space": "MNI152NLin6Asym",
            "res": 2,
            "desc": "preproc",
            "suffix": "bold",
            "extension": [".nii", ".nii.gz"],
        }
    }
    subj_data = layout.get(**query)
    return subj_data


def collect_run_data(
    layout,
    bold_file,
):
    """Collect files and metadata related to a given BOLD file."""
    queries = {}
    run_data = {
        "mask": {"desc": "brain", "suffix": "mask", "extension": [".nii", ".nii.gz"]},
        "confounds": {"desc": "confounds", "suffix": "timeseries", "extension": ".tsv"},
    }
    for k, v in queries.items():
        run_data[k] = layout.get_nearest(bold_file, **v)

    return run_data
