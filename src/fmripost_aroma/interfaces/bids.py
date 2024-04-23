"""BIDS-related interfaces for fMRIPost-AROMA."""

from nipype import logging
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink

LOGGER = logging.getLogger("nipype.interface")


class DerivativesDataSink(BaseDerivativesDataSink):
    """Store derivative files.

    A child class of the niworkflows DerivativesDataSink, using xcp_d's configuration files.
    """

    out_path_base = ""
