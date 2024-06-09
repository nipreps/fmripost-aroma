.. include:: links.rst

.. _Usage :

Usage Notes
===========
.. warning::
   *fMRIPost-AROMA* includes a tracking system to report usage statistics and errors
   for debugging and grant reporting purposes.
   Users can opt-out using the ``--notrack`` command line argument.


Execution and the BIDS format
-----------------------------
The *fMRIPost-AROMA* workflow takes as principal input the path of the dataset
that is to be processed.
The input dataset is required to be in valid :abbr:`BIDS (Brain Imaging Data
Structure)` format, and it must include at least one T1w structural image and
(unless disabled with a flag) a BOLD series.
We highly recommend that you validate your dataset with the free, online
`BIDS Validator <https://bids-standard.github.io/bids-validator/>`_.

The exact command to run *fMRIPost-AROMA* depends on the Installation_ method.
The common parts of the command follow the `BIDS-Apps
<https://github.com/BIDS-Apps>`_ definition.
Example: ::

    fmripost_aroma data/bids_root/ out/ participant -w work/

Further information about BIDS and BIDS-Apps can be found at the
`NiPreps portal <https://www.nipreps.org/apps/framework/>`__.


Command-Line Arguments
----------------------
.. argparse::
   :ref: fmripost_aroma.cli.parser._build_parser
   :prog: fmripost_aroma
   :nodefault:
   :nodefaultconst:

.. _prev_derivs:

Reusing precomputed derivatives
-------------------------------

Reusing a previous, partial execution of *fMRIPost-AROMA*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*fMRIPost-AROMA* will pick up where it left off a previous execution, so long as the work directory
points to the same location, and this directory has not been changed/manipulated.
Some workflow nodes will rerun unconditionally, so there will always be some amount of
reprocessing.

Troubleshooting
---------------
Logs and crashfiles are output into the
``<output dir>/fmripost_aroma/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`Debugging Nipype Workflows <https://miykael.github.io/nipype_tutorial/notebooks/basic_debug.html>`_
page.

**Support and communication**.
The documentation of this project is found here: https://fmripost_aroma.org/en/latest/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/nipreps/fmripost_aroma/issues.

If you have a problem or would like to ask a question about how to use *fMRIPost-AROMA*,
please submit a question to `NeuroStars.org <https://neurostars.org/tag/fmripost_aroma>`_ with an ``fmripost_aroma`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

Previous questions about *fMRIPost-AROMA* are available here:
https://neurostars.org/tag/fmripost_aroma/

To participate in the *fMRIPost-AROMA* development-related discussions please use the
following mailing list: https://mail.python.org/mailman/listinfo/neuroimaging
Please add *[fmripost_aroma]* to the subject line when posting on the mailing list.


.. include:: license.rst
