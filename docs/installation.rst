.. include:: links.rst

------------
Installation
------------
There are two ways to install *fMRIPost-AROMA*:

* using container technologies (RECOMMENDED); or
* within a `Manually Prepared Environment (Python 3.10+)`_, also known as
  *bare-metal installation*.

The ``fmripost-aroma`` command-line adheres to the `BIDS-Apps recommendations
for the user interface <usage.html#execution-and-the-bids-format>`__.
Therefore, the command-line has the following structure::

  $ fmripost-aroma <input_bids_path> <derivatives_path> <analysis_level> <named_options>

The ``fmripost-aroma`` command-line options are documented in the :ref:`usage`
section.

The command as shown works for a *bare-metal* environment set-up (second option above).
If you choose the recommended container-based installation, then
the command-line will be composed of a preamble to configure the
container execution followed by the ``fmripost-aroma`` command-line options
as if you were running it on a *bare-metal* installation.
The command-line structure above is then modified as follows::

  $ <container_command_and_options> <container_image> \
       <input_bids_path> <derivatives_path> <analysis_level> <fmriprep_named_options>

Therefore, once specified the container options and the image to be run
the command line is the same as for the *bare-metal* installation but dropping
the ``fmripost-aroma`` executable name.

Containerized execution (Docker and Singularity)
================================================
*fMRIPost-AROMA* is a *NiPreps* application, and therefore follows some overarching principles
of containerized execution drawn from the BIDS-Apps protocols.
For detailed information of containerized execution of *NiPreps*, please visit the corresponding
`Docker <https://www.nipreps.org/apps/docker/>`__
or `Singularity <https://www.nipreps.org/apps/singularity/>`__ subsections.


Manually Prepared Environment (Python 3.10+)
============================================

.. warning::

   This method is not recommended! Please consider using containers.

Make sure all of *fMRIPost-AROMA*'s `External Dependencies`_ are installed.
These tools must be installed and their binaries available in the
system's ``$PATH``.
A relatively interpretable description of how your environment can be set-up
is found in the `Dockerfile <https://github.com/nipreps/fmripost-aroma/blob/master/Dockerfile>`_.
As an additional installation setting, FreeSurfer requires a license file (see :ref:`fs_license`).

On a functional Python 3.10 (or above) environment with ``pip`` installed,
*fMRIPost-AROMA* can be installed using the habitual command ::

    $ python -m pip install fmripost-aroma

Check your installation with the ``--version`` argument ::

    $ fmripost-aroma --version


External Dependencies
---------------------
*fMRIPost-AROMA* is written using Python 3.8 (or above), and is based on
nipype_.

*fMRIPost-AROMA* requires some other neuroimaging software tools that are
not handled by the Python's packaging system (Pypi) used to deploy
the ``fmripost-aroma`` package:

- FSL_ (version 6.0.7.7)
- ANTs_ (version 2.5.1)
- AFNI_ (version 24.0.05)
- `C3D <https://sourceforge.net/projects/c3d/>`_ (version 1.4.0)
- FreeSurfer_ (version 7.3.2)
- `bids-validator <https://github.com/bids-standard/bids-validator>`_ (version 1.14.0)
- `connectome-workbench <https://www.humanconnectome.org/software/connectome-workbench>`_ (version 1.5.0)

Not running on a local machine? - Data transfer
===============================================
If you intend to run *fMRIPost-AROMA* on a remote system, you will need to
make your data available within that system first.

For instance, here at the Poldrack Lab we use Stanford's
:abbr:`HPC (high-performance computing)` system, called Sherlock.
Sherlock enables `the following data transfer options
<https://www.sherlock.stanford.edu/docs/user-guide/storage/data-transfer/>`_.

Alternatively, more comprehensive solutions such as `Datalad
<https://www.datalad.org/>`_ will handle data transfers with the appropriate
settings and commands.
Datalad also performs version control over your data.
