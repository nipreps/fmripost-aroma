# fmripost-aroma

[![Docker Image](https://img.shields.io/badge/docker-nipreps/fmripost--aroma-brightgreen.svg?logo=docker&style=flat)](https://hub.docker.com/r/nipreps/fmripost-aroma/tags/)
[![PyPI - Version](https://img.shields.io/pypi/v/fmripost-aroma.svg)](https://pypi.org/project/fmripost-aroma)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fmripost-aroma.svg)](https://pypi.org/project/fmripost-aroma)

fMRIPost-AROMA is a BIDS App for running ICA-AROMA on a preprocessed
fMRI dataset.
It is intended to replace the built-in workflow in fMRIPrep 23.0 and earlier,
but accepts any BIDS derivatives dataset where BOLD images have been resampled
to MNI152NLin6Asym at 2 mm3 resolution.

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
docker pull nipreps/fmripost-aroma:main
```

## License

`fmripost-aroma` is distributed under the terms of the [Apache 2](https://spdx.org/licenses/Apache-2.0.html) license.
