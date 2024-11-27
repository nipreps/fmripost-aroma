# fMRIPost-template Docker Container Image distribution
#
# MIT License
#
# Copyright (c) The NiPreps Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Ubuntu 22.04 LTS - Jammy
ARG BASE_IMAGE=ubuntu:jammy-20240125

#
# Build wheel
#
FROM python:slim AS src
RUN pip install build
RUN apt-get update && \
    apt-get install -y --no-install-recommends git
COPY . /src
RUN python -m build /src

#
# Download stages
#

# Utilities for downloading packages
FROM ${BASE_IMAGE} AS downloader
# Bump the date to current to refresh curl/certificates/etc
RUN echo "2023.07.20"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    binutils \
                    bzip2 \
                    ca-certificates \
                    curl \
                    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# AFNI
FROM downloader AS afni
# Bump the date to current to update AFNI
RUN echo "2023.07.20"
RUN mkdir -p /opt/afni-latest \
    && curl -fsSL --retry 5 https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \
    | tar -xz -C /opt/afni-latest --strip-components 1 \
    --exclude "linux_openmp_64/*.gz" \
    --exclude "linux_openmp_64/funstuff" \
    --exclude "linux_openmp_64/shiny" \
    --exclude "linux_openmp_64/afnipy" \
    --exclude "linux_openmp_64/lib/RetroTS" \
    --exclude "linux_openmp_64/lib_RetroTS" \
    --exclude "linux_openmp_64/meica.libs" \
    # Keep only what we use
    && find /opt/afni-latest -type f -not -name "3dTshift" -delete

# Connectome Workbench 1.5.0
FROM downloader AS workbench
RUN mkdir /opt/workbench && \
    curl -sSLO https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip && \
    unzip workbench-linux64-v1.5.0.zip -d /opt && \
    rm workbench-linux64-v1.5.0.zip && \
    rm -rf /opt/workbench/libs_linux64_software_opengl /opt/workbench/plugins_linux64 && \
    strip --remove-section=.note.ABI-tag /opt/workbench/libs_linux64/libQt5Core.so.5

# Micromamba
FROM downloader AS micromamba

# Install a C compiler to build extensions when needed.
# traits<6.4 wheels are not available for Python 3.11+, but build easily.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /
# Bump the date to current to force update micromamba
RUN echo "2024.02.06"
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

ENV MAMBA_ROOT_PREFIX="/opt/conda"
COPY env.yml /tmp/env.yml
WORKDIR /tmp
RUN micromamba create -y -f /tmp/env.yml && \
    micromamba clean -y -a

# UV_USE_IO_URING for apparent race-condition (https://github.com/nodejs/node/issues/48444)
# Check if this is still necessary when updating the base image.
ENV PATH="/opt/conda/envs/fmripost_template/bin:$PATH" \
    UV_USE_IO_URING=0
RUN npm install -g svgo@^3.2.0 bids-validator@1.14.10 && \
    rm -r ~/.npm

#
# Main stage
#
FROM ${BASE_IMAGE} AS fmripost_template

# Configure apt
ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"

# Some baseline tools; bc is needed for FreeSurfer, so don't drop it
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    bc \
                    ca-certificates \
                    curl \
                    gcc \
                    git \
                    gnupg \
                    libc-dev \
                    lsb-release \
                    netbase \
                    xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Configure PPAs for libpng12 and libxp6
RUN GNUPGHOME=/tmp gpg --keyserver hkps://keyserver.ubuntu.com --no-default-keyring --keyring /usr/share/keyrings/linuxuprising.gpg --recv 0xEA8CACC073C3DB2A \
    && GNUPGHOME=/tmp gpg --keyserver hkps://keyserver.ubuntu.com --no-default-keyring --keyring /usr/share/keyrings/zeehio.gpg --recv 0xA1301338A3A48C4A \
    && echo "deb [signed-by=/usr/share/keyrings/linuxuprising.gpg] https://ppa.launchpadcontent.net/linuxuprising/libpng12/ubuntu jammy main" > /etc/apt/sources.list.d/linuxuprising.list \
    && echo "deb [signed-by=/usr/share/keyrings/zeehio.gpg] https://ppa.launchpadcontent.net/zeehio/libxp/ubuntu jammy main" > /etc/apt/sources.list.d/zeehio.list

# Dependencies for AFNI; requires a discontinued multiarch-support package from bionic (18.04)
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           ed \
           gsl-bin \
           libglib2.0-0 \
           libglu1-mesa-dev \
           libglw1-mesa \
           libgomp1 \
           libjpeg62 \
           libpng12-0 \
           libxm4 \
           libxp6 \
           netpbm \
           tcsh \
           xfonts-base \
           xvfb \
    && curl -sSL --retry 5 -o /tmp/multiarch.deb http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/multiarch-support_2.27-3ubuntu1.5_amd64.deb \
    && dpkg -i /tmp/multiarch.deb \
    && rm /tmp/multiarch.deb \
    && apt-get install -f \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && gsl2_path="$(find / -name 'libgsl.so.19' || printf '')" \
    && if [ -n "$gsl2_path" ]; then \
         ln -sfv "$gsl2_path" "$(dirname $gsl2_path)/libgsl.so.0"; \
    fi \
    && ldconfig

# Install files from stages
COPY --from=afni /opt/afni-latest /opt/afni-latest
COPY --from=workbench /opt/workbench /opt/workbench

# AFNI config
ENV PATH="/opt/afni-latest:$PATH" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_PLUGINPATH="/opt/afni-latest"

# Workbench config
ENV PATH="/opt/workbench/bin_linux64:$PATH" \
    LD_LIBRARY_PATH="/opt/workbench/lib_linux64:$LD_LIBRARY_PATH"

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users fmripost_template
WORKDIR /home/fmripost_template
ENV HOME="/home/fmripost_template" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=micromamba /opt/conda/envs/fmripost_template /opt/conda/envs/fmripost_template

ENV MAMBA_ROOT_PREFIX="/opt/conda"
RUN micromamba shell init -s bash && \
    echo "micromamba activate fmripost_template" >> $HOME/.bashrc
ENV PATH="/opt/conda/envs/fmripost_template/bin:$PATH" \
    CPATH="/opt/conda/envs/fmripost_template/include:$CPATH" \
    LD_LIBRARY_PATH="/opt/conda/envs/fmripost_template/lib:$LD_LIBRARY_PATH"

# Precaching atlases
COPY scripts/fetch_templates.py fetch_templates.py
RUN python fetch_templates.py && \
    rm fetch_templates.py && \
    find $HOME/.cache/templateflow -type d -exec chmod go=u {} + && \
    find $HOME/.cache/templateflow -type f -exec chmod go=u {} +

# FSL environment
ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1 \
    FSLDIR="/opt/conda/envs/fmripost_template" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Installing fMRIPost-template
COPY --from=src /src/dist/*.whl .
RUN pip install --no-cache-dir $( ls *.whl )[test]

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} + && \
    rm -rf $HOME/.npm $HOME/.conda $HOME/.empty

# For detecting the container
ENV IS_DOCKER_8395080871=1

RUN ldconfig
WORKDIR /tmp
ENTRYPOINT ["/opt/conda/envs/fmripost_template/bin/fmripost-template"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="fMRIPost-template" \
      org.label-schema.description="fMRIPost-template - A template fMRI post-processing workflow" \
      org.label-schema.url="https://fmripost_template.org" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/fmripost_template" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
