# hash:sha256:eacb79786f81d2e61d792e9657cf9d3c9f7f7dacaab64cba5e4afc2c8ae60e62
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    h5py==3.16.0 \
    nibabel \
    numpy==2.5.0 \
    pynrrd==1.1.3 \
    scipy==1.18.0
