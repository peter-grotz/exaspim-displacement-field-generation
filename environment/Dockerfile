# hash:sha256:fb5b1bbc551eb63575140a2dc468ae16ae875a3985f91ec6da09a961ac6222c6
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    h5py==3.16.0 \
    numpy==2.5.0 \
    pynrrd==1.1.3 \
    scipy==1.18.0
