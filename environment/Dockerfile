# hash:sha256:fdc0701050750eb535faa0ada37ad8f8209b44c90e2bf94f381f2eb58c689625
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    h5py \
    numpy \
    pynrrd \
    scipy
