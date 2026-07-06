# hash:sha256:88bce6e4b9f508e39d15b8a7623b94c7f38371d08a9ad9910484d5ed22c8dbd3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    h5py==3.16.0 \
    nibabel==5.4.2 \
    numpy==2.5.0 \
    pynrrd==1.1.3 \
    scipy==1.18.0
