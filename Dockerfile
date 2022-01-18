# FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Install needed apt packages
COPY apt.txt apt.txt
RUN apt-get -qq update && xargs -a apt.txt apt-get -qq install -y --no-install-recommends \
    && rm -rf /var/cache/*

# Create user home directory
ENV HOME_DIR /home/

WORKDIR ${HOME_DIR}

# Copy user files
COPY . ${HOME_DIR}

RUN pip install -U --no-cache-dir -e .
