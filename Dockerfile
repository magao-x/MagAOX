ARG BASE_IMAGE=ghcr.io/magao-x/magao-x-setup:cli
FROM ${BASE_IMAGE}
ARG MAGAOX_ROLE=container
ENV MAGAOX_ROLE=${MAGAOX_ROLE}
WORKDIR /opt/MagAOX/source/magao-x-setup
USER root
RUN bash -lx steps/install_MagAOX.sh
USER xsup
