#!/bin/bash
# adapted from
# https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-yum-repo
set -exuo pipefail
yum install -y yum-utils
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
yum install -y intel-mkl-64bit
