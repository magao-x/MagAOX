FROM ubuntu:jammy-20220130
RUN mkdir /vendor
# Note this ADD automatically uncompresses the sdk:
ADD sdk_linux64_22.02.17.tgz /vendor
WORKDIR /vendor/sdk_linux64_22.02.17
RUN apt update
RUN apt-get upgrade -y
RUN apt-get install -y curl libusb-1.0-0-dev usbutils build-essential
RUN bash install.sh