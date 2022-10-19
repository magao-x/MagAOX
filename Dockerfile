FROM ubuntu:jammy-20220130
RUN yum update -y && yum install -y which passwd sudo
ADD . /opt/MagAOX/source/MagAOX
WORKDIR /opt/MagAOX/source/MagAOX/setup
RUN bash setup_users_and_groups.sh
RUN bash provision.sh
USER xsup