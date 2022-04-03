FROM centos:7
RUN yum update -y && yum install -y which passwd sudo
RUN adduser -G wheel xsup
RUN echo -e "[user]\ndefault=$myUsername\n[interop]appendWindowsPath=false" > /etc/wsl.conf
RUN echo -e "extremeAO!\nextremeAO!\n" | passwd xsup
RUN env
USER xsup