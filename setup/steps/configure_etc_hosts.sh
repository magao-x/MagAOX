#!/bin/bash
sudo tee /etc/hosts <<'HERE'
127.0.0.1      localhost localhost.localdomain localhost4 localhost4.localdomain4
::1            localhost localhost.localdomain localhost6 localhost6.localdomain6

# Instrument LAN
# note: 192.168.0.240--192.168.0.254 reserved for DHCP
192.168.0.10   exao1 aoc
192.168.0.12   exao2 rtc
192.168.0.22   kvmrtc
192.168.0.13   exao3 icc
192.168.0.23   kvmicc
192.168.0.14   exao4 coc
192.168.0.140  pdu0
192.168.0.141  pdu1
192.168.0.142  pdu2
192.168.0.143  pdu3
192.168.0.150  acromagdio1
192.168.0.151  acromagdio2
192.168.0.160  picomotorctl0
192.168.0.170  moxadio1
192.168.0.230  fxgenmodwfs

# Point-to-point links
192.168.2.3   icc-from-rtc
192.168.2.2   rtc-from-icc
192.168.20.5  acc000-from-rtc
192.168.20.2  rtc-from-acc000
HERE
