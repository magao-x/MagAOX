# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.provider :virtualbox do |vb|
    vb.memory = 2048
  end
  config.vm.provision "shell", path: "setup/provision.sh"
  config.ssh.forward_agent = true
  config.ssh.forward_x11 = true
  config.vm.network "forwarded_port", guest: 7624, host: 7624
  config.vm.network "forwarded_port", guest: 80, host: 8080
  config.vm.network "forwarded_port", guest: 9999, host: 9999

  config.vm.define "ICC",  autostart: false do |icc|
    icc.vm.box = "centos/7"
    icc.vm.synced_folder ".", "/vagrant", type: "nfs"
    icc.vm.network "private_network", ip: "172.16.200.2"  # needed for NFS
  end

  config.vm.define "AOC", primary: true do |aoc|
    aoc.vm.box = "generic/ubuntu1804"
    aoc.vm.synced_folder ".", "/vagrant"
  end
end
