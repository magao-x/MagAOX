# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "centos/7"
  config.vm.network "private_network", ip: "172.16.200.2"  # needed for NFS
  config.vm.provider :virtualbox do |vb|
    vb.memory = 2048
  end
  config.vm.synced_folder ".", "/vagrant", type: "nfs"
  config.vm.provision "shell", path: "setup/provision.sh"
  config.ssh.forward_agent = true
  config.ssh.forward_x11 = true
end
