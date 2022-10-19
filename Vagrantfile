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
  config.vm.network "forwarded_port", guest: 8000, host: 8000
  config.vm.network "forwarded_port", guest: 9999, host: 9900

  config.vm.box = "generic/centos7"
  config.vm.synced_folder ".", "/vagrant"

end
