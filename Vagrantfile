# -*- mode: ruby -*-
# vi: set ft=ruby :
$script = <<-SCRIPT
set -exuo pipefail
/vagrant/setup/provision_as_root.sh
/vagrant/setup/makeDirs.sh --dev
usermod -G magaox vagrant
/bin/sudo -u vagrant bash /vagrant/setup/provision_as_user.sh
echo "Finished!"
SCRIPT

Vagrant.configure("2") do |config|
  config.vm.box = "centos/7"
  config.vm.network "private_network", ip: "172.16.200.2"  # needed for NFS
  config.vm.synced_folder ".", "/vagrant", type: "nfs"
  config.vm.provision "shell", inline: $script
end
