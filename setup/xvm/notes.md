# UTM image format

```
% ls
Data         config.plist
% ls Data/
A9AE992A-88C1-4E38-907D-0F6168418BAB.qcow2 efi_vars.fd
```

```
cp config.plist config.json
plutil -convert json config.json
```

```
    "Information": {
        "Icon": "orange_x.png",
        "Name": "MagAO-X",
        "UUID": "8C4F4050-0A2F-436F-ADA9-58E213C4C2CF",
        "IconCustom": true
    },
```

orange_x.png goes in the Data subfolder of the .utm bundle

# To get a KDE desktop

```
sudo dnf install epel-release -y
sudo dnf config-manager --set-enabled crb
sudo dnf update -y
sudo dnf groupinstall -y "KDE Plasma Workspaces"
sudo systemctl set-default graphical.target
cat <<'HERE' | sudo tee -a /etc/sddm.conf
[Autologin]
User=xdev
HERE
sudo systemctl enable sddm
sudo systemctl start sddm

sudo -u xdev kwriteconfig5 --file kscreenlockerrc --group Daemon --key Autolock false
sudo -u xdev kwriteconfig5 --file kscreenlockerrc --group Daemon --key LockOnResume false
sudo -u xdev kwriteconfig5 --file powermanagementprofilesrc --group AC --group SuspendSession --key idleTime --delete
sudo -u xdev kwriteconfig5 --file powermanagementprofilesrc --group AC --group SuspendSession --key suspendThenHibernate --delete
sudo -u xdev kwriteconfig5 --file powermanagementprofilesrc --group AC --group SuspendSession --key suspendType --delete
```

# make kickstart

```
% hdiutil create -srcfolder ./kickstart -format UDRO -volname "OEMDRV" -fs "MS-DOS FAT32" ./oemdrv.dmg
```

# adding repos to kickstart


from multipass ex:
```
url ${KS_OS_REPOS} ${KS_PROXY}
url ${KS_BASE_OS_REPOS} ${KS_PROXY}
repo --name="AppStream" ${KS_APPSTREAM_REPOS} ${KS_PROXY}
repo --name="Extras" ${KS_EXTRAS_REPOS} ${KS_PROXY}
```

https://dl.rockylinux.org/pub/rocky/9.3/BaseOS/aarch64/os/

# graphical non interactive

```
graphical --non-interactive
```
https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/performing_an_advanced_rhel_8_installation/kickstart-commands-and-options-reference_installing-rhel-as-an-experienced-user#graphical_kickstart-commands-for-installation-program-configuration-and-flow-control

