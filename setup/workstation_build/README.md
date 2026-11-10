## Example of a manual MagAOX installation on an existing system:

### Role and sources

* MagAOX role:
  * **magaox01**
  * Initial role for provisioning:   **workstation**
* MagAOX source:
  * Repository => **https://github.com/drbitboy/MagAOX.git**
  * Branch => **resurrector-20250428-resuctrl-xsup-permissions**
* MagAOX configuration source:
  * Repository => **https://github.com/drbitboy/magao-x-config.git**
  * Branch => **resurrector-indi-compression**

### Initial setup

* As sudo-capable user:
```
sudo apt update
sudo apt upgrade -y
sudo reboot
```

### Get sources
* As sudo-capable user:
```
cd
mkdir githubalt

git clone \
  --depth=1 \
  -b resurrector-20250428-resuctrl-xsup-pemissions \
  https://github.com/drbitboy/MagAOX.git \
  githubalt/MagAOX

git clone \
  --depth=1 \
  -b resurrector-indi-compression \
  https://github.com/drbitboy/magao-x-config.git \
  githubalt/config
```

### Provision MagAOX system
* As sudo-capable user:
```
cd
cd githubalt/MagAOX/setup
MAGAOX_ROLE=workstation ./pre_provision.sh
source /etc/profile.d/magaox_role.sh
bash -l ./provision.sh"
```

### Alternative to compile OpenBLAS for TARGET=HASWELL:
```
OPENBLAS_MAKE_DEFINES=TARGET=HASWELL bash -l ./provision.sh"
```

### Change system role to **magaox01**
* As sudo-capable user:
```
echo MAGAOX_ROLE=magaox01 | sudo tee /etc/profile.d/magaox_role.sh
source /etc/profile.d/magaox_role.sh
```
## End of manual installation
___
___
## Single-command example of automated creation of two new Multipass VMs, plus installation and provisioning of MagAOX on same:

### Create and provision multipass VM named magaox01, plus second cloned VM named magaox02

* N.B. Roles will change to magaox01 and magaox02 after provisioning
* N.B. The VM hostnames specified below (**-v=magaox01** and **-2=magaox02**) are mandatory when using the config branch specified below (**resurrector-indi-compression**)
```
./magaox_multipass_setup.sh \
  -v=magaox01 -2=magaox02 \
  -M=drbitboy/MagAOX,MagAOX,resurrector-20250428-resuctrl-xsup-permissions \
  -r=drbitboy/magao-x-config,config,resurrector-indi-compression
```
## End of automated installation
___
___
## Running MagAOX system, assuming either installation procedure above is complete

### Register the proclist's apps as systemd services and create the INDI driver/server system directories (/opt/MagAOX/sys/.../); su to non-privileged user **xsup** to run MagAOX system

* As sudo-capable user on both/either magaox01 and magaox02:
```
sudo xctrl register   ### Creation, provisioning are complete after this step
sudo su - xsup
```

### Start INDI server and all INDI drivers under sentinel (systemd) control
### Update maths_1 and maths_2 INDI drivers, view results

* As user xsup:
```
xctrl startup
xctrl status

setINDI maths_1.val.value=1.1   ### on magaox01
setINDI maths_2.val.value=1.2   ### on magaox02

getINDI -t 1
```

### Stop an INDI driver

```
xctrl stop maths_1              ### on magaox01
xctrl status                    ### all apps
```

### Start an INDI driver

```
xctrl start maths_1             ### on magaox01
xctrl status                    ### all apps
```

### Stop everything

```
xctrl shutdown --all
```
