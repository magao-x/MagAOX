This document describes how to install the realtime linux kernel and the NVIDIA GPU driver.

## 1 Gettting and installing kernel-rt

See http://linuxsoft.cern.ch/cern/centos/7/rt/x86_64/repoview/RT.group.html

Perform the following steps to install the Cern RT kernel repo and install the RT kernel itself:
```
wget http://linuxsoft.cern.ch/cern/centos/7/rt/CentOS-RT.repo
sudo cp CentOS-RT.repo /etc/yum.repos.d/
wget linuxsoft.cern.ch/cern/slc5X/i386/RPM-GPG-KEYs/RPM-GPG-KEY-cern
sudo cp RPM-GPG-KEY-cern /etc/pki/rpm-gpg/
sudo yum groupinstall RT
sudo yum kernel-rt-devel
```
Now reboot.  It should fail to start X if you have a graphics environment going.  That's fine, you'd need to kill it anyway to install the driver.

Once at a prompt, type `uname -a` and verify that the PREEMPT kernel-rt is running.

## 2 Installing the NVIDIA driver & CUDA

With hacks to make it build against the rt kernel.

These steps inspired by: https://gitlab.manjaro.org/packages/community/realtime-kernels/linux416-rt-extramodules/blob/master/nvidia/PKGBUILD but with additions.

### 2.1 Shutdown X
You will need to switch to `multi-user.target` to do the install.  
- Temporary change:
From command line:
```
sudo systemctl isolate multi-user.target
```
Or from grub boot menu, add `systemd.unit=multi-user.target` at end of the linux16 line.
- Permanent change:
I find it easier to do this with the default changed. In case anything goes wrong it is easier to reboot:
```
sudo systemctl set-default multi-user.target
```

### 2.2 Preparing The Driver
Get the CUDA `.run` file from the NVIDIA website, and make it executable (`chmod +x cuda_10.0.130_410.48_linux.run`).  Next, unpack it:
```
./cuda_10.0.130_410.48_linux.run --extract=$(pwd)/cuda_10.0.130_410.48_linux
```
Now cd to that directory:
```
cd cuda_10.0.130_410.48_linux
```
And now unpack the driver run file:
```
./NVIDIA-Linux-x86_64-410.48.run -x
```
and cd to that directory:
```
cd NVIDIA-Linux-x86_64-410.48/
```
And now run the `nvrthack_410.48.sh` script, which you can obtain from Jared, while in that directory.



### 2.3 Build and Install
Next, switch to root.  Then type:
```
export IGNORE_PREEMPT_RT_PRESENCE=1
```
which will cause the build configuration to accept that you have the PREEMPT kernel. Then:
```
./nvidia-installer
```

Now reboot.  After it completes, in a terminal type
```
dmesg | grep NV
```
which should produce two lines something like
```
[    5.466790] NVRM: loading NVIDIA UNIX x86_64 Kernel Module  410.48  Thu Sep  6 06:36:33 CDT 2018 (using threaded interrupts)
[    5.569665] nvidia-modeset: Loading NVIDIA Kernel Mode Setting Driver for UNIX platforms  410.48  Thu Sep  6 06:18:22 CDT 2018
```
in addition to several others.  This indicates that the driver is being loaded and works.  You can also run `nvidia-smi` to verify that it is working with the installed GPU.

### 2.4 Restoring X
To test booting into graphical mode, if you changed the default target you can do:
- Temporarily, from command line:
```
sudo systemctl isolate graphical.target
```
or from grub boot menu, add `systemd.unit=graphical.target` at end of the linux16 line.
- To change the default:
```
/etc/systemd/system/default.target to /usr/lib/systemd/system/graphical.target
```

### 2.5 Installing CUDA
Now we can install the rest of CUDA, if not already done.  For this, run the original (packed) install file:
```
./cuda_10.0.130_410.48_linux.run
```
Accept the EULA (of course reading it carefully first). When it asks if you want to install the driver SAY NO!!!
```
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 410.48?
(y)es/(n)o/(q)uit: n
```

After it's installed, created a file `/etc/ld.so.conf.d/cuda.conf` with contents:
```
/etc/ld.so.conf.d/cuda.conf
```
and run `ldconfig`.  Then add `/usr/local/cuda-10.0/bin` to your path.  This can be done for a single user in `~/.bashrc` with the line:
```
PATH=$PATH:/usr/local/cuda-10.0/bin
```
or for all users by creating a file `/etc/profile.d/cuda.sh` with the same contents.

### 2.6 Build The Samples
In whatever directory you installed the source (chosen during CUDA install), cd to `NVIDIA_CUDA-10.0_Samples` and type make.  It should complete without errors.

Then cd to `0_Simple/matrixMulCUBLAS/` and run `./matrixMulCUBLAS` which should give a result like:
```
[Matrix Multiply CUBLAS] - Starting...
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

MatrixA(640,480), MatrixB(480,320), MatrixC(640,320)
Computing result using CUBLAS...done.
Performance= 3643.26 GFlop/s, Time= 0.054 msec, Size= 196608000 Ops
Computing result using host CPU...done.
Comparing CUBLAS Matrix Multiply with CPU results: PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```
