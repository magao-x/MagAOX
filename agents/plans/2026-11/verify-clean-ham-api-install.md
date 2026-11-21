# Task Description
Confirm that Hamamatsu API and SDK install on a new machine will not affect the existing libraries.

## Problem Statement
Please examine ``/opt/hamamatsu_new/install_firebird_el9.sh`` and verify that an install on a new machine (using a copy of the ``/opt/hamamatsu_new/`` directory) will not modify any existing libraries that are unrelated to the Hamamatsu API and SDK.
## Discussion
- the purpose of this script is to install the Hamamatsu API and firebird drivers on a RHEL9/Rocky 9 machine that contains the MagAO-X (``/opt/MagAOX``) source code
- this script is meant to replace an existing, vendor-provided script (``install.sh``) that performs an in-place upgrade of gcc and other lower-level libraries
- this script should NOT be performing any in-place upgrades of the lower-level libraries on the machine
- the hamamatsu api and sdk are located within ``/opt/hamamatsu_new/``
- api and sdk documentation can be found in the corresponding ``docs`` file within that directory
## Requirements
1. Executing the install_firebird_el9 script on a new machine running RHEL9 should not affect the pre-existing lower-level libraries that are unrelated to either the Hirebird driver, the Hamamatsu (ie. gcc) on said machine
2. The ``install_firebird_el9.sh`` script should successfully run on a new RHEL9 machine after copying the ``/opt/hamamatsu_new/`` directory over from this machine
## Tests and Metrics
1. Test that an install on a clean RHEL9 machine with a copy of ``/opt/hamamatsu_new/`` runs without errors
2. Test that running the script on a clean RHEL9 machine does not damage the libraries/apps in the existing (``/opt/MagAOX``) source code

## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan
<!-- This section will be filled out by the agent -->

## Task Summary
<!-- The agent should summarize the task as they understand it -->
Audit `/opt/hamamatsu_new/hamamatsu_api/install_firebird_el9.sh` (the plan text says `/opt/hamamatsu_new/install_firebird_el9.sh`, but the script actually lives one level down in `hamamatsu_api/`). The goal is to confirm that running it on a fresh RHEL/Rocky 9 machine, from a copy of `/opt/hamamatsu_new/`:
1. installs only the DCAM runtime, the Active Silicon FireBird SDK, and the FireBird kernel modules, and
2. does not install, upgrade, or remove any unrelated system packages or libraries (gcc, kernel-headers, libusb, etc.) or anything under `/opt/MagAOX`.

Then define a test procedure on a clean machine that demonstrates both.

### Static analysis performed (read-only, on exao0, 2026-11-13)
Every script reachable from the wrapper was read, the vendor RPM was extracted into a scratch directory and inspected (no install), and `/opt/hamamatsu_new` was diffed against the original vendor tree in `/opt/hamamatsu/DCAM-API_Lite_for_Linux_v24.12.6898`.

**Origin of the gcc upgrade in the vendor flow.** The vendor top-level `api/install.sh` runs `sudo yum install -y kernel-devel kernel-headers gcc` (and `apt install ... gcc-12` on Debian) unconditionally whenever `fbd` is selected. With no version pin, this upgrades gcc and kernel-headers to the latest available. On exao0, `kernel-headers` is `el9_8` while the running kernel is `el9_6`, which is consistent with this behaviour (or a general `dnf update`). In `/opt/hamamatsu_new` the block is commented out, and that is the *only* difference from the vendor tree apart from the three new files (`install_firebird_el9.sh`, `patches/`, `README-firebird-el9.md`). The usb branch also contains `yum install -y libusb-devel`, but it is unreachable with `fbd` only.

**Call graph of `install_firebird_el9.sh`.** The wrapper never calls the vendor top-level `api/install.sh`, so even the commented-out yum block is bypassed.

| Step | Writes to | Unrelated-system impact |
|---|---|---|
| Precheck: arch, kernel build tree, `gcc`, `patch`, Secure Boot | nothing | none. It **fails** rather than installing missing prerequisites |
| `api/runtime/install.sh fbd` | `/usr/local/hamamatsu_dcam/**`, symlink `/usr/local/lib/libdcamapi.so`, `/etc/ld.so.conf.d/hamamatsu_dcam.conf`, runs `ldconfig` | `hamamatsu_dcam.conf` adds `/usr/local/lib` (already listed in `local.conf`) plus two Hamamatsu-only directories. No soname collisions (checked with `ldconfig -p`) |
| `api/driver/firebird/install.sh --maincall` (only if RPM is absent) | `tar -x` into the copied `api/driver/firebird/` (re-extracts byte-identical RPM/DEB; md5 checked), `rpm -ivh as-fbd-linux-ham-3.2.2-1` → `/usr/local/activesilicon/**` only, `/etc/ld.so.conf.d/aslphxapi.conf`, `ldconfig` | RPM `Requires:` only `/bin/sh`, so **no dependency pulls and no package upgrades**. Every payload file is under `/usr/local/activesilicon` |
| ↳ RPM `%post` scriptlet | runs `make -C .../mdadrv/linux` (`all` = uninstall, clean, build, install) on the **unpatched** source | touches only ASL files: `/lib/modules/$(uname -r)/kernel/drivers/misc/asl*.ko`, `/etc/udev/rules.d/89-activesilicon.rules`, `/etc/modules-load.d/activesilicon.conf`, `/lib/firmware/activesilicon` (rm), `depmod -a`. On EL9 the unpatched build is expected to fail, so `%post` fails non-fatally (rpm warns, package stays installed, the vendor script still exits 0) |
| Patch `active-silicon-firebird-el9.patch` | 7 `.c` files under `/usr/local/activesilicon/source/driver/mdadrv/linux/`, stamp file | none. The dry-run is gated: it aborts if the patch does not apply cleanly |
| `make clean/build/install`, `depmod -a`, `modprobe asl*` | same ASL module/udev/modules-load paths as above | `depmod -a` regenerates `modules.dep*` for the running kernel (standard, additive) |

**Conclusion of static analysis.** No step installs, upgrades, or removes an RPM other than `as-fbd-linux-ham`. No step writes to `/opt/MagAOX`, `/usr/lib64`, `/usr/include`, or the compiler toolchain. Nothing outside the Hamamatsu/Active Silicon paths is replaced. The only system-wide side effects are shared-cache regenerations (`ldconfig`, `depmod -a`, `udevadm control --reload-rules`) and new config drop-ins (`hamamatsu_dcam.conf`, `aslphxapi.conf`, `89-activesilicon.rules`, `activesilicon.conf`). Requirement 1 is satisfied by inspection. Requirement 2 still needs the clean-machine test below.

## Key Assumptions
<!-- The agent should list any assumptions they have made -->
- The whole `/opt/hamamatsu_new/` directory is copied, not just `hamamatsu_api_firebird_el9_bundle.tar.gz`. The bundle omits `hamamatsu_sdk/`, which holds the `dcamapi4.h`/`dcamprop.h` headers that `apps/orcaCtrl/Makefile` includes from `/opt/hamamatsu_new/hamamatsu_sdk/dcamsdk4/inc/`.
- The copy is writable by the invoking (non-root) user, because the vendor FireBird script runs `tar -x` into `api/driver/firebird/` without sudo.
- The target is x86_64 Rocky/RHEL 9 with Secure Boot disabled and `gcc`, `make`, `patch`, and `kernel-devel-$(uname -r)` already present (the wrapper checks for these but never installs them).
- The target does not have `as-fbd-linux-ham` installed. If it does, the RPM step is skipped and the existing source tree is reused.
- No clean Rocky 9 machine is available (user, 2026-11-13). A Rocky 9 container stands in for a clean machine (T0). The real new machine is verified with before/after snapshots during its actual deployment (T1–T4).
- `/opt/MagAOX` is already set up on the new machine before the Hamamatsu install (user, 2026-11-13).
- The new machine gets the exact-version `kernel-devel-$(uname -r)` (user, 2026-11-13). Because `kernel-devel` `Requires: gcc` unversioned, this cannot upgrade an existing gcc.
- All tests that touch a real machine are run by the user (AGENTS.md rule 0). The agent only supplies scripts and interprets the results.

## Requirements
<!-- The agent should list the requirements to which they are planning -->
- R1: The install adds or changes only Hamamatsu/Active Silicon files, the new `as-fbd-linux-ham` RPM, ASL kernel modules, and the config drop-ins listed above. The set of installed RPMs and their versions is otherwise identical before and after.
- R2: `install_firebird_el9.sh` exits 0 on a clean RHEL9 machine from a copy of `/opt/hamamatsu_new/`, and the `aslenum/asldma/aslcxp/aslser` modules end up loaded.
- R3: Libraries and apps from `/opt/MagAOX` (and the shared libs they link: mxlib, cfitsio, fftw, ImageStreamIO, CUDA, etc.) are unchanged and still resolve and run after the install.

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->
Resolved (user answers, 2026-11-13):
1. **Test machine.** No clean Rocky 9 machine is available, so the tests were restructured into T0 (container) plus T1–T4 (on the new machine during deployment).
2. **MagAO-X provisioning.** `/opt/MagAOX` will already be set up, so T3 applies.
3. **`kernel-devel`.** The exact-version `kernel-devel-$(uname -r)` is installed.
4. **Deliverables.** The snapshot scripts stay as an appendix to this plan; nothing is committed to the repo apart from this plan.
5. **Optional hardening.** The fixes are described below under "Optional Script Fixes", pending user approval of each.

Open:
6. **T0 container test.** (Answered: approved for exao0, but blocked and then skipped by user decision; see Progress Log.) Running T0 on exao0 needs `podman` (present) to pull `docker.io/rockylinux/rockylinux:9` and to `dnf install` packages *inside the container only*. The host is not modified. Should the agent run T0 itself on exao0, or should the user run it on a workstation?
7. **Which optional fixes (A–E below) to apply?** (Answered: A, B, C.)

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->
Without a clean machine, verification is split into two parts:
- **T0** runs in a disposable Rocky 9 container. It exercises everything except loading the modules, on a truly clean package set.
- **T1–T4** run on the real new machine as its deployment. They are bracketed by `snapshot.sh` before and after, and compared with `compare.sh` (Appendix).

**T0 – Clean-environment package audit (R1, most of R2).** Run in a throwaway `rockylinux:9` container with a read-write copy of `/opt/hamamatsu_new` mounted in.
1. Install the prerequisites: `gcc make patch kernel-devel-$(uname -r)` (host kernel version, from the Rocky vault repo if it is no longer in BaseOS), `kmod`, `findutils`, `procps-ng`, `pciutils`, `file`.
2. Take a snapshot with `snapshot.sh /snap/before` (MagAO-X sections are skipped automatically when `/opt/MagAOX` is absent).
3. Install a shim `sudo` (the container already runs as root) that runs its arguments but turns `modprobe`, `depmod`, `udevadm` and `insmod` into logged no-ops. Point `mokutil` at "not found".
4. Run `install_firebird_el9.sh`. The `lsmod` check at the end is expected to fail in a container (no modules can be loaded), so check the log for everything up to that point.
5. Take a snapshot with `snapshot.sh /snap/after` and run `compare.sh`.

Pass criteria:
- the RPM diff is exactly `+as-fbd-linux-ham-3.2.2-1.x86_64`
- gcc and kernel-headers are unchanged
- there are no new `rpm -Va` entries
- new or changed files appear only under `/usr/local/hamamatsu_dcam`, `/usr/local/activesilicon`, `/usr/local/lib/libdcamapi.so`, `/etc/ld.so.conf.d/{hamamatsu_dcam,aslphxapi}.conf`, `/etc/udev/rules.d/89-activesilicon.rules`, `/etc/modules-load.d/activesilicon.conf`, and `/lib/modules/*/kernel/drivers/misc/asl*.ko`
- all four `.ko` files build from the patched source

This proves that the script itself causes no unrelated package changes. It also checks that the patch and build work against the target kernel-devel.

**T1 – Install succeeds on the new machine (R2).** Run `snapshot.sh ~/ham-snap/before`, then `./install_firebird_el9.sh 2>&1 | tee ~/ham-snap/install.log; echo "exit=${PIPESTATUS[0]}" >> ~/ham-snap/install.log`. Pass if the exit code is 0, `lsmod` shows `aslenum asldma aslcxp aslser`, and `rpm -q as-fbd-linux-ham` succeeds. A `%post` scriptlet warning is expected (see Follow-up).

**T2 – No unrelated changes on the new machine (R1).** Run `snapshot.sh ~/ham-snap/after` and `compare.sh ~/ham-snap/before ~/ham-snap/after`. The pass criteria are the same as T0.

**T3 – MagAO-X untouched and functional (R3).** In the `compare.sh` output:
- `magaox-files` and `magaox-bin.sha256` show no differences
- `ldd` shows no differences and no `not found`

Then rebuild one existing app (e.g. `make -C /opt/MagAOX/source/MagAOX/apps/<app>`) and `orcaCtrl`, confirming both link. Optionally run the existing unit tests.

**T4 – Idempotence.** Re-run the script and take a third snapshot. Pass if it exits 0, takes the "already installed" and "already applied" paths, and `compare.sh after after2` differs only in the timestamps of the rebuilt `asl*.ko` files.

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->
1. Record the static-analysis findings above (done).
2. Write `snapshot.sh` and `compare.sh` as an appendix to this plan (done, see Appendix).
3. ~~Run T0 in a container.~~ Skipped by user decision on 2026-11-13, after rootless podman was blocked (see Progress Log). R1 and R2 are verified by T1–T4 on exao3 instead.
4. (Done for A, B, C.) Apply whichever optional script fixes the user approves (Q7) to `/opt/hamamatsu_new/hamamatsu_api/install_firebird_el9.sh`, then regenerate `hamamatsu_api_firebird_el9_bundle.tar.gz`. Re-run T0 after any change.
5. During deployment of the new machine, the user runs T1–T4 and returns the snapshot directories and `install.log`. The agent analyses them and records the results here.
6. Commit this plan on `joshua-liberman/hamsci-app-cpp` (or a new `joshua-liberman/<feature>` branch if preferred).

### Optional Script Fixes (pending user approval)
All of these are edits to `install_firebird_el9.sh` only. None of them changes what gets installed. The vendor files remain untouched.

**A. Check for `make` before starting.** The README lists `make` as a prerequisite, but the script never checks for it. Without `make` today, the script installs the DCAM runtime and the RPM first and only then fails at `make clean`, leaving a half-finished install.
```bash
command -v make >/dev/null || fail "make is required to build the FireBird modules."
```

**B. Check the driver directory is writable before starting.** The vendor `api/driver/firebird/install.sh` runs `tar -zxvf ... -C $MAINDIR` *without* sudo. If the copied `/opt/hamamatsu_new` is root-owned (e.g. copied with `sudo cp -r`), `tar` fails. The vendor script ignores the error and still runs `rpm -ivh` on the RPM already present in `x86_64/`. The install happens to succeed anyway, but only by accident. A precheck makes the requirement explicit and fails before anything is changed:
```bash
[[ -w "$(dirname "$DRIVER_INSTALLER")" ]] || fail "$(dirname "$DRIVER_INSTALLER") must be writable by $(id -un) (the vendor installer extracts into it)."
```

**C. Verify that the RPM actually installed.** The vendor driver installer always ends in `exit 0`, even if `rpm -ivh` fails (e.g. a file conflict or a corrupt RPM). Today such a failure only shows up indirectly as "FireBird source was not installed at ...". An explicit check gives the real cause:
```bash
    bash "$DRIVER_INSTALLER" --maincall
    rpm -q as-fbd-linux-ham >/dev/null 2>&1 || fail "Active Silicon RPM installation failed; see rpm output above."
```

**D. Refuse to run if the vendor gcc/yum block has been re-enabled (defensive).** The script never calls `api/install.sh`, so this guards only against a future edit that starts calling it, or against someone running the vendor script by mistake. Since it protects against something the script does not do, it is low value; it is listed for completeness.
```bash
grep -qE '^[[:space:]]*sudo (yum|dnf|apt)' "$ROOT_DIR/api/install.sh" && fail "Vendor api/install.sh has package-manager installs enabled; do not use it."
```

**E. Skip the RPM's unpatched `%post` build (behaviour change, discuss).** Install the RPM directly with `sudo rpm -ivh --noscripts`, then copy `aslphxapi.conf` and run `ldconfig` ourselves, instead of calling the vendor driver installer. This avoids the known-failing root build of the unpatched source and the scary scriptlet warning, and removes the vendor `tar`/`rm` into the copied directory (making B unnecessary).

Downsides:
- It no longer follows the vendor flow.
- It skips `%post`'s `make uninstall` of any older ASL modules, though the wrapper's own `make install` overwrites them anyway.
- `rpm -e` later still runs `%preun`.

The agent recommends A, B and C. D is optional. E is only worth doing if the `%post` warning is a problem in practice.

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->
- **`%post` on unpatched source.** The RPM's `%post` runs `make all` (as root) before the EL9 patch is applied. It is expected to fail at the build step and leave no modules installed. If a future kernel lets the unpatched build *succeed*, the unpatched modules are installed briefly, then overwritten by the wrapper's patched build. Either way, only ASL files are affected.
- **`%preun` on removal.** `rpm -e as-fbd-linux-ham` runs `make uninstall`, which removes the ASL modules, the udev rule, and `/lib/firmware/activesilicon`. Note that the vendor Makefile's uninstall misspells `activesilcon.conf`, so `/etc/modules-load.d/activesilicon.conf` is left behind.
- **Missing udev helper.** The udev rule runs `/usr/local/activesilicon/bin64/udev_start`, which the RPM does not ship. This is a harmless failed `RUN` on board enumeration, but it will appear in the journal.
- **Shared-cache regeneration.** `ldconfig` regenerates the cache and may create missing soname symlinks in *any* configured directory. On a freshly provisioned host this is the same thing the MagAO-X setup already does, but T2's `find -newer` will reveal it if it happens.
- **Kernel updates.** Kernel updates require rebuilding the modules (re-run the wrapper or the `make` steps). `PATCH_STAMP` prevents double patching.
- **Secure Boot.** Hosts with Secure Boot enabled are rejected. See `doc/add_sign.txt` for signing.
- **Bundle tarball.** The tarball does not include `hamamatsu_sdk/`. Copying only the tarball leaves `orcaCtrl` without headers.
- **Plan text path.** The plan text refers to `/opt/hamamatsu_new/install_firebird_el9.sh`. The actual path is `/opt/hamamatsu_new/hamamatsu_api/install_firebird_el9.sh`.

## Progress Log
**2026-11-13**
- The user approved fixes A, B and C (D and E were not approved), and approved running T0 on exao0.
- Applied A, B and C to `/opt/hamamatsu_new/hamamatsu_api/install_firebird_el9.sh`. The original is backed up in the agent scratchpad. Diff:
  ```diff
  +command -v make >/dev/null || fail "make is required to build the FireBird modules."
   command -v patch >/dev/null || fail "patch is required to apply the EL9 compatibility patch."
  +[[ -w "$(dirname "$DRIVER_INSTALLER")" ]] || fail "$(dirname "$DRIVER_INSTALLER") must be writable by $(id -un) (the vendor installer extracts into it)."
  ...
       bash "$DRIVER_INSTALLER" --maincall
  +    rpm -q as-fbd-linux-ham >/dev/null 2>&1 || fail "Active Silicon RPM installation failed; see rpm output above."
  ```
- Added the writable-directory prerequisite to `README-firebird-el9.md`.
- Regenerated `hamamatsu_api_firebird_el9_bundle.tar.gz`. Before regenerating, the file list was confirmed identical and only the two edited files differed.
- Created `~/hamamatsu_new.zip` (123 MB, 488 entries) on exao0 for copying to exao3. `unzip -t` passes and execute permissions are preserved. The existing root-owned `/opt/hamamatsu_new.zip` (Sep 11) was left untouched and predates these fixes.
- **T0 blocked.** Rootless podman fails with `newgidmap: gid range [65537-131073) -> [68157440-68222976) not allowed`. `getsubids -g jliberman` returns the `/etc/subgid` entry `1040:68157440:65536`, which is keyed by the user's *GID*, and `newgidmap` refuses it. Root podman needs an interactive sudo password. Changing `/etc/subgid` is a system-admin decision, so T0 has not been run.
- **T0 skipped** by user decision. Verification relies on T1–T4 with the appendix snapshot scripts during the exao3 install.
- **Deployment note for exao3 (fix B).** If the ZIP is extracted with `sudo unzip` into `/opt`, run `sudo chown -R $USER: /opt/hamamatsu_new` before `install_firebird_el9.sh`. Otherwise the new writability precheck stops the install before anything is changed.

## Appendix: Snapshot Scripts
Both scripts are read-only, apart from writing to the output directory. `snapshot.sh` needs `sudo` only for `rpm -Va`.

`snapshot.sh`:
```bash
#!/usr/bin/env bash
# Capture system state relevant to the Hamamatsu/FireBird install. Usage: snapshot.sh <outdir>
set -uo pipefail
out="${1:?usage: snapshot.sh <outdir>}"
mkdir -p "$out"

rpm -qa --qf '%{NAME}-%{VERSION}-%{RELEASE}.%{ARCH}\n' | sort > "$out/rpms.txt"
sudo rpm -Va 2>&1 | sort > "$out/rpm-Va.txt"
ldconfig -p | sort > "$out/ldconfig.txt"
{ gcc --version | head -1; rpm -q gcc kernel-headers kernel-devel; uname -r; } > "$out/toolchain.txt" 2>&1
lsmod | awk 'NR>1 {print $1}' | sort > "$out/lsmod.txt"

# path, type, size, mtime, link target for system locations (catches non-RPM writes)
find /usr/local /usr/lib64 /usr/include /etc/ld.so.conf.d /etc/udev/rules.d /etc/modules-load.d /etc/modprobe.d \
     /lib/modules/"$(uname -r)"/kernel/drivers/misc -xdev \( -type f -o -type l \) \
     -printf '%p %y %s %T@ %l\n' 2>/dev/null | sort > "$out/sysfiles.txt"

if [[ -d /opt/MagAOX ]]; then
    # Skip large runtime data dirs (logs, telem, rawimages, cacao*); record metadata for code/config, hash the binaries.
    (cd /opt/MagAOX && find bin source vendor drivers config config-* calib -xdev \( -type f -o -type l \) \
         -printf '%p %y %s %T@ %l\n' 2>/dev/null | sort) > "$out/magaox-files.txt"
    (cd /opt/MagAOX/bin && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$out/magaox-bin.sha256"

    for b in /opt/MagAOX/bin/*; do
        if [[ -f "$b" && -x "$b" ]] && file -b "$b" | grep -q ELF; then
            echo "== $b"
            ldd "$b" 2>&1 | sed -E 's/ \(0x[0-9a-f]+\)$//'
        fi
    done > "$out/ldd.txt"
fi

echo "snapshot written to $out"
```

`compare.sh`:
```bash
#!/usr/bin/env bash
# Diff two snapshots from snapshot.sh. Usage: compare.sh <before> <after>
set -uo pipefail
a="${1:?usage: compare.sh <before> <after>}"
b="${2:?usage: compare.sh <before> <after>}"
expected='/usr/local/(hamamatsu_dcam|activesilicon)/|/usr/local/lib/libdcamapi\.so|/etc/ld\.so\.conf\.d/(hamamatsu_dcam|aslphxapi)\.conf|89-activesilicon\.rules|modules-load\.d/activesilicon\.conf|/drivers/misc/asl(enum|dma|cxp|ser)\.ko'

for f in rpms.txt toolchain.txt rpm-Va.txt lsmod.txt magaox-files.txt magaox-bin.sha256 ldd.txt; do
    [[ -f "$a/$f" ]] || continue
    echo "##### $f"
    diff "$a/$f" "$b/$f" && echo "(identical)"
done

echo "##### sysfiles.txt: changes OUTSIDE expected Hamamatsu/ASL paths"
diff "$a/sysfiles.txt" "$b/sysfiles.txt" | grep -E '^[<>]' | grep -vE "$expected" || echo "(none)"

echo "##### sysfiles.txt: changes INSIDE expected paths (count)"
diff "$a/sysfiles.txt" "$b/sysfiles.txt" | grep -E '^[<>]' | grep -cE "$expected"

echo "##### ldd 'not found' after install"
grep -n 'not found' "$b/ldd.txt" 2>/dev/null || echo "(none)"
```
