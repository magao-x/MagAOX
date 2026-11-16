#!/usr/bin/env python3
"""List the semaphore readers of a milk stream in post order, with each thread's
scheduling class, RT priority, allowed CPUs, last CPU, and cpuset.

Read-only: opens the stream metadata and reads /proc.  Does not register a semaphore.

usage: semreaders.py aol1_imWFS2 [aol1_imWFS2_cbuff ...]
"""
import os
import sys

from ImageStreamIOWrap import Image

POL = {0: "OTHER", 1: "FIFO", 2: "RR", 3: "BATCH", 5: "IDLE", 6: "DEADLINE"}


def rd(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "?"


def threads(pid):
    out = []
    for tid in sorted(os.listdir(f"/proc/{pid}/task"), key=int):
        stat = rd(f"/proc/{pid}/task/{tid}/stat")
        f = stat[stat.rfind(")") + 2:].split()
        # fields after comm: f[36] = processor, f[37] = rt_priority, f[38] = policy
        st = {k: v.strip() for k, v in (l.split(":", 1) for l in rd(f"/proc/{pid}/task/{tid}/status").splitlines() if ":" in l)}
        out.append((tid, rd(f"/proc/{pid}/task/{tid}/comm"), POL.get(int(f[38]), f[38]), f[37], st.get("Cpus_allowed_list", "?"), f[36],
                    st.get("voluntary_ctxt_switches", "?"), st.get("nonvoluntary_ctxt_switches", "?")))
    return out


def show_pid(label, pid):
    if pid <= 0 or not os.path.exists(f"/proc/{pid}"):
        print(f"  {label}: pid {pid} (none/stale)")
        return
    print(f"  {label}: pid {pid} {rd(f'/proc/{pid}/comm')}  cpuset={rd(f'/proc/{pid}/cpuset')} cgroup={rd(f'/proc/{pid}/cgroup')}")
    print(f"      {'tid':>8} {'comm':16} {'pol':6} {'rtp':>3} {'allowed':>12} {'last':>4} {'vol_cs':>10} {'invol_cs':>10}")
    for t in threads(pid):
        print(f"      {t[0]:>8} {t[1]:16} {t[2]:6} {t[3]:>3} {t[4]:>12} {t[5]:>4} {t[6]:>10} {t[7]:>10}")


for name in sys.argv[1:]:
    im = Image()
    if im.open(name) != 0:
        print(f"could not open {name}")
        continue
    nsem = im.md.sem
    print(f"=== {name}: {nsem} semaphores, posted in index order 0..{nsem - 1}")
    wpids = set(p for p in im.semWritePID if p > 0)
    for p in wpids:
        show_pid("writer", p)
    for i, p in enumerate(im.semReadPID):
        if p > 0:
            show_pid(f"sem[{i}] reader", p)
    im.close()
