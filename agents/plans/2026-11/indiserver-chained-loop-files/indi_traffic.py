#!/usr/bin/env python3
"""Read-only INDI traffic profiler: which devices keep getINDI from going quiet.

usage: indi_traffic.py [host:]port [seconds] [device]

Sends one getProperties (like getINDI), then only listens. Prints a per-second
timeline (def/set/del/message counts and the top def devices that second), and for
t >= 3 s (after the reply to our own request): defs per device with the number of
distinct properties, so "dumps" ~ how many full def dumps that device sent.
"""
import re, socket, sys, time
from collections import Counter, defaultdict

hp = sys.argv[1] if len(sys.argv) > 1 else "7624"
host, port = (hp.rsplit(":", 1) if ":" in hp else ("localhost", hp))
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
dev = sys.argv[3] if len(sys.argv) > 3 else ""

s = socket.create_connection((host, int(port)))
s.sendall(("<getProperties version='1.7'%s/>\n" % (" device='%s'" % dev if dev else "")).encode())
s.settimeout(0.1)

tagre = re.compile(rb"<((?:def|set|new)\w*Vector|delProperty|message)\b([^>]*)>")
devre = re.compile(rb"device\s*=\s*['\"]([^'\"]*)['\"]")
namre = re.compile(rb"name\s*=\s*['\"]([^'\"]*)['\"]")
t0 = time.time(); lastdef = t0; quiet = None; buf = b""
sec = defaultdict(Counter)       # second -> tag counts
secdev = defaultdict(Counter)    # second -> def counts per device
devdefs = Counter(); devprops = defaultdict(set); defkeys = Counter(); setkeys = Counter()

while time.time() - t0 < dur:
    try:
        d = s.recv(1 << 20)
        if not d:
            print("server closed connection"); break
    except socket.timeout:
        d = b""
    buf += d
    end = buf.rfind(b">")
    now = time.time() - t0
    i = int(now)
    for m in tagre.finditer(buf, 0, end + 1):
        k = m.group(1).decode()[:3].replace("mes", "message")
        a = m.group(2)
        dm, nm = devre.search(a), namre.search(a)
        dv = dm.group(1).decode() if dm else ""
        nmv = nm.group(1).decode() if nm else ""
        sec[i][k] += 1
        if k == "def":
            lastdef = time.time()
            secdev[i][dv] += 1
            if now >= 3:
                devdefs[dv] += 1; devprops[dv].add(nmv); defkeys[dv + "." + nmv] += 1
        elif k == "set" and now >= 3:
            setkeys[dv + "." + nmv] += 1
    buf = buf[end + 1:]
    if quiet is None and time.time() - lastdef > 2.0:
        quiet = time.time() - t0

print("sec  def   set   del  msg  top def devices this second")
for i in sorted(sec):
    c = sec[i]
    top = ", ".join("%s:%d" % kv for kv in secdev[i].most_common(4))
    print("%3d %5d %5d %4d %4d  %s" % (i, c["def"], c["set"], c["del"], c["message"], top))
print("getINDI-style quiet exit (2 s without def):", "never" if quiet is None else "%.1f s" % quiet)
print("\nt>=3s defs by device (defs, distinct props, ~dumps):")
for dv, n in devdefs.most_common(25):
    print("  %-24s %7d %5d %7.1f" % (dv or "<none>", n, len(devprops[dv]), n / max(1, len(devprops[dv]))))
print("\nt>=3s top def keys:", defkeys.most_common(15))
print("\nt>=3s top set keys:", setkeys.most_common(15))
