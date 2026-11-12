#!/usr/bin/env python3
"""Minimal getINDI-like client: send a getProperties, count messages per second.

usage: client.py port seconds [device]
Prints per-second counts of def*, set*, new*, other; and a getINDI-style verdict:
getINDI quits after a quiet interval with no def* (default 2 s here).
"""
import re, socket, sys, time

port, dur = int(sys.argv[1]), float(sys.argv[2])
dev = sys.argv[3] if len(sys.argv) > 3 else ""
s = socket.create_connection(("127.0.0.1", port))
s.sendall((f"<getProperties version='1.7'{' device=%r' % dev if dev else ''}/>\n").encode())
s.settimeout(0.1)
tagre = re.compile(rb"<(def|set|new|del|message|getProperties)(\w*)[ >/]")
t0 = time.time(); lastdef = t0; buf = b""; sec = {}; keys = {}
quiet_exit = None
while time.time() - t0 < dur:
    try:
        d = s.recv(1 << 16)
        if not d:
            print("server closed"); break
    except socket.timeout:
        d = b""
    buf += d
    i = int(time.time() - t0)
    c = sec.setdefault(i, {"def": 0, "set": 0, "new": 0, "other": 0})
    end = buf.rfind(b">")
    for m in tagre.finditer(buf[:end + 1]):
        k = m.group(1).decode()
        c[k if k in c else "other"] += 1
        if k == "def":
            lastdef = time.time()
            dm = re.search(rb"device=['\"]([^'\"]*)['\"][^>]*name=['\"]([^'\"]*)", buf[m.start():m.start() + 300])
            if dm:
                kk = dm.group(1).decode() + "." + dm.group(2).decode()
                keys[kk] = keys.get(kk, 0) + 1
    buf = buf[end + 1:]
    if quiet_exit is None and time.time() - lastdef > 2.0:
        quiet_exit = time.time() - t0
for i in sorted(sec):
    print(i, sec[i])
print("getINDI-style quiet exit at:", "never" if quiet_exit is None else f"{quiet_exit:.1f}s")
top = sorted(keys.items(), key=lambda x: -x[1])[:10]
print("top def keys:", top)
