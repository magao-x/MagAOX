#!/usr/bin/env python3
"""Fake INDI driver mimicking MagAOXApp getProperties/snoop behavior on stdin/stdout.

Device name is basename(argv[0]).  Config is read from $SIMCFG (JSON):
  {"<dev>": {"props": ["state", ...], "snoops": [["dev","prop"], ...], "setdt": 1.0}}
"""
import json, os, re, sys, threading, time
from datetime import datetime, timezone

name = os.path.basename(sys.argv[0])
cfg = json.load(open(os.environ["SIMCFG"])).get(name, {})
props = cfg.get("props", ["state"])
snoops = [tuple(s) for s in cfg.get("snoops", [])]
setdt = cfg.get("setdt", 0)
univ_at = cfg.get("universal_at", 0)
t_start = time.time()
lock = threading.Lock()
got = {s: False for s in snoops}
retry = {s: [0.0, 0.0] for s in snoops}  # next time, delay
log = open(os.environ.get("SIMLOG", "/dev/null") + "/" + name + ".log", "w", buffering=1)
counts = {"gp_all": 0, "gp_named": 0, "snoop_gp_sent": 0}


def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def out(s):
    with lock:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()


def defprop(p):
    out(f"<defTextVector device='{name}' name='{p}' state='Idle' perm='ro' timestamp='{ts()}'>"
        f"<defText name='current'>x</defText></defTextVector>")


def gp(d, p):
    counts["snoop_gp_sent"] += 1
    out(f"<getProperties version='1.7' device='{d}' name='{p}'/>")


def resend_all():
    for s in snoops:
        got[s] = False
        retry[s] = [time.time() + 1.0, 1.0]
        gp(*s)


def reader():
    buf = ""
    tagre = re.compile(r"<(getProperties|def\w+Vector|set\w+Vector|new\w+Vector)\b([^>]*)>")
    while True:
        chunk = sys.stdin.readline()
        if not chunk:
            os._exit(0)
        buf += chunk
        for m in tagre.finditer(buf):
            tag, attrs = m.group(1), m.group(2)
            a = dict(re.findall(r"(\w+)=['\"]([^'\"]*)['\"]", attrs))
            dev, pn = a.get("device", ""), a.get("name", "")
            if tag == "getProperties":
                if dev and dev != name:
                    continue
                if not pn:
                    counts["gp_all"] += 1
                    for p in props:
                        defprop(p)
                    resend_all()  # MagAOXApp: possible server restart, re-register
                elif pn in props:
                    counts["gp_named"] += 1
                    defprop(pn)
            elif tag.startswith(("def", "set")):
                if (dev, pn) in got:
                    got[(dev, pn)] = True
        buf = buf[buf.rfind(">") + 1:] if ">" in buf else buf


threading.Thread(target=reader, daemon=True).start()
resend_all()
last = time.time()
while True:
    time.sleep(0.1)
    now = time.time()
    for s in snoops:  # retry missing subscriptions with backoff, as MagAOXApp
        if not got[s] and now >= retry[s][0]:
            retry[s][1] = min(retry[s][1] * 2, 60.0)
            retry[s][0] = now + retry[s][1]
            gp(*s)
    if setdt and now - last >= setdt:
        last = now
        out(f"<setTextVector device='{name}' name='{props[0]}' state='Ok' timestamp='{ts()}'>"
            f"<oneText name='current'>{now:.1f}</oneText></setTextVector>")
    if univ_at and now - t_start >= univ_at:
        univ_at = 0
        out(f"<message timestamp='{ts()}' message='universal from {name}'/>")
    log.write(f"{now:.1f} {counts} missing={[s for s in snoops if not got[s]]}\n")
