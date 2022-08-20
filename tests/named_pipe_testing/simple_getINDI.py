"""
( echo '<getProperties/>' ; sleep 20 ) 
| telnet localhost 7624 
| sed 's/^$/	/' 
| tr -d \n 
| tr \t \n 
| grep -vE 'message|^</' 
| tee x.x
"""

import re
import sys
import socket
nl,tab = chr(10),chr(9)
rgx_discards = re.compile('^[\t ]*<(message|/)')

try:
  sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
  sock.connect(('127.0.0.1',7624,))
  ##sock.send(b'<getProperties devicedevice="timeSeriesSimulator" name="function_out" />\n')
  sock.send(b'<getProperties />\n')
  pfx = ''
  outline = False
  while True:
    s = pfx + sock.recv(1024).decode('8859').replace('\r','')
    if not s: break
    lines = s.split(nl)
    if nl == s[-1]: pfx = lines.pop()
    else          : pfx = ''
    if not lines: continue
    if 1 < len(lines): lines.reverse()
    while lines:
      nextline = lines.pop()
      if not nextline: nextline = tab
      if tab == nextline[0]:
        if outline: print(outline) ; sys.stdout.flush()
        nextline = nextline[1:]
        outline = ''
      if rgx_discards.match(nextline): continue
      outline += nextline
except KeyboardInterrupt as e: print('KeyboardInterrupt received; exiting ...')
except:
  import traceback as tb
  tb.print_exc()
finally:
  sock.close()
