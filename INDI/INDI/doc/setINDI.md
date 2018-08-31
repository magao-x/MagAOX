setINDI
=======

[TOC]

------------------------------------------------------------------------

# NAME 

setINDI − set one or more writable INDI property values

# SYNOPSIS 

**setINDI \[options\] {\[type\] device.property.e1\[;e2...\]=v1\[;v2...\]} ...
setINDI \[options\] {\[type\] device.property.e1=v1\[;e2=v2...\]} ...**

# DESCRIPTION 

setINDI connects to an indiserver and sends commands to set new values for specified properties. Each property is specified using three components followed by the new value in the following form:
```
device.property.element=value
```
Since a property may be an array, the syntax allows for multiple elements to be specified simultaneously. Two different syntax forms are supported. One form lists each element name separated by semicolons, then an equal sign, then each corresponding value also separated by semicolons. The other form lists each element=value together, each pair separated by a semicolon. In either form, all elements are updated atomically. If the property is of type BLOB then each element value is the name of a file to be sent.

# OPTIONS 

<table>
<colgroup>
<col width="10%" />
<col width="30%" />
<col width="60%" />
</colgroup>
<tbody>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-d &lt;f&gt;</p></td>
<td align="left"><p>use file descriptor f already open as a socket to the indiserver. This is useful for scripts to make a session connection one time then reuse it for each invocation. If the file descriptor seems to be being closed, check that the close-on-exec flag is off; for example in perl use something like:</p>
`#!/usr/bin/perl                                             ` <br \>
`use Socket;` <br \>
`use Fcntl;` <br \>
`socket(SOCK, PF\_INET, SOCK\_STREAM, getprotobyname(’tcp’));` <br \>
`connect(SOCK, sockaddr\_in(7624,inet\_aton(’localhost’)));` <br \>
`fcntl(SOCK,F\_SETFD,0);` <br \>
`$directfd = fileno(SOCK);` <br \>
`&runindi ("./setINDI", "-d", "$directfd", "x.y.z=10");` <br \>
`&runindi ("./setINDI", "-d", "$directfd", "a.b.c=hello");` <br \>
`sub runindi { if (fork()) { wait(); } else { exec @\_; } }` <br \>
</td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-h &lt;h&gt;</p></td>
<td align="left"><p>connect to alternate host h; the default is localhost.</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-p &lt;p&gt;</p></td>
<td align="left"><p>connect using alternate port p; the default is 7624.</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-t &lt;t&gt;</p></td>
<td align="left"><p>wait no longer than t seconds to accomplish setting the new values; the default is 2 seconds.</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-v</p></td>
<td align="left"><p>generate additional information on stderr. Additional v’s report successively more information. -w after sending the given property specifications, wait for the State of each property to becomes either Ok or Alert. This is useful for properties that have been designed to offer synchronous operation. Note this flag can not be used at the same time as the explicit type codes (see next) because these codes effectively suppress getting any response from the indiserver.</p></td>
</tr>
</tbody>
</table>

# TYPE 

Each property may optionally be preceded by a type code:

<table>
<colgroup>
<col width="20%" />
<col width="20%" />
<col width="60%" />
</colgroup>
<tbody>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-x</p></td>
<td align="left"><p>next property is of type Text</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-n</p></td>
<td align="left"><p>next property is of type Number</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-s</p></td>
<td align="left"><p>next property is of type Switch</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-b</p></td>
<td align="left"><p>next property is of type BLOB</p></td>
</tr>
</tbody>
</table>

If all properties are preceded by their type code, then a round trip to the server to discover their definitions is avoided and the session is much more efficient. However, this also precludes any error checking so each type indicated must in fact be correct or the commands will be silently ignored.

When developing a script of commands, one strategy is to use getINDI to get the exact property definitions one time, try the desired commands without the type codes to benefit from error checking, then add the type codes in the final optimized version.

# EXIT STATUS 

The setINDI program exits with a status of 0 if it succeeded in sending the commands to set new values for each specified property. It exits with 1 if there was at least one property for which a value could not be set within the given timeout period. It exits with 2 if there was some other error such as not being able to connect to the indiserver.

# EXAMPLES 

Send new lat/long numeric location values atomically to the Mount driver:

setINDI ’Mount.Location.Latitude;Longitude=30;100’

Same, but with alternative syntax and indicate type for greater efficiency:

setINDI -n ’Mount.Location.Latitude=30;Longitude=100’

# SEE ALSO 

evalINDI, getINDI, indiserver
http://www.clearskyinstitute.com/INDI/INDI.pdf


------------------------------------------------------------------------

This setINDI.md generated from the man documentation with the commands:
```
groff -mandoc -Thtml setINDI.man > setINDI.html
pandoc -t markdown_github setINDI.html > setINDI.md
```
and then tweaked for presentation and doxygen compatibility.
