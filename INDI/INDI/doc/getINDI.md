getINDI
=======

[NAME](#NAME)
[SYNOPSIS](#SYNOPSIS)
[DESCRIPTION](#DESCRIPTION)
[OPTIONS](#OPTIONS)
[OUTPUT FORMAT](#OUTPUT%20FORMAT)
[EXIT STATUS](#EXIT%20STATUS)
[EXAMPLES](#EXAMPLES)
[SEE ALSO](#SEE%20ALSO)

------------------------------------------------------------------------

NAME []()
---------

getINDI − get INDI property values

SYNOPSIS []()
-------------
```
getINDI [options] [device.property.element ...]
```

DESCRIPTION []()
----------------

getINDI connects to an indiserver and reports the current value of one or more properties. Each property is specified using three components in the form:

```
device.property.element
```

Any component may be an asterisk, "\*", to serve as a wild card that matches all names in that component of the specification. If no property is specified, then all properties match, i.e., it is as if the specification "\*.\*.\*" were given.

The last component of the property specification is usually the element name, but may be a reserved name to indicate an attribute of the property as a whole. These reserved names are as follows:

<table>
<colgroup>
<col width="10%" />
<col width="30%" />
<col width="60%" />
</colgroup>
<tbody>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>_LABEL</p></td>
<td align="left"><p>report the label attribute</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>_GROUP</p></td>
<td align="left"><p>report the group attribute</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>_STATE</p></td>
<td align="left"><p>report the state attribute</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>_PERM</p></td>
<td align="left"><p>report the permission attribute</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>_TO</p></td>
<td align="left"><p>report the timeout attribute</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>_TS</p></td>
<td align="left"><p>report the timestamp attribute</p></td>
</tr>
</tbody>
</table>

OPTIONS []()
------------

<table>
<colgroup>
<col width="10%" />
<col width="30%" />
<col width="60%" />
</colgroup>
<tbody>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-1</p></td>
<td align="left"><p>print just the value if expectiong exactly one matching property</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-B</p></td>
<td align="left"><p>enable downloading BLOBs</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-d &lt;f&gt;</p></td>
<td align="left"><p>use file descriptor f already open as a socket to the indiserver. This is useful for scripts to make a session connection one time then reuse it for each invocation. If the file descriptor seems to be being closed, check that the close-on-exec flag is off; for example in perl use something like:</p>
`#!/usr/bin/perl` <br />
`use Socket;`<br />
`use Fcntl;`<br />
`socket(SOCK, PF_INET, SOCK_STREAM, getprotobyname(’tcp’));`<br />
`connect(SOCK, sockaddr_in(7624,inet_aton(’localhost’)));`<br />
`fcntl(SOCK,F_SETFD,0);`<br />
`$directfd = fileno(SOCK);`<br />
`%props = split (/[=0/, ’getINDI -d $directfd’);`<br />

</td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-h &lt;h&gt;</p></td>
<td align="left"><p>connect to alternate host h; the default is localhost.</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-m</p></td>
<td align="left"><p>continue to monitor for subsequent changes to each specified property until timeout.</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-p &lt;p&gt;</p></td>
<td align="left"><p>connect using alternate port p; the default is 7624.</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-q</p></td>
<td align="left"><p>suppress some error message.</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-t &lt;t&gt;</p></td>
<td align="left"><p>wait no longer than t seconds of no activity to gather the values for all the specified properties; the default is 2 seconds. Specify 0 to wait forever.</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-v</p></td>
<td align="left"><p>generate additional information on stderr. This is cumulative in that specifying more -v options will generate more output.</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-w</p></td>
<td align="left"><p>Usually only readable properties are shown. If this flag is set, then all properties, including those that are write-only, are shown.</p></td>
</tr>
</tbody>
</table>

OUTPUT FORMAT []()
------------------

For properties that are not BLOBs, the output of getINDI is one line per property. Unless the -1 option is given, each line is of the form:
```
property=value
```
A property that is a BLOB is saved in a file name device.property.element.format. Z compression is handled automatically, other formats are left unchanged. Note that BLOBs are not read by default, only when the -B option is used.

EXIT STATUS []()
----------------

The getINDI program exits with a status of 0 if it suceeded in finding the value for each specified property. It exits with 1 if there was at least one property for which no value was found within the given timeout period. It exits with 2 if there was some other error such as not being able to connect to the indiserver.

EXAMPLES []()
-------------

In a perl script, gather all properties for the default indiserver and save them in an associative array %props which can then be used to look up a property value by name:
```
%props = split (/[=0/, ’getINDI’);
```

Wait up to ten seconds to get the values of all properties from the Mount device on the given host and non-standard port:
```
getINDI -h indihost -p 7655 -t 10 "Mount.\*.\*"
```
Print just current value of the wind speed element from the weather device:
```
getINDI -1 Weather.Wind.Speed
```
SEE ALSO []()
-------------

evalINDI

setINDI

indiserver

http://www.clearskyinstitute.com/INDI/INDI.pdf

------------------------------------------------------------------------
This getINDI.md generated from the man documentation with the commands:
```
groff -mandoc -Thtml getINDI.man > getINDI.html
pandoc -t markdown_github getINDI.html > getINDI.md
```
and then tweaked for presentation.
