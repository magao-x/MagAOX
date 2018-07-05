evalINDI
========

[NAME](#NAME)
[SYNOPSIS](#SYNOPSIS)
[DESCRIPTION](#DESCRIPTION)
[OPTIONS](#OPTIONS)
[EXIT STATUS](#EXIT%20STATUS)
[EXAMPLES](#EXAMPLES)
[SEE ALSO](#SEE%20ALSO)

------------------------------------------------------------------------

NAME []()
---------

evalINDI − evaluate an expression of INDI property values

SYNOPSIS []()
-------------

```
evalINDI [options] [exp]
```

DESCRIPTION []()
----------------

evalINDI connects to an indiserver and listens for the values of properties to evaluate an arithmetic expression. Each property is specified using three components enclosed in double quotes in the following form:
```
"device.property.element"
```

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
<td align="left"><p>_STATE</p></td>
<td align="left"><p>the state attribute, where for the purposes of evaluation the usual keywords <em>Idle, Ok, Busy</em> and <em>Alert</em> are converted to the numeric values of 0, 1, 2 and 3 respectively.</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>_TS</p></td>
<td align="left"><p>evaluate the timestamp attribute as the number of UNIX seconds from epoch</p></td>
</tr>
</tbody>
</table>

Switch vectors evalute to 0 or 1 based on the state values of Off and On, respectively. Light vectors evaluate to 0-3 similarly to the keywords described above for \_STATE.

The arithmetic expression, *exp,* follows the form of that used in the C programming language. The operators supported include:

```
! + - * / && || > >= == != < <=
```

and the mathematical functions supported include:

`sin(rad)` `cos(rad)` `tan(rad)` `asin(x)` `acos(x)` `atan(x)` `atan2(y,x)` `abs(x)` `degrad(deg)` `raddeg(rad)` `floor(x)` `log(x)` `log10(x)` `exp(x)` `sqrt(x)`  `pow(x,exp)`

The value of PI can be specified using a constant named "pi".

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
<td align="left"><p>-b</p></td>
<td align="left"><p>Ring the terminal bell when expression evaluates as true.</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-d &lt;f&gt;</p></td>
<td align="left"><p>use file descriptor f already open as a socket to the indiserver. This is useful for scripts to make a session connection one time then reuse it for each invocation. If the file descriptor seems to be being closed, check that the close-on-exec flag is off; for example in perl use something like:</p>

`#!/usr/bin/perl`<br/>
`use Socket;`<br/>
`use Fcntl;`<br/>
`socket(SOCK, PF_INET, SOCK_STREAM, getprotobyname(’tcp’));`<br/>
`connect(SOCK, sockaddr_in(7624,inet_aton(’localhost’)));`<br/>
`fcntl(SOCK,F_SETFD,0);`<br/>
`$directfd = fileno(SOCK);`<br/>
`&runindi ("./evalINDI", "-d", "$directfd", "\"x.y.z\"==1");`<br/>
`sub runindi { if (fork()) { wait(); } else { exec @_; } }`<br/>
</td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-e</p></td>
<td align="left"><p>print each updated expression value after each evaluation</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-f</p></td>
<td align="left"><p>print the final expression value</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-h &lt;h&gt;</p></td>
<td align="left"><p>connect to alternate host h; the default is localhost.</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-i</p></td>
<td align="left"><p>read the expression from stdin</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-o</p></td>
<td align="left"><p>print each operand each time it changes value in the form property=value</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-p &lt;p&gt;</p></td>
<td align="left"><p>connect using alternate port p; the default is 7624.</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-t &lt;t&gt;</p></td>
<td align="left"><p>wait no longer than t seconds to gather the initial values for all the specified properties; 0 means forever, the default is 2 seconds.</p></td>
</tr>
<tr class="even">
<td align="left"></td>
<td align="left"><p>-v</p></td>
<td align="left"><p>generate additional information on stderr. This is cumulative in that specifying more -v options will generate more output.</p></td>
</tr>
<tr class="odd">
<td align="left"></td>
<td align="left"><p>-w</p></td>
<td align="left"><p>evaluate the expression as many times as necessary until it evaluates to a value other than zero.</p></td>
</tr>
</tbody>
</table>

EXIT STATUS []()
----------------

The evalINDI program exits with a statis of 0 if the expression evaluates to non-0. It exits with 1 if the expression evaluated to 0. It exits with 2 if there was some other error such as not being able to connect to the indiserver.

EXAMPLES []()
-------------

Print 0/1 whether the Front or Rear elements of the Security property are in a state of Alert:
```
evalINDI -f ’"Security.Security.Front"==3 || "Security.Security.Rear"==3’
```
Exit 0 if the Security property as a whole is in a state of Ok:
```
evalINDI ’"Security.Security._STATE"==1’
```
Wait forever for RA and Dec to be near zero and watch their values as they change:
```
evalINDI -t 0 -wo ’abs("Mount.EqJ2K.RA")<.01 && abs("Mount.EqJ2K.Dec")<.01’
```
Wait forever for the wind speed to become larger than 50:
```
evalINDI -t 0 -w ’"Weather.Wind.Speed">50’
```
SEE ALSO []()
-------------

getINDI

setINDI

indiserver

http://www.clearskyinstitute.com/INDI/INDI.pdf

------------------------------------------------------------------------
This evalINDI.md generated from the man documentation with the commands:
```
groff -mandoc -Thtml evalINDI.man > evalINDI.html
pandoc -t markdown_github evalINDI.html > evalINDI.md
```
and then tweaked for presentation.
