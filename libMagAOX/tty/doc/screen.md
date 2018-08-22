

Using `screen` to talk to TTY devices.

The `screen` program needs to be installed. You will need root privileges (sudo works).

```
screen /dev/ttyUSBX 115200
```

where you replace the dev node with the correct path, and the baud rate with the correct value.  See the particular devices specs for the correct baud rate to use.  Also See https:// for a list of baud rates. 

Note that screen does not echo, so if the device doesn't echo you won't see the characters you type.

