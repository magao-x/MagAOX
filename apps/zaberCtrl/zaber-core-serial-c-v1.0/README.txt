The Zaber Thin API in C
=======================
1.0

Thanks for choosing to use a Zaber device! This library exists as a minimal set
of functions to implement simple communication with a Zaber device without any
fancy bells or whistles. Its aim is to be useful to a developer or hobbyist who
wants to quickly integrate a Zaber device into a project, but is unfamiliar
with the intricacies of serial I/O, or simply does not have the time or 
patience for it.

Usage & Documentation
=====================
Depending on your device, you will need to communicate with it using either
the ASCII or binary protocol. Zaber recommends using the ASCII protocol
whenever possible. The files `za_serial.c` and `za_serial.h` should be added to
your project if you are using the ASCII protocol, and `zb_serial.c` and 
`zb_serial.h` should be used for the binary protocol. Regardless of which
protocol you choose, you will also need to add `z_common.h` to your project. 
It is not recommended to use both protocols simultaneously, but in the event 
you do choose to include both in one source file, there should be no issues.

Note that although this library is intended to ease the burden of serial I/O
in C, it does little to abstract the communication protocols of Zaber devices.
It is recommended that before trying to communicate with your device, you first
read the appropriate protocol manual. These manuals can be found at 
http://www.zaber.com/wiki/Manuals. 

Please see "example.c" for a short demonstration of the intended usage of the
API. There is also full Doxygen-generated documentation available online at
http://www.zaber.com/support/docs/api/core-c/.

Compilation Notes
=================
This API compiles using GCC without errors or warnings with the following 
arguments: -Wall -Wextra -std=c99 -pedantic

This means that the API conforms to the C99 standard, and does not use any GNU
extensions. It has also been tested with the Microsoft Visual C++ compiler, and
Apple's LLVM-powered Clang compiler. Please contact us if it does not compile
gracefully with your compiler of choice.

License
=======
This API is licensed under the Apache Software License Version 2.0. 
See LICENSE.txt for the full text.

Contact Us
==========
If you need to contact Zaber for any reason, please send an email to
contact@zaber.com. More detailed contact information can be found online at
http://www.zaber.com/contact/. Please do not hesitate to ask us for help with
this library or our devices.

