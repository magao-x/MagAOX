# The MagAO-X common INDI library.

INDI stands for "Instrument Neutral Distributed Interface".  

This code is originally from the LBTI project, where it is called "libcommon".  The original author is Paul Grenz.  We have modified it to be used as part of MagAO-X.  It will need to be periodically updated with bug-fixes, etc., from the LBTI_INDI code base.

# Changes Made
The following changes were made for MagAO-X:
 - Unneeded files removed.
 - Calls to ::signal commented out in IndiConnection::construct
 - Use of libcommon config system removed from IndiDriver and IndiConnection.
 - Makefile modified as follows:
   -- Include common.mk to inherit MagAO-X build settings.
   -- Changed to targets all and clean to conform to MagAO-X conventions
   -- Changed to ar -r, and ranlib is ar -s, and added command to make shared library.
