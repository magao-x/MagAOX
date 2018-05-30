# The lilxml library.

This code is originally from the LBTI project.  The original author is Elwood Downey.  We have modified it slightly to be used as part of MagAO-X.  It will need to be periodically updated with bug-fixes, etc., from the LBTI_INDI code base.

# Changes Made
The following changes were made for MagAO-X:
 - Added extern "C" protection to headers.
 - Removed extraneous files.
 - Makefile modified as follows:
   -- Include common.mk to inherit MagAO-X build settings.
   -- Changed to targets all and clean to conform to MagAO-X conventions
   -- Changed to ar -r, and ranlib is ar -s, and added command to make shared library.
