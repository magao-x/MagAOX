# libcommon -- The LBTI INDI application framework

The original author is Paul Grenz of the LBTI project.

## Changes

The following changes were made for use in MagAO-X:

- IndiElement.cpp line 438 changed to:
 - ssValue >> iValue;
 - return ssValue.good();
- IndiConnection.cpp: commented out ::signal on lines 118-120
- MutexLock.hpp: added noexcept(true) to d'tor decl.
- ReadWriteLock.hpp: added noexcept(true) to d'tor decl.
- IndiDriver.hpp: added noexcept(true) to d'tor decl.
- Removed logging of config elements, IndiConnection.ccp line 138

TODO:
- Should fully remove config system (comment it out), so we don't need to trick it with tmp files.
- Analyze codacy issues with d'tors which throw (MutexLock and ReadWriteLock).

