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
- 2018-09-02: Fully removed logging and config from IndiConnection/Driver/Client, and removed those files from the repo.

TODO:
- Should fully remove config system (comment it out), so we don't need to trick it with tmp files.
- Analyze codacy issues with d'tors which throw (MutexLock and ReadWriteLock).


## Version History

2018-05-17: The first addition of this code to the MagAO-X repo was based on the LBTI_INDI svn rep:
```
Working Copy Root Path: /home/jrmales/Source/LBTI_INDI
URL: http://lbti.as.arizona.edu/svn/lbti/LBTI_INDI
Relative URL: ^/LBTI_INDI
Repository Root: http://lbti.as.arizona.edu/svn/lbti
Repository UUID: ba7a21ae-fe2f-0410-95a9-ce593fb397f7
Revision: 9792
Node Kind: directory
Schedule: normal
Last Changed Author: lbti
Last Changed Rev: 9792
Last Changed Date: 2018-05-17 14:55:08 -0700 (Thu, 17 May 2018)
```


