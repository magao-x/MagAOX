/** \file qwpPupilTracker.cpp
 * \brief The MagAO-X QWP pupil tracker main program source file.
 *
 * \ingroup qwpPupilTracker_files
 */

#include "qwpPupilTracker.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::qwpPupilTracker xapp;

    return xapp.main( argc, argv );
}
