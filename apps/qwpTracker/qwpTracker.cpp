/** \file qwpTracker.cpp
 * \brief The MagAO-X dual QWP rotation tracker main program source file.
 *
 * \ingroup qwpTracker_files
 */

#include "qwpTracker.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::qwpTracker xapp;

    return xapp.main( argc, argv );
}
