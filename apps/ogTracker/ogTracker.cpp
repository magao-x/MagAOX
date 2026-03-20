/** \file ogTracker.cpp
 * \brief The MagAO-X ogTracker main program source file.
 *
 * \ingroup ogTracker_files
 */

#include "ogTracker.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::ogTracker xapp;
    return xapp.main( argc, argv );
}
