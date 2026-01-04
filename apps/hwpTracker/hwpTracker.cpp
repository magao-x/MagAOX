/** \file hwpTracker.cpp
 * \brief The MagAO-X K-mirror rotation tracker main program source file.
 *
 * \ingroup hwpTracker_files
 */

#include "hwpTracker.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::hwpTracker xapp;

    return xapp.main( argc, argv );
}
