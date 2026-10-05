/** \file flowRPM.cpp
 * \brief The MagAO-X flowRPM main program.
 *
 * \ingroup flowRPM_files
 */

#include "flowRPM.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::flowRPM xapp;

    return xapp.main( argc, argv );
}
