/** \file counterOffload.cpp
 * \brief The MagAO-X tweeter counter-offload producer main program source file.
 *
 * \ingroup counterOffload_files
 */

#include "counterOffload.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::counterOffload xapp;

    return xapp.main( argc, argv );
}
