/** \file resuctrl.cpp
 * \brief Manage the MagAO-X apps of a role as systemd services: main program
 *
 * \ingroup resuctrl_files
 */

#include "resuctrl.hpp"

int main( int argc, char **argv )
{
    resuctrl::resuCtrl rc;
    return rc.main( argc, argv );
}
