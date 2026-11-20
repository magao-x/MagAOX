/** \file xctrl.cpp
 * \brief Manage the MagAO-X apps of a role as systemd services: main program
 *
 * \ingroup xctrl_files
 */

#include "xctrl.hpp"

int main( int argc, char **argv )
{
    xctrl::xCtrl rc;
    return rc.main( argc, argv );
}
