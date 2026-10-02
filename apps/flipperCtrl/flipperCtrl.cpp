/** \file flipperCtrl.cpp
 * \brief Main entrypoint for the MagAO-X two-position flipper controller.
 * \author MagAO-X developers
 *
 * \ingroup flipperCtrl_files
 */

#include "flipperCtrl.hpp"

/// Run the power-managed flipper controller.
/** \returns The application exit status. */
int main( int argc /**< [in] number of command-line arguments */, char **argv /**< [in] command-line arguments */ )
{
    MagAOX::app::flipperCtrl xapp;

    return xapp.main( argc, argv );
}
