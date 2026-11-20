/** \file wooferTweeterRecon.cpp
 * \brief Main entrypoint for the MagAO-X woofer-tweeter pseudo-open-loop reconstructor
 *
 * \ingroup wooferTweeterRecon_files
 */

#include "wooferTweeterRecon.hpp"
int main( int argc, char **argv )
{
    MagAOX::app::wooferTweeterRecon xapp;

    return xapp.main( argc, argv );
}
