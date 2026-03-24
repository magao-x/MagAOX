/** \file w2tcsOffloader.cpp
 * \brief The MagAO-X Woofer To Telescope Control System (TCS) offloader main program.
 *
 * \ingroup w2tcsOffloader_files
 */

#include "w2tcsOffloader.hpp"
/// Run the w2tcsOffloader application entry point.
int main( int argc, char **argv )
{
    MagAOX::app::w2tcsOffloader xapp;

    return xapp.main( argc, argv );
}
