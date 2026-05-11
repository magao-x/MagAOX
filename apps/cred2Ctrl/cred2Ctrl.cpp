/** \file cred2Ctrl.cpp
 * \brief Main program for the MagAO-X C-RED 2 camera controller.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup cred2Ctrl_files
 */

#include "cred2Ctrl.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::cred2Ctrl app;

    return app.main( argc, argv );
}
