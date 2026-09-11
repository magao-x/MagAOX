/** \file orcaCtrl.cpp
 * \brief Main program for the MagAO-X Hamamatsu camera controller.
 *
 * \author Joshua Liberman (jliberman54@gmail.com)
 *
 * \ingroup orcaCtrl_files
 */

#include "orcaCtrl.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::orcaCtrl app;

    return app.main( argc, argv );
}
