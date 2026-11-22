/** \file psfAcqGUI_main.cpp
 * \brief Main program for the PSF acquisition GUI, e.g. `psfAcqGUI camacq-acq`.
 */

#include "app.hpp"
#include "psfAcq/psfAcq.hpp"

int main( int argc, char *argv[] )
{
    QApplication qapp( argc, argv );

    xqt::app<xqt::psfAcq> app;

    try
    {
        return app.main( argc, argv );
    }
    catch( const std::exception &e )
    {
        std::cerr << e.what() << "\n";
        std::cerr << "try " << argv[0] << " -h for more information." << std::endl;
    }
}
