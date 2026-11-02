/** \file dlDataCollection.cpp
 * \brief Main entrypoint for synchronized PWFS training-data collection.
 *
 * \ingroup dlDataCollection_files
 */

#include "dlDataCollection.hpp"

/// Run the collector using the standard MagAOX application lifecycle.
int main( int argc /**< [in] Number of command-line arguments. */, char **argv /**< [in] Command-line arguments. */ )
{
    MagAOX::app::dlDataCollection xapp;

    return xapp.main( argc, argv );
}
