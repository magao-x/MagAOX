/** \file po4ao.cpp
 * \brief Main entry point for the MagAO-X modal PO4AO application.
 *
 * \ingroup po4ao_files
 */

#include "po4ao.hpp"

/// Run the configured PO4AO application and return its exit status.
int main( int    argc /**< [in] Number of command-line arguments. */,
          char **argv /**< [in] Command-line argument strings. */ )
{
    MagAOX::app::po4ao xapp;

    return xapp.main( argc, argv );
}
