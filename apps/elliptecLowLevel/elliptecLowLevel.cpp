/** \file elliptecLowLevel.cpp
 * \brief The MagAO-X low-level Elliptec bus controller main program.
 * \author Miles Lucas
 *
 * \ingroup elliptecLowLevel_files
 */

#include "elliptecLowLevel.hpp"

/// Run the low-level Elliptec bus controller.
int main( int    argc, /**< [in] number of command-line arguments */
          char **argv /**< [in] command-line arguments */ )
{
    MagAOX::app::elliptecLowLevel ell;

    return ell.main( argc, argv );
}
