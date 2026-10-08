/** \file virtualPDU.cpp
 * \brief Virtual power distribution unit entrypoint.
 * \ingroup virtualPDU_files
 */
#include "virtualPDU.hpp"

/// Run the configured virtual power distribution unit.
int main( int argc /**< [in] command-line argument count */,
          char **argv /**< [in] command-line arguments */ )
{
    MagAOX::app::virtualPDU app;
    return app.main( argc, argv );
}
