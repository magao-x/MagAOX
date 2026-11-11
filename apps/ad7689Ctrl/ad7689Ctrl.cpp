/** \file ad7689Ctrl.cpp
 * \brief The MagAO-X AD7689 Controller main program source file.
 *
 * \ingroup ad7689Ctrl_files
 */

#include "ad7689Ctrl.hpp"

/// Run the standalone AD7689 MagAO-X application.
int main( int argc /**< [in] argument count */, char **argv /**< [in] argument values */ )
{
    MagAOX::app::ad7689Ctrl xapp;

    return xapp.main( argc, argv );
}