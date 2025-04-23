/** \file modalFilter.cpp
 * \brief The MagAO-X modal filter main program source file.
 *
 * \ingroup modalFilter_files
 */

#include "modalFilter.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::modalFilter xapp;

    return xapp.main( argc, argv );
}
