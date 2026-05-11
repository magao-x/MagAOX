/** \file zaberLowLevelBinary.cpp
 * \brief The MagAO-X low-level binary-protocol Zaber controller main program.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup zaberLowLevelBinary_files
 */

#include "zaberLowLevelBinary.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::zaberLowLevelBinary zll;

    return zll.main( argc, argv );
}
