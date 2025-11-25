/** \file mcp3208Ctrl.cpp
 * \brief The MagAO-X MCP3208 Controller main program source file.
 *
 * \ingroup mcp3208Ctrl_files
 */

#include "mcp3208Ctrl.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::mcp3208Ctrl xapp;

    return xapp.main( argc, argv );
}