/** \file mcp3008Ctrl.cpp
 * \brief The MagAO-X MCP3008 Controller main program source file.
 *
 * \ingroup mcp3008Ctrl_files
 */

#include "mcp3008Ctrl.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::mcp3008Ctrl xapp;

    return xapp.main( argc, argv );
}
