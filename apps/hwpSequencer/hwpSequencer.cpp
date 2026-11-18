/** \file hwpSequencer.cpp
 * \brief The MagAO-X HWP sequencer main program source file.
 *
 * \ingroup hwpSequencer_files
 */

#include "hwpSequencer.hpp"

int main( int argc, char **argv )
{
    MagAOX::app::hwpSequencer xapp;

    return xapp.main( argc, argv );
}
