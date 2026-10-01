/** \file dmTemporalResponse.cpp
 * \brief The MagAO-X DM temporal response measurement main program source file.
 *
 * \author Katie Twitchell (twitchell@arizona.edu)
 *
 * \ingroup dmTemporalResponse_files
 */

#include "dmTemporalResponse.hpp"

/// The main program for the dmTemporalResponse application.
int main( int    argc, /**< [in] the number of command line arguments */
          char **argv  /**< [in] the command line arguments */
)
{
    MagAOX::app::dmTemporalResponse xapp;

    return xapp.main( argc, argv );
}
