/** \file dmSpeckle.cpp
  * \brief The MagAO-X DM speckle maker main program source file.
  *
  * \ingroup dmSpeckle_files
  */

#include "sparkleClock.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::sparkleClock xapp;

   return xapp.main(argc, argv);

}
