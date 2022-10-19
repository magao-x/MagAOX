/** \file usbtempMon.cpp
  * \brief The MagAO-X xxxxx main program source file.
  *
  * \ingroup usbtempMon_files
  */

#include "usbtempMon.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::usbtempMon xapp;

   return xapp.main(argc, argv);

}
