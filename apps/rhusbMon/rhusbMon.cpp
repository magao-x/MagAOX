/** \file rhusbMon.cpp
  * \brief The MagAO-X RH USB Monitor main program source file.
  *
  * \ingroup rhusbMon_files
  */

#include "rhusbMon.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::rhusbMon xapp;

   return xapp.main(argc, argv);

}
