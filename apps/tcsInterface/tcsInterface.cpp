/** \file tcsInterface.cpp
  * \brief The MagAO-X TCS Interface  main program source file.
  *
  * \ingroup tcsInterface_files
  */

#include "tcsInterface.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::tcsInterface xapp;

   return xapp.main(argc, argv);

}
