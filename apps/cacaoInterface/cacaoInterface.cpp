/** \file cacaoInterface.cpp
  * \brief The MagAO-X CACAO Interface main program source file.
  *
  * \ingroup cacaoInterface_files
  */

#include "cacaoInterface.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::cacaoInterface xapp;

   return xapp.main(argc, argv);

}
