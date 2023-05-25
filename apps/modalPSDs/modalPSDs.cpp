/** \file modalPSDs.cpp
  * \brief The MagAO-X modalPSDs main program source file.
  *
  * \ingroup modalPSDs_files
  */

#include "modalPSDs.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::modalPSDs xapp;

   return xapp.main(argc, argv);

}
