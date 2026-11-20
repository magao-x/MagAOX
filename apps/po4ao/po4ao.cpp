/** \file po4ao.cpp
  * \brief The MagAO-X ImageStreamIO integrator main program source file.
  *
  * \ingroup po4ao_files
  */

#include "po4ao.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::po4ao xapp;

   return xapp.main(argc, argv);

}
