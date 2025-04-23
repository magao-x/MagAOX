/** \file nnReconstructor.cpp
  * \brief The MagAO-X ImageStreamIO integrator main program source file.
  *
  * \ingroup nnReconstructor_files
  */

#include "nnReconstructor.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::nnReconstructor xapp;

   return xapp.main(argc, argv);

}
