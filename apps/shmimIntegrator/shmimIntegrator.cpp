/** \file shmimIntegrator.cpp
  * \brief The MagAO-X ImageStreamIO integrator main program source file.
  *
  * \ingroup shmimIntegrator_files
  */

#include "shmimIntegrator.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::shmimIntegrator xapp;

   return xapp.main(argc, argv);

}
