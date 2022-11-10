/** \file photonCounter.cpp
  * \brief The MagAO-X photon counting image processor.
  *
  * \ingroup photonCounter_files
  */

#include "photonCounter.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::photonCounter xapp;

   return xapp.main(argc, argv);

}
