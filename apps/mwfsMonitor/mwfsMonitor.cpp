/** \file mwfsMonitor.cpp
  * \brief monitor a vAPP PSF and pull out the hMWFS signal
  *
  * \ingroup mwfsMonitor_files
  */

#include "mwfsMonitor.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::mwfsMonitor xapp;

   return xapp.main(argc, argv);

}
