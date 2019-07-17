/** \file zaberLowLevel.cpp
  * \brief The MagAO-X Low-Level Zaber Controller
  *
  * \ingroup zaberLowLevel_files
  */

#include "zaberLowLevel.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::zaberLowLevel zll;

   return zll.main(argc, argv);

}
