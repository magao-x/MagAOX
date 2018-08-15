/** \file zaberCtrl.cpp
  * \brief The MagAO-X Zaber Controller
  *
  * \ingroup zaberCtrl_files
  */

#include "zaberCtrl.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::zaberCtrl zc;

   return zc.main(argc, argv);

}
