/** \file xindiserver.cpp
  * \brief The MagAO-X indiserver wrapper main program.
  *
  * \ingroup xindiserver_files
  */

#include "xindiserver.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::xindiserver xindi;

   return xindi.main(argc, argv);

}
