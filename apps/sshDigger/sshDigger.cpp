/** \file sshDigger.cpp
  * \brief The MagAO-X SSH tunnel manager
  *
  * \ingroup sshDigger_files
  */

#include "sshDigger.hpp"


int main(int argc, char **argv)
{
   MagAOX::app::sshDigger dig;

   return dig.main(argc, argv);

}
