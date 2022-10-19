/** \file xt1121DCDU.cpp
  * \brief The MagAO-X xt1121-based D.C. Distribution Unit controller main program.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  * 
  * \ingroup xt1121DCDU_files
  */


#include "xt1121DCDU.hpp"

int main(int argc, char ** argv)
{
   MagAOX::app::xt1121DCDU dcdu;

   return dcdu.main(argc, argv);
}
