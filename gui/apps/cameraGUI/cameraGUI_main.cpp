
#include "app.hpp"
#include "camera.hpp"


int main(int argc, char *argv[])
{
   if(argc < 2)
   {
      std::cerr << "Must specify DM INDI name.\n";
      return -1;
   }
   
   QApplication qapp(argc, argv);


   xqt::app<xqt::camera> app;

   return app.main(argc, argv);
   
   
}
   
