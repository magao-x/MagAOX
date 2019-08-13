
#include <QApplication>

#include "dmCtrl.hpp"

   
int main(int argc, char *argv[])
{
   if(argc < 2)
   {
      std::cerr << "Must specify DM INDI name.\n";
      return -1;
   }
   
   std::string dmName = argv[1];
   
   //int data_type;
   QApplication app(argc, argv);

   multiIndiPublisher client(dmName, "127.0.0.1", 7624);

   xqt::dmCtrl dm(dmName);
   
   dm.subscribe(&client);
   
   dm.show();

   client.activate();
   
   int rv = app.exec();
   
   client.quitProcess();
   client.deactivate();
   
   return rv;
}
   
