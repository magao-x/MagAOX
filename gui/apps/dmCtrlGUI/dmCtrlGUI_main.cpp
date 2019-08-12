
#include <QApplication>

#include "dmCtrl.hpp"

   
int main(int argc, char *argv[])
{
   //int data_type;
   QApplication app(argc, argv);

   std::string dmName = "woofer";
   
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
   
