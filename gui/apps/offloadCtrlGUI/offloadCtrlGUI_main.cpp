
#include <QApplication>

#include "offloadCtrl.hpp"

#include <unistd.h>

int main(int argc, char *argv[])
{
   
   
   //int data_type;
   QApplication app(argc, argv);

   multiIndiPublisher client("offloadCtrl", "127.0.0.1", 7624);

   xqt::offloadCtrl oc;
   
   oc.subscribe(&client);
   
   oc.show();

   client.activate();
   
   int rv = app.exec();
   
   client.quitProcess();

   return rv;
}
   
