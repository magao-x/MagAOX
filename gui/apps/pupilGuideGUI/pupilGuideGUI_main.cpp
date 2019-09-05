
#include <QApplication>

#include "pupilGuide.hpp"

#include <unistd.h>

int main(int argc, char *argv[])
{
   
   
   //int data_type;
   QApplication app(argc, argv);

   multiIndiPublisher client("pupilGuide", "127.0.0.1", 7624);

   xqt::pupilGuide dm;
   
   dm.subscribe(&client);
   
   dm.show();

   client.activate();
   
   int rv = app.exec();
   
   client.quitProcess();

   return rv;
}
   
