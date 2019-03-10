
#include <QApplication>

#include "modwfsGUI.hpp"

   
int main(int argc, char *argv[])
{
   //int data_type;
   QApplication app(argc, argv);

   multiIndiPublisher client("modwfsGUI", "127.0.0.1", 7624);

   xqt::modwfsGUI modwfs;
   
   modwfs.subscribe(&client);
   
   modwfs.show();

   client.activate();
   
   int rv = app.exec();
   
   client.quitProcess();
   client.deactivate();
   
   return rv;
}
   
