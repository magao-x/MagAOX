
#include <QApplication>

#include "pwrGUI.hpp"

   
int main(int argc, char *argv[])
{
   //int data_type;
   QApplication app(argc, argv);

   multiIndiPublisher client("pwrGUI", "127.0.0.1", 7624);

   xqt::pwrGUI pwr;
   
   pwr.subscribe(&client);
   
   pwr.show();

   client.activate();
   
   int rv = app.exec();
   
   client.quitProcess();
   client.deactivate();
   
   return rv;
}
   
