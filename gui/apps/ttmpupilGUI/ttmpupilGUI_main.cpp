
#include <QApplication>

#include "ttmpupilGUI.hpp"

   
int main(int argc, char *argv[])
{
   //int data_type;
   QApplication app(argc, argv);

   multiIndiPublisher client("ttmpupilGUI", "127.0.0.1", 7624);

   xqt::ttmpupilGUI ttmpupil;
   
   ttmpupil.subscribe(&client);
   
   ttmpupil.show();

   client.activate();
   
   int rv = app.exec();
   
   client.quitProcess();
   client.deactivate();
   
   return rv;
}
   
