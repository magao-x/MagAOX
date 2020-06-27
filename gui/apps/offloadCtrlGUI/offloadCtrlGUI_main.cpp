
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "offloadCtrl.hpp"

#include <unistd.h>

int main(int argc, char *argv[])
{
   
   
   //int data_type;
   QApplication app(argc, argv);

   // set stylesheet
   QFile file(":/dark.qss");
   file.open(QFile::ReadOnly | QFile::Text);
   QTextStream stream(&file);
   app.setStyleSheet(stream.readAll());

   multiIndiPublisher client("offloadCtrl", "127.0.0.1", 7624);

   xqt::offloadCtrl oc;
   
   oc.subscribe(&client);
   
   oc.show();

   client.activate();
   
   int rv = app.exec();
   
   client.quitProcess();

   return rv;
}
   
