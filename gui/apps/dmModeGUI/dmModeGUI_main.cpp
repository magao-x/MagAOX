
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "dmModeGUI.hpp"
#include <iostream>
   
int main(int argc, char *argv[])
{
   if(argc!=2)
   {
      std::cerr << "Must provide exactly one argument containing the INDI device name.\n";
      return -1;
   }
   
   
   std::string deviceName = argv[1];
   
   QApplication app(argc, argv);

   // set stylesheet
   QFile file(":/magaox.qss");
   file.open(QFile::ReadOnly | QFile::Text);
   QTextStream stream(&file);
   app.setStyleSheet(stream.readAll());
   
   multiIndiPublisher client(deviceName+"GUI", "127.0.0.1", 7624);

   xqt::dmModeGUI dmm(deviceName);
    
   dmm.m_deviceName = deviceName;
    
   dmm.subscribe(&client);
   dmm.show();
   client.activate();
   int rv = app.exec();
   client.quitProcess();
   client.deactivate();
   
   return rv;
}
   
