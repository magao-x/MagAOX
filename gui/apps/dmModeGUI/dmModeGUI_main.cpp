
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "dmMode.hpp"

#include "multiIndiManager.hpp"

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
   
   multiIndiManager mgr(deviceName, "127.0.0.1", 7624);
   
   xqt::dmMode dmm(deviceName);
   mgr.addSubscriber(&dmm); 
   mgr.activate();
   
   dmm.show();

   int rv = app.exec();
   
   return rv;
}
   
