
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "stage.hpp"
   
#include "multiIndiManager.hpp"
   
int main(int argc, char *argv[])
{
   if(argc < 2)
   {
      std::cerr << "Must specify stage INDI name.\n";
      return -1;
   }
   
   std::string stageName = argv[1];
   
   QApplication app(argc, argv);

   // set stylesheet
   QFile file(":/magaox.qss");
   file.open(QFile::ReadOnly | QFile::Text);
   QTextStream stream(&file);
   app.setStyleSheet(stream.readAll());

   multiIndiManager mgr(stageName, "127.0.0.1", 7624);

   xqt::stage stage(stageName);
   mgr.addSubscriber(&stage);
   mgr.activate();
   
   stage.show();
   
   int rv = app.exec();
   
   return rv;
}
   
