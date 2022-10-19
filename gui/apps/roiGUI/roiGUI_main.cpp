
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "roi.hpp"
   
#include "multiIndiManager.hpp"
   
int main(int argc, char *argv[])
{
   if(argc < 2)
   {
      std::cerr << "Must specify DM INDI name.\n";
      return -1;
   }
   
   std::string roiName = argv[1];
   
   QApplication app(argc, argv);

   // set stylesheet
   QFile file(":/magaox.qss");
   file.open(QFile::ReadOnly | QFile::Text);
   QTextStream stream(&file);
   app.setStyleSheet(stream.readAll());

   multiIndiManager mgr(roiName, "127.0.0.1", 7624);

   xqt::roi roi(roiName);
   mgr.addSubscriber(&roi);
   mgr.activate();
   
   roi.show();
   
   int rv = app.exec();
   
   return rv;
}
   
