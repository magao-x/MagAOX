
#include <QApplication>
#include <QFile>
#include <QTextStream>


#include "loopCtrl.hpp"

   
#include "multiIndiManager.hpp"
   
   
int main(int argc, char *argv[])
{
   if(argc < 2)
   {
      std::cerr << "Must specify loop INDI name.\n";
      return -1;
   }
   
   std::string procName = argv[1];
   
   QApplication app(argc, argv);

   // set stylesheet
   QFile file(":/magaox.qss");
   file.open(QFile::ReadOnly | QFile::Text);
   QTextStream stream(&file);
   app.setStyleSheet(stream.readAll());

   multiIndiManager mgr(procName, "127.0.0.1", 7624);

   xqt::loopCtrl loop(procName);
   mgr.addSubscriber(&loop);
   
   loop.show();
   
   int rv = app.exec();
   
   
   return rv;
}
   
