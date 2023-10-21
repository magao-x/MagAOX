
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "xyAlign.hpp"

#include "multiIndiManager.hpp"

int main(int argc, char *argv[])
{
      
   //int data_type;
   QApplication app(argc, argv);

   // set stylesheet
   QFile file(":/magaox.qss");
   file.open(QFile::ReadOnly | QFile::Text);
   QTextStream stream(&file);
   app.setStyleSheet(stream.readAll());

   multiIndiManager mgr("xyAlign", "127.0.0.1", 7624);
   
   xqt::xyAlign ca;
   mgr.addSubscriber(&ca);
   mgr.activate();
      
   ca.show();

   int rv = app.exec();
   
   return rv;
}
   
