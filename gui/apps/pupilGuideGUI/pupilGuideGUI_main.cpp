
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "pupilGuide.hpp"

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

   multiIndiManager mgr("pupilGuide", "127.0.0.1", 7624);
   xqt::pupilGuide pg;
   
   mgr.addSubscriber(&pg);
   mgr.activate();

   pg.show();

   int rv = app.exec();
   
   return rv;
}
   
