
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "pwrGUI.hpp"
#include <iostream>
   
#include "multiIndiManager.hpp"

int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   
   // set stylesheet
   QFile file(":/magaox.qss");
   file.open(QFile::ReadOnly | QFile::Text);
   QTextStream stream(&file);
   app.setStyleSheet(stream.readAll());
   
   
   multiIndiManager mgr("pwrGUI", "127.0.0.1", 7624);
   
   xqt::pwrGUI pwr;
      
   mgr.addSubscriber(&pwr);
   mgr.activate();

   pwr.show();
   
   int rv = app.exec();
    
   return rv;
}
   
