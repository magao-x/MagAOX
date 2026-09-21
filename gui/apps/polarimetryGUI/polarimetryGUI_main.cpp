#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "polarimetry.hpp"

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

   multiIndiManager mgr("polarimetry", "127.0.0.1", 7624);

   xqt::polarimetry polGui;
   mgr.addSubscriber(&polGui);
   mgr.activate();

   polGui.show();

   return app.exec();
}

