
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "pwrGUI.hpp"
#include <iostream>
   
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   
   // set stylesheet
   QFile file(":/dark.qss");
   file.open(QFile::ReadOnly | QFile::Text);
   QTextStream stream(&file);
   app.setStyleSheet(stream.readAll());
   
    multiIndiPublisher client("pwrGUI", "127.0.0.1", 7624);

    xqt::pwrGUI pwr;

    pwr.subscribe(&client);

    pwr.show();

    client.activate();

    int rv = app.exec();
    
    client.quitProcess();
    client.deactivate();
   
    return rv;
}
   
