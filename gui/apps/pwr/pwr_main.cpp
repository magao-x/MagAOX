
#include <QApplication>

#include "pwrGUI.hpp"
#include <iostream>
   
int main(int argc, char *argv[])
{
   std::cerr << "1\n";
   //int data_type;
   QApplication app(argc, argv);
// 
    std::cerr << "2\n";
    multiIndiPublisher client("pwrGUI", "127.0.0.1", 7624);
// 
    std::cerr << "3\n";
    xqt::pwrGUI pwr;
//    
    std::cerr << "4\n";
    pwr.subscribe(&client);
//    
    std::cerr << "5\n";
    pwr.show();
// 
    std::cerr << "6\n";
    client.activate();
//    
    std::cerr << "7\n";
    int rv = app.exec();
//    
    std::cerr << "8\n";
    client.quitProcess();
    client.deactivate();
   
    return rv;
}
   
