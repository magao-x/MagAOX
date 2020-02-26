
#include <QApplication>

#include "pupilGuide.hpp"

#include "multiIndiManager.hpp"

int main(int argc, char *argv[])
{
      
   //int data_type;
   QApplication app(argc, argv);

   multiIndiManager mgr("pupilGuide", "127.0.0.1", 7624);
   
   xqt::pupilGuide dm;
   mgr.addSubscriber(&dm);
      
   dm.show();

   int rv = app.exec();
   
   return rv;
}
   
