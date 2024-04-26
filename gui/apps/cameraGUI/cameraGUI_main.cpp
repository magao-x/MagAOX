
#include "app.hpp"
#include "camera.hpp"


int main(int argc, char *argv[])
{
   
    QApplication qapp(argc, argv);

    xqt::app<xqt::camera> app;

    try
    {
        return app.main(argc, argv);
    }
    catch(const std::exception & e)
    {
        std::cerr << e.what() << "\n";
        std::cerr << "try " << argv[0] << " -h for more information." << std::endl;
    }
}
   
