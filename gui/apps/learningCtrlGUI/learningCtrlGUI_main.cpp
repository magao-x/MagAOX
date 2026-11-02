/** \file learningCtrlGUI_main.cpp
 * \brief Standard Qt/INDI launcher for the shared NN and PO4AO control panel.
 */
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "learningCtrl.hpp"
#include "multiIndiManager.hpp"

/// Launch one panel against the existing local INDI aggregation service.
int main( int   argc, /**< [in] Argument count. */
          char *argv[] /**< [in] Arguments; first positional argument is the companion device. */ )
{
    if( argc != 2 )
    {
        std::cerr << "Usage: learningCtrlGUI <po4aoTrainer device>\n";
        return 1;
    }
    const std::string device = argv[1];
    QApplication      app( argc, argv );
    QFile             file( ":/magaox.qss" );
    if( !file.open( QFile::ReadOnly | QFile::Text ) )
    {
        std::cerr << "Could not load the MagAO-X stylesheet\n";
        return 1;
    }
    QTextStream stream( &file );
    app.setStyleSheet( stream.readAll() );
    multiIndiManager  manager( device, "127.0.0.1", 7624 );
    xqt::learningCtrl panel( device );
    manager.addSubscriber( &panel );
    manager.activate();
    panel.show();
    return app.exec();
}
