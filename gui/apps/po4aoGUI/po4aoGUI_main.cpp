/** \file po4aoGUI_main.cpp
 * \brief Standard Qt/INDI launcher for the shared NN and PO4AO control panel.
 */
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "po4ao.hpp"
#include "multiIndiManager.hpp"

/// Launch one panel against the existing local INDI aggregation service.
int main( int   argc, /**< [in] Argument count. */
          char *argv[] /**< [in] Arguments; optional PO4AO, trainer and independent NN device names. */ )
{
    if( argc > 4 )
    {
        std::cerr << "Usage: po4aoGUI [po4ao device [po4aoTrainer device [nnReconstructor device]]]\n";
        return 1;
    }
    const std::string runtime = argc > 1 ? argv[1] : "po4ao";
    const std::string trainer = argc > 2 ? argv[2] : "po4aoTrainer";
    const std::string nn      = argc > 3 ? argv[3] : "nnReconstructor";
    QApplication      app( argc, argv );
    QFile             file( ":/magaox.qss" );
    if( !file.open( QFile::ReadOnly | QFile::Text ) )
    {
        std::cerr << "Could not load the MagAO-X stylesheet\n";
        return 1;
    }
    QTextStream stream( &file );
    app.setStyleSheet( stream.readAll() );
    multiIndiManager  manager( runtime, "127.0.0.1", 7624 );
    xqt::po4ao panel( trainer, nullptr, runtime, nn );
    manager.addSubscriber( &panel );
    manager.activate();
    panel.show();
    return app.exec();
}
