/** \file loPredCtrlAccGUI_main.cpp
 * \brief Start the loPredCtrlAcc DDSPC operator panel.
 */
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "loPredCtrlAccPanel.hpp"
#include "multiIndiManager.hpp"

/// Run loPredCtrlAccGUI for the named device (default loPredCtrlAcc) on the standard workstation INDI connection.
int main( int   argc, /**< [in] Argument count. */
          char *argv[] /**< [in,out] Qt command-line arguments; the first is an optional INDI device name. */ )
{
    QApplication app( argc, argv );

    QFile style( ":/magaox.qss" );
    if( style.open( QFile::ReadOnly | QFile::Text ) )
    {
        QTextStream stream( &style );
        app.setStyleSheet( stream.readAll() );
    }

    const std::string device = argc > 1 ? argv[1] : "loPredCtrlAcc";

    xqt::loPredCtrlAccPanel panel( device );
    multiIndiManager        manager( device + "GUI", "127.0.0.1", 7624 );
    manager.addSubscriber( &panel );
    manager.activate();

    panel.show();
    return app.exec();
}
