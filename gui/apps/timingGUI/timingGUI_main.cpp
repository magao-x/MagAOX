/** \file timingGUI_main.cpp
 * \brief Start the MagAO-X timing and modulation panel.
 */
#include <QApplication>
#include <QFile>
#include <QTextStream>

#include "multiIndiManager.hpp"
#include "timingPanel.hpp"

/// Run timingGUI using the standard workstation INDI connection.
int main( int   argc, /**< [in] Argument count. */
          char *argv[] /**< [in,out] Qt command-line arguments. */ )
{
    QApplication app( argc, argv );

    QFile style( ":/magaox.qss" );
    if( style.open( QFile::ReadOnly | QFile::Text ) )
    {
        QTextStream stream( &style );
        app.setStyleSheet( stream.readAll() );
    }

    xqt::timingPanel panel;
    multiIndiManager manager( "timingGUI", "127.0.0.1", 7624 );
    manager.addSubscriber( &panel );
    manager.activate();

    panel.show();
    return app.exec();
}
