/** \file pwfsAlignment.cpp
 * \brief Draw measured and target pyramid wavefront sensor pupils in rtimv.
 */
#include "pwfsAlignment.hpp"

#include <cmath>

pwfsAlignment::~pwfsAlignment()
{
    detachOverlay();
}

void pwfsAlignment::detachOverlay()
{
    delete m_items.data();
    m_scene.clear();
    m_measured   = {};
    m_target     = {};
    m_enabled    = false;
    m_dictionary = nullptr;
    m_deviceName.clear();
}

int pwfsAlignment::attachOverlay( rtimvOverlayAccess &access, mx::app::appConfigurator &config )
{
    detachOverlay();
    config.configUnused( m_deviceName, "pwfsAlignment", "name" );

    if( m_deviceName.empty() )
    {
        return 1;
    }

    if( !access.m_graphicsView || !access.m_graphicsView->scene() || !access.m_dictionary )
    {
        return -1;
    }

    m_scene      = access.m_graphicsView->scene();
    m_dictionary = access.m_dictionary;
    m_dictionary->try_emplace( m_deviceName + ".numPupils.value" );
    m_dictionary->try_emplace( m_deviceName + ".sm_frameSize.width" );
    m_dictionary->try_emplace( m_deviceName + ".sm_frameSize.height" );

    for( int quadrant = 1; quadrant <= 4; ++quadrant )
    {
        const auto prefix = m_deviceName + ".quadrant" + std::to_string( quadrant ) + ".";

        for( const char *element : { "x", "y", "D", "set-x", "set-y", "set-D" } )
        {
            m_dictionary->try_emplace( prefix + element );
        }
    }

    pluginLogInfo( std::format( "configured for {}", m_deviceName ) );
    return 0;
}

bool pwfsAlignment::number( const std::string &key, double &value ) const
{
    const auto it = m_dictionary->find( key );
    char       buffer[512];

    if( it == m_dictionary->end() || it->second.getBlobStr( buffer, sizeof( buffer ) ) >= sizeof( buffer ) )
    {
        return false;
    }

    bool ok = false;
    value   = QByteArray( buffer ).toDouble( &ok );
    return ok && std::isfinite( value );
}

void pwfsAlignment::createDrawing( drawing &items, const QPen &stroke )
{
    for( size_t i = 0; i < items.m_circles.size(); ++i )
    {
        auto *edge = new QGraphicsLineItem( m_items );
        edge->setPen( stroke );
        edge->setAcceptedMouseButtons( Qt::NoButton );
        items.m_edges[i] = edge;

        auto *circle = new QGraphicsEllipseItem( m_items );
        circle->setPen( stroke );
        circle->setAcceptedMouseButtons( Qt::NoButton );
        items.m_circles[i] = circle;
    }
}

void pwfsAlignment::updateDrawing( drawing &items, const std::string &prefix, int count, double height )
{
    std::array<QPointF, 4> centers{};
    std::array<bool, 4>    valid{};

    for( int i = 0; i < 4; ++i )
    {
        const auto key = m_deviceName + ".quadrant" + std::to_string( i + 1 ) + "." + prefix;
        double     x, y, diameter;
        valid[i] = i < count && number( key + "x", x ) && number( key + "y", y ) && number( key + "D", diameter ) &&
                   diameter > 0;
        auto *circle = items.m_circles[i];
        circle->setVisible( valid[i] );

        if( valid[i] )
        {
            centers[i] = QPointF( x, height - 1 - y );
            circle->setRect( x - diameter / 2, centers[i].y() - diameter / 2, diameter, diameter );
        }
    }

    // Three pupils form a triangle. Four form a quadrilateral in fitter quadrant order.
    const std::array<int, 4> from{ 0, 0, 2, 1 };
    const std::array<int, 4> to{ 1, 2, count == 3 ? 1 : 3, 3 };

    for( int i = 0; i < 4; ++i )
    {
        auto      *edge    = items.m_edges[i];
        const bool visible = i < count && valid[from[i]] && valid[to[i]];
        edge->setVisible( visible );

        if( visible )
        {
            edge->setLine( QLineF( centers[from[i]], centers[to[i]] ) );
        }
    }
}

int pwfsAlignment::updateOverlay()
{
    if( !m_enabled || !m_scene )
    {
        return 0;
    }

    double count, width, height;

    if( !number( m_deviceName + ".numPupils.value", count ) || ( count != 3 && count != 4 ) ||
        !number( m_deviceName + ".sm_frameSize.width", width ) || width <= 0 ||
        !number( m_deviceName + ".sm_frameSize.height", height ) || height <= 0 )
    {
        if( m_items )
        {
            m_items->hide();
        }

        return 0;
    }

    if( !m_items )
    {
        m_items = new QGraphicsWidget;
        m_items->setAcceptedMouseButtons( Qt::NoButton );
        m_scene->addItem( m_items );
        createDrawing( m_measured, QPen( Qt::red ) );
        createDrawing( m_target, QPen( QColor( "lime" ), 0.1 ) );
    }

    updateDrawing( m_measured, "", static_cast<int>( count ), height );
    updateDrawing( m_target, "set-", static_cast<int>( count ), height );
    m_items->show();
    return 0;
}

void pwfsAlignment::keyPressEvent( QKeyEvent *event )
{
    if( event->text() == "P" )
    {
        if( m_enabled )
        {
            disableOverlay();
        }
        else
        {
            enableOverlay();
        }
    }
}

bool pwfsAlignment::overlayEnabled()
{
    return m_enabled;
}

void pwfsAlignment::enableOverlay()
{
    m_enabled = true;
    updateOverlay();
}

void pwfsAlignment::disableOverlay()
{
    m_enabled = false;

    if( m_items )
    {
        m_items->hide();
    }
}

std::vector<std::string> pwfsAlignment::info()
{
    return { "PWFS Alignment overlay" };
}
