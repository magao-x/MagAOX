/** \file acquisition.cpp
 * \brief Implements the acquisition overlay for detected-star annotations.
 */

#include "acquisition.hpp"

#include <algorithm>

#define errPrint( expl ) std::cerr << "acquisition: " << __FILE__ << " " << __LINE__ << " " << expl << std::endl;

namespace
{
constexpr size_t c_maxTrackedStars{ 128 };

void hideStarOverlay( StretchCircle *sc, QTextEdit *te )
{
    if( sc != nullptr )
    {
        sc->setVisible( false );
    }

    if( te != nullptr )
    {
        te->setVisible( false );
    }
}

void clearStarOverlays( std::vector<QPointer<StretchCircle>> &starCircles,
                        std::vector<QPointer<QTextEdit>>     &starLabels )
{
    for( size_t s = 0; s < starCircles.size(); ++s )
    {
        if( starCircles[s] != nullptr )
        {
            starCircles[s]->remove();
        }
    }

    for( size_t s = 0; s < starLabels.size(); ++s )
    {
        if( starLabels[s] != nullptr )
        {
            starLabels[s]->setVisible( false );
            starLabels[s]->deleteLater();
        }
    }

    starCircles.clear();
    starLabels.clear();
}
} // namespace

acquisition::acquisition() : rtimvOverlayInterface()
{
}

acquisition::~acquisition()
{
    std::lock_guard<std::mutex> guard( m_starCircleMutex );
    clearStarOverlays( m_starCircles, m_starLabels );
}

void acquisition::ensureStarOverlay( size_t n )
{
    if( n >= m_starCircles.size() || n >= m_starLabels.size() )
    {
        return;
    }

    if( m_starCircles[n].isNull() )
    {
        StretchCircle *sc = new StretchCircle;
        sc->setPenColor( m_color.c_str() );
        sc->setPenWidth( 0 );
        sc->setVisible( false );
        sc->setStretchable( false );
        sc->setRemovable( false );
        connect( sc, SIGNAL( remove( StretchCircle * ) ), this, SLOT( stretchCircleRemove( StretchCircle * ) ) );
        emit newStretchCircle( sc );
        m_starCircles[n] = sc;
    }

    if( m_starLabels[n].isNull() )
    {
        QTextEdit *te = new QTextEdit( m_roa.m_graphicsView );
        QFont      qf = te->currentFont();
        qf.setPixelSize( m_fontSize );
        te->setCurrentFont( qf );
        te->setVisible( false );
        te->setTextColor( m_color.c_str() );
        m_roa.m_graphicsView->textEditSetup( te );
        m_starLabels[n] = te;
    }
}

int acquisition::attachOverlay( rtimvOverlayAccess &roa, mx::app::appConfigurator &config )
{
    m_roa = roa;
    m_qgs = roa.m_graphicsView->scene();

    config.configUnused( m_deviceName, mx::app::iniFile::makeKey( "acquisition", "fitter" ) );

    if( m_deviceName == "" )
    {
        pluginLogInfo( "not configured" );

        m_enableable = false;
        disableOverlay();
        return 1; // Tell rtimv to unload me since not configured.
    }

    pluginLogInfo( std::format( "enabling for {}", m_deviceName ) );

    config.configUnused( m_cameraName, mx::app::iniFile::makeKey( "acquisition", "camera" ) );
    config.configUnused( m_circRad, mx::app::iniFile::makeKey( "acquisition", "radius" ) );
    config.configUnused( m_color, mx::app::iniFile::makeKey( "acquisition", "color" ) );
    config.configUnused( m_fontSize, mx::app::iniFile::makeKey( "acquisition", "fontSize" ) );

    m_enableable = true;
    m_enabled    = true;

    if( m_roa.m_dictionary != nullptr )
    {
        // Register the camera size and a bounded number of star properties up front
        // so remote update jitter cannot force runtime growth of the shared dictionary.
        ( *m_roa.m_dictionary )[m_cameraName + ".fg_frameSize.width"].setBlob( nullptr, 0 );
        ( *m_roa.m_dictionary )[m_cameraName + ".fg_frameSize.height"].setBlob( nullptr, 0 );
        ( *m_roa.m_dictionary )[m_deviceName + ".num_stars.current"].setBlob( nullptr, 0 );

        for( size_t n = 0; n < c_maxTrackedStars; ++n )
        {
            std::string star = ".star_" + std::to_string( n );

            ( *m_roa.m_dictionary )[m_deviceName + star + ".x"].setBlob( nullptr, 0 );
            ( *m_roa.m_dictionary )[m_deviceName + star + ".y"].setBlob( nullptr, 0 );
            ( *m_roa.m_dictionary )[m_deviceName + star + ".peak"].setBlob( nullptr, 0 );
            ( *m_roa.m_dictionary )[m_deviceName + star + ".fwhm"].setBlob( nullptr, 0 );
        }
    }

    connect( this,
             SIGNAL( newStretchCircle( StretchCircle * ) ),
             m_roa.m_mainWindowObject,
             SLOT( addStretchCircle( StretchCircle * ) ) );

    {
        std::lock_guard<std::mutex> guard( m_starCircleMutex );

        m_starCircles.resize( c_maxTrackedStars, nullptr );
        m_starLabels.resize( c_maxTrackedStars, nullptr );

        for( size_t n = 0; n < c_maxTrackedStars; ++n )
        {
            // Pre-create the full bounded overlay set so delayed INDI updates only
            // show and hide items instead of tearing down Qt objects mid-stream.
            ensureStarOverlay( n );
        }
    }

    if( m_enabled )
        enableOverlay();
    else
        disableOverlay();

    return 0;
}

bool acquisition::blobExists( const std::string &propel )
{
    if( m_roa.m_dictionary->count( m_deviceName + "." + propel ) == 0 )
    {
        return false;
    }

    if( ( *m_roa.m_dictionary )[m_deviceName + "." + propel].getBlobSize() == 0 )
    {
        return false;
    }

    return true;
}

bool acquisition::getBlobStr( const std::string &deviceName, const std::string &propel )
{
    if( m_roa.m_dictionary->count( deviceName + "." + propel ) == 0 )
    {
        return false;
    }

    if( ( ( *m_roa.m_dictionary )[deviceName + "." + propel].getBlobStr( m_blob, sizeof( m_blob ) ) ) ==
        sizeof( m_blob ) )
    {
        return false;
    }

    if( m_blob[0] == '\0' )
    {
        return false;
    }

    return true;
}

bool acquisition::getBlobStr( const std::string &propel )
{
    return getBlobStr( m_deviceName, propel );
}

template <>
int acquisition::getBlobVal<int>( const std::string &device, const std::string &propel, int defVal )
{
    if( getBlobStr( device, propel ) )
    {
        return atoi( m_blob );
    }
    else
    {
        return defVal;
    }
}

template <>
int acquisition::getBlobVal<int>( const std::string &propel, int defVal )
{
    if( getBlobStr( propel ) )
    {
        return atoi( m_blob );
    }
    else
    {
        return defVal;
    }
}

template <>
float acquisition::getBlobVal<float>( const std::string &propel, float defVal )
{
    if( getBlobStr( propel ) )
    {
        return strtod( m_blob, 0 );
    }
    else
    {
        return defVal;
    }
}

int acquisition::updateOverlay()
{
    if( !m_enabled )
        return 0;

    if( m_roa.m_dictionary == nullptr )
        return 0;

    if( m_roa.m_graphicsView == nullptr )
        return 0;

    // Get curr size
    m_width  = getBlobVal<int>( m_cameraName, "fg_frameSize.width", -1 );
    m_height = getBlobVal<int>( m_cameraName, "fg_frameSize.height", -1 );

    int rawStarCount = getBlobVal<int>( "num_stars.current", 0 );
    if( rawStarCount < 0 )
    {
        rawStarCount = 0;
    }

    size_t nstars = std::min( static_cast<size_t>( rawStarCount ), c_maxTrackedStars );

    m_nStars = nstars;

    if( m_width <= 0 || m_height <= 0 )
    {
        std::lock_guard<std::mutex> guard( m_starCircleMutex );

        for( size_t n = 0; n < m_starCircles.size(); ++n )
        {
            hideStarOverlay( m_starCircles[n], m_starLabels[n] );
        }

        return 0;
    }

    for( size_t n = 0; n < m_starCircles.size(); ++n )
    {
        std::lock_guard<std::mutex> guard( m_starCircleMutex );

        ensureStarOverlay( n );

        StretchCircle *sc = m_starCircles[n];
        QTextEdit     *te = m_starLabels[n];

        if( sc == nullptr || te == nullptr )
        {
            continue;
        }

        if( n >= m_nStars )
        {
            hideStarOverlay( sc, te );
            continue;
        }

        std::string star = "star_" + std::to_string( n );

        float x = getBlobVal<float>( star + ".x", -1 );
        float y = getBlobVal<float>( star + ".y", -1 );

        if( x >= 0 && y >= 0 )
        {
            // Move the circle
            float xc = x - 0.5 * ( m_circRad );
            float yc = ( m_height - y ) - 0.5 * ( m_circRad );

            sc->setRect( xc, yc, m_circRad, m_circRad );
            sc->setVisible( true );

            // Format the number
            char tmp[32];
            snprintf( tmp, sizeof( tmp ), "%ld", n );
            te->setText( tmp );

            // Set size based on the font size
            QFontMetrics fm( te->currentFont() );
            QSize        textSize = fm.size( 0, tmp );
            te->resize( textSize.width() + 5, textSize.height() + 5 );

            // Place the number
            // Take scene coordinates to viewport coordinates.
            QRectF sbr   = sc->sceneBoundingRect();
            float  qpf_x = sbr.x() + sc->rect().width() * 0.5 - sc->radius();
            float  qpf_y = sbr.y() + sc->rect().height() * 0.5 - sc->radius();
            QPoint qr    = m_roa.m_graphicsView->mapFromScene( QPointF( qpf_x, qpf_y ) );

            te->move( qr.x(), qr.y() );

            te->setVisible( true );
        }
        else
        {
            hideStarOverlay( sc, te );
        }
    }

    return 0;
}

void acquisition::keyPressEvent( QKeyEvent *ke )
{
    if( ke == nullptr || ke->text().isEmpty() )
    {
        return;
    }

    char key = ke->text()[0].toLatin1();

    if( key == 'A' )
    {
        if( m_enabled )
            disableOverlay();
        else
            enableOverlay();
    }
}

bool acquisition::overlayEnabled()
{
    return m_enabled;
}

void acquisition::enableOverlay()
{
    if( m_enableable == false )
    {
        return;
    }

    m_enabled = true;
}

void acquisition::disableOverlay()
{
    std::lock_guard<std::mutex> guard( m_starCircleMutex );

    for( size_t n = 0; n < m_nStars; ++n )
    {
        hideStarOverlay( m_starCircles[n], m_starLabels[n] );
    }

    m_enabled = false;
}

void acquisition::stretchCircleRemove( StretchCircle *sb )
{
    std::lock_guard<std::mutex> guard( m_starCircleMutex );

    for( size_t n = 0; n < m_starCircles.size(); ++n )
    {
        if( m_starCircles[n] == sb )
        {
            m_starCircles[n] = nullptr;
            return;
        }
    }
}

std::vector<std::string> acquisition::info()
{
    std::vector<std::string> vinfo;
    vinfo.push_back( "Acquisition overlay: " + m_deviceName );

    return vinfo;
}
