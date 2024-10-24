
#include "acquisition.hpp"

#define errPrint( expl ) std::cerr << "acquisition: " << __FILE__ << " " << __LINE__ << " " << expl << std::endl;

acquisition::acquisition() : rtimvOverlayInterface()
{
}

acquisition::~acquisition()
{
}

int acquisition::attachOverlay( rtimvOverlayAccess &roa, mx::app::appConfigurator &config )
{
    m_roa = roa;
    m_qgs = roa.m_graphicsView->scene();

    std::cerr << "acq\n";
    config.configUnused( m_deviceName, mx::app::iniFile::makeKey( "acquisition", "fitter" ) );

    if( m_deviceName == "" )
    {
        m_enableable = false;
        disableOverlay();
        return 1; // Tell rtimv to unload me since not configured.
    }

    config.configUnused( m_cameraName, mx::app::iniFile::makeKey( "acquisition", "camera" ) );
    config.configUnused( m_circRad, mx::app::iniFile::makeKey( "acquisition", "radius" ) );
    config.configUnused( m_color, mx::app::iniFile::makeKey( "acquisition", "color" ) );
    config.configUnused( m_fontSize, mx::app::iniFile::makeKey( "acquisition", "fontSize" ) );

    m_enableable = true;
    m_enabled = true;

    if( m_roa.m_dictionary != nullptr )
    {
        // Register these
        ( *m_roa.m_dictionary )[m_cameraName + ".fg_frameSize.width"].setBlob( nullptr, 0 );
        ( *m_roa.m_dictionary )[m_cameraName + ".fg_frameSize.height"].setBlob( nullptr, 0 );
        ( *m_roa.m_dictionary )[m_deviceName + ".num_stars.current"].setBlob( nullptr, 0 );
        /*( *m_roa.m_dictionary )[m_deviceName + ".star_0.x"].setBlob( nullptr, 0 );
        ( *m_roa.m_dictionary )[m_deviceName + ".star_0.y"].setBlob( nullptr, 0 );
        ( *m_roa.m_dictionary )[m_deviceName + ".star_1.x"].setBlob( nullptr, 0 );
        ( *m_roa.m_dictionary )[m_deviceName + ".star_1.y"].setBlob( nullptr, 0 );*/

        //(*m_roa.m_dictionary)[m_deviceName + ""].setBlob(nullptr, 0);
    }

    connect( this,
             SIGNAL( newStretchCircle( StretchCircle * ) ),
             m_roa.m_mainWindowObject,
             SLOT( addStretchCircle( StretchCircle * ) ) );

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

    std::string sstr;

    // Get curr size
    m_width = getBlobVal<int>( m_cameraName, "fg_frameSize.width", -1 );
    m_height = getBlobVal<int>( m_cameraName, "fg_frameSize.height", -1 );

    size_t nstars = getBlobVal<int>("num_stars.current", 0);

    if(m_nStars != nstars)
    {
        m_nStars = nstars;
        for( size_t n = 0; n < m_nStars; ++n )
        {
            std::string star = ".star_" + std::to_string( n );

            ( *m_roa.m_dictionary )[m_deviceName + star + ".x"].setBlob( nullptr, 0 );
            ( *m_roa.m_dictionary )[m_deviceName + star + ".y"].setBlob( nullptr, 0 );
            ( *m_roa.m_dictionary )[m_deviceName + star + ".peak"].setBlob( nullptr, 0 );
            ( *m_roa.m_dictionary )[m_deviceName + star + ".fwhm"].setBlob( nullptr, 0 );
        }

        std::lock_guard<std::mutex> guard( m_starCircleMutex );

        //deallocate circles
        for( size_t s = 0; s < m_starCircles.size(); ++s )
        {
            if( m_starCircles[s] != nullptr )
            {
                m_starCircles[s]->deleteLater();
            }
        }

        //deallocate labels
        for( size_t s = 0; s < m_starLabels.size(); ++s )
        {
            if( m_starLabels[s] != nullptr )
            {
                m_starLabels[s]->deleteLater();
            }
        }

        m_starCircles.resize( m_nStars, nullptr );
        m_starLabels.resize( m_nStars, nullptr );

        for( size_t n = 0; n < m_nStars; ++n )
        {
            //create circle
            m_starCircles[n] = new StretchCircle;
            m_starCircles[n]->setPenColor( m_color.c_str() );
            m_starCircles[n]->setPenWidth( 0 );
            m_starCircles[n]->setVisible( false );
            m_starCircles[n]->setStretchable( false );
            m_starCircles[n]->setRemovable( false );
            connect( m_starCircles[n],
                     SIGNAL( remove( StretchCircle * ) ),
                     this,
                     SLOT( stretchCircleRemove( StretchCircle * ) ) );
            emit newStretchCircle( m_starCircles[n] );

            //create label
            m_starLabels[n] = new QTextEdit( m_roa.m_graphicsView );
            QFont qf;
            qf = m_starLabels[n]->currentFont();
            qf.setPixelSize( m_fontSize );
            m_starLabels[n]->setCurrentFont( qf );
            m_starLabels[n]->setVisible( false );
            m_starLabels[n]->setTextColor( m_color.c_str() );
            m_roa.m_graphicsView->textEditSetup(m_starLabels[n]);
        }
    }

    std::vector<float> xs( m_nStars, -1 );
    std::vector<float> ys( m_nStars, -1 );

    for( size_t n = 0; n < m_nStars; ++n )
    {
        std::string star = "star_" + std::to_string( n );

        xs[n] = getBlobVal<float>( star + ".x", -1 );
        ys[n] = getBlobVal<float>( star + ".y", -1 );

        //Now for each valid star position, set up the overlay
        if( xs[n] >= 0 && ys[n] >= 0)
        {
            std::lock_guard<std::mutex> guard( m_starCircleMutex );

            StretchCircle *sc = m_starCircles[n]; //Just for convenience
            QTextEdit * te = m_starLabels[n];

            //Move the circle
            float xc = xs[n] - 0.5 * ( m_circRad );
            float yc = ( m_height - ys[n] ) - 0.5 * ( m_circRad );

            sc->setRect( xc, yc, m_circRad, m_circRad );
            sc->setVisible( true );

            //Format the number
            char tmp[32];
            snprintf( tmp, sizeof( tmp ), "%ld", n );
            te->setText( tmp );

            //Set size based on the font size
            QFontMetrics fm(te->currentFont());
            QSize textSize = fm.size(0, tmp);
            te->resize( textSize.width()+5,textSize.height()+5 );

            //Place the number
            //Take scene coordinates to viewport coordinates.
            QRectF sbr = sc->sceneBoundingRect();
            float qpf_x = sbr.x() + sc->rect().width() * 0.5 - sc->radius();
            float qpf_y = sbr.y() + sc->rect().height() * 0.5 - sc->radius();
            QPoint qr = m_roa.m_graphicsView->mapFromScene(QPointF( qpf_x , qpf_y ));

            te->move( qr.x(), qr.y() );

            te->setVisible( true );
        }
    }

    return 0;
}

void acquisition::keyPressEvent( QKeyEvent *ke )
{
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
    for( size_t n = 0; n < m_nStars; ++n )
    {
        if(m_starCircles[n] != nullptr)
        {
            m_starCircles[n]->setVisible(false);
        }
        if(m_starLabels[n] != nullptr)
        {
            m_starLabels[n]->setVisible(false);
        }

    }

    m_enabled = false;
}

void acquisition::stretchCircleRemove( StretchCircle *sb )
{
    static_cast<void>( sb );
    /*
    std::cerr << "acquisition::stretchBoxRemove 1\n";
    std::lock_guard<std::mutex> guard(m_roiBoxMutex);
    if(!m_roiBox)
    {
        return;
    }

    std::cerr << "acquisition::stretchBoxRemove 2\n";

    if(sb != m_roiBox)
    {
        return;
    }

    std::cerr << "acquisition::stretchBoxRemove 3\n";

    m_roiBox = nullptr;
    */
}

std::vector<std::string> acquisition::info()
{
    std::vector<std::string> vinfo;
    vinfo.push_back( "Acquisition overlay: " + m_deviceName );

    return vinfo;
}
