/** \file warnings.cpp
 * \brief Defines the warnings rtimv overlay plugin.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 */

#include "warnings.hpp"

#define errPrint( expl ) std::cerr << "warnings: " << __FILE__ << " " << __LINE__ << " " << expl << std::endl;

warnings::warnings() : rtimvOverlayInterface(), m_blob{ '\0' }
{
}

warnings::~warnings()
{
}

int warnings::attachOverlay( rtimvOverlayAccess &roa, mx::app::appConfigurator &config )
{
    m_roa = roa;

    config.configUnused( m_deviceName, mx::app::iniFile::makeKey( "rules", "device" ) );

    if( m_deviceName == "" )
    {
        pluginLogInfo( "not configured" );
        m_enableable = false;
        disableOverlay();
        return 1; // Tell rtimv to unload me since not configured.
    }

    pluginLogInfo( std::format( "enabling for {}", m_deviceName ) );

    config.configUnused( m_cautionKeys, mx::app::iniFile::makeKey( "rules", "cautions" ) );
    config.configUnused( m_warningKeys, mx::app::iniFile::makeKey( "rules", "warnings" ) );
    config.configUnused( m_alertKeys, mx::app::iniFile::makeKey( "rules", "alerts" ) );

    if( m_cautionKeys.size() == 0 && m_warningKeys.size() == 0 && m_alertKeys.size() == 0 )
    {
        m_enableable = false;
        disableOverlay();
        return 1;
    }

    connect( this,
             SIGNAL( warningLevel( rtimv::warningLevel ) ),
             m_roa.m_mainWindowObject,
             SLOT( borderWarningLevel( rtimv::warningLevel ) ) );
    connect( this,
             SIGNAL( textOverlayRefreshRequested( char ) ),
             m_roa.m_mainWindowObject,
             SLOT( textOverlayRefreshRequested( char ) ) );

    if( m_roa.m_dictionary != nullptr )
    {
        for( size_t n = 0; n < m_cautionKeys.size(); ++n )
        {
            ( *m_roa.m_dictionary )[m_deviceName + ".caution." + m_cautionKeys[n]].setBlob( nullptr, 0 );
        }

        for( size_t n = 0; n < m_warningKeys.size(); ++n )
        {
            ( *m_roa.m_dictionary )[m_deviceName + ".warning." + m_warningKeys[n]].setBlob( nullptr, 0 );
        }

        for( size_t n = 0; n < m_alertKeys.size(); ++n )
        {
            ( *m_roa.m_dictionary )[m_deviceName + ".alert." + m_alertKeys[n]].setBlob( nullptr, 0 );
        }
    }

    m_enableable = true;
    m_enabled    = true;
    enableOverlay();

    return 0;
}

int warnings::updateOverlay()
{
    if( !m_enabled )
        return 0;

    if( m_roa.m_dictionary == nullptr )
        return 0;

    if( m_roa.m_graphicsView == nullptr )
        return 0;

    std::vector<std::string> activeCautions = activeKeys( m_cautionKeys, ".caution." );
    std::vector<std::string> activeWarnings = activeKeys( m_warningKeys, ".warning." );
    std::vector<std::string> activeAlerts   = activeKeys( m_alertKeys, ".alert." );

    if( activeCautions != m_activeCautions || activeWarnings != m_activeWarnings || activeAlerts != m_activeAlerts )
    {
        m_activeCautions = std::move( activeCautions );
        m_activeWarnings = std::move( activeWarnings );
        m_activeAlerts   = std::move( activeAlerts );

        emit textOverlayRefreshRequested( textOverlayKey() );
    }

    bool caution = !m_activeCautions.empty();
    bool warn    = !m_activeWarnings.empty();
    bool alert   = !m_activeAlerts.empty();

    if( alert )
    {
        emit warningLevel( rtimv::warningLevel::alert );
    }
    else if( warn )
    {
        emit warningLevel( rtimv::warningLevel::warning );
    }
    else if( caution )
    {
        emit warningLevel( rtimv::warningLevel::caution );
    }
    else
    {
        emit warningLevel( rtimv::warningLevel::normal );
    }

    return 0;
}

void warnings::keyPressEvent( QKeyEvent *ke )
{
    static_cast<void>( ke );
}

bool warnings::hasTextOverlay()
{
    return true;
}

char warnings::textOverlayKey()
{
    return 'w';
}

std::string warnings::textOverlayTitle()
{
    return "warnings";
}

std::string warnings::textOverlayText()
{
    std::string text;
    text = "                     active warnings                     \n";
    text += "                  press 'w' to exit warnings            \n";
    text += "\n";

    appendActive( text, "Alerts", m_alertKeys, ".alert." );
    appendActive( text, "Warnings", m_warningKeys, ".warning." );
    appendActive( text, "Cautions", m_cautionKeys, ".caution." );

    return text;
}

bool warnings::overlayEnabled()
{
    return m_enabled;
}

void warnings::enableOverlay()
{
    if( m_enableable == false )
        return;

    m_enabled = true;
}

void warnings::disableOverlay()
{
    if( m_roa.m_graphicsView != nullptr )
    {
        for( size_t n = 0; n < m_roa.m_graphicsView->statusTextNo(); ++n )
        {
            m_roa.m_graphicsView->statusTextText( n, "" );
        }
    }

    m_activeCautions.clear();
    m_activeWarnings.clear();
    m_activeAlerts.clear();
    m_enabled = false;
}

std::vector<std::string> warnings::info()
{
    std::vector<std::string> vinfo;
    vinfo.push_back( "Warnings overlay: " + m_deviceName );
    /*if(m_deviceName != "")
    {
        vinfo.push_back("                   " + m_deviceName);
    }*/

    return vinfo;
}

bool warnings::keyOn( const std::string &key )
{
    if( m_roa.m_dictionary == nullptr )
        return false;

    if( ( ( *m_roa.m_dictionary )[key].getBlobStr( m_blob, sizeof( m_blob ) ) ) == sizeof( m_blob ) )
    {
        errPrint( "bad string" );
        return false;
    }

    return std::string( m_blob ) == "on";
}

bool warnings::anyOn( const std::vector<std::string> &keys, const std::string &prefix )
{
    for( size_t n = 0; n < keys.size(); ++n )
    {
        if( keyOn( m_deviceName + prefix + keys[n] ) )
            return true;
    }

    return false;
}

std::vector<std::string> warnings::activeKeys( const std::vector<std::string> &keys, const std::string &prefix )
{
    std::vector<std::string> active;

    for( size_t n = 0; n < keys.size(); ++n )
    {
        if( keyOn( m_deviceName + prefix + keys[n] ) )
        {
            active.push_back( keys[n] );
        }
    }

    return active;
}

void warnings::appendActive( std::string                    &text,
                             const std::string              &heading,
                             const std::vector<std::string> &keys,
                             const std::string              &prefix )
{
    bool any = false;

    for( size_t n = 0; n < keys.size(); ++n )
    {
        if( !keyOn( m_deviceName + prefix + keys[n] ) )
            continue;

        if( !any )
        {
            text += heading + ":\n";
            any = true;
        }

        text += "  " + keys[n] + "\n";
    }

    if( !any )
    {
        text += heading + ": none\n";
    }

    text += "\n";
}
