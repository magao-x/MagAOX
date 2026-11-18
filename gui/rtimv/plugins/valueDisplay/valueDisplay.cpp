/** \file valueDisplay.cpp
 * \brief Implements the two-value INDI text overlay for rtimv.
 */

#include "valueDisplay.hpp"

#include <QEvent>
#include <QTextDocument>

#include <algorithm>
#include <cmath>
#include <cstring>

valueDisplay::valueDisplay() = default;

valueDisplay::~valueDisplay()
{
    detachOverlay();
}

void valueDisplay::detachOverlay()
{
    m_enabled = m_enableable = false;

    if( m_view )
    {
        m_view->removeEventFilter( this );
        m_view->viewport()->removeEventFilter( this );
        m_view->globalStatsText()->removeEventFilter( this );
        m_view->saveBox()->removeEventFilter( this );
    }

    delete m_text.data();
    m_view       = nullptr;
    m_dictionary = nullptr;
}

int valueDisplay::attachOverlay( rtimvOverlayAccess &access, mx::app::appConfigurator &config )
{
    detachOverlay();
    m_values        = {};
    bool configured = false;

    for( size_t n = 0; n < m_values.size(); ++n )
    {
        auto      &value  = m_values[n];
        const auto suffix = std::to_string( n + 1 );
        config.configUnused( value.m_key, "valueDisplay", "key" + suffix );

        if( value.m_key.empty() )
        {
            continue;
        }

        const auto  parts = QString::fromStdString( value.m_key ).split( '.' );
        std::string label;
        std::string precision = "3";
        config.configUnused( label, "valueDisplay", "label" + suffix );
        config.configUnused( precision, "valueDisplay", "precision" + suffix );
        bool validPrecision = false;
        value.m_precision   = QString::fromStdString( precision ).toInt( &validPrecision );

        if( parts.size() != 3 || parts[0].trimmed().isEmpty() || parts[1].trimmed().isEmpty() ||
            parts[2].trimmed().isEmpty() || !validPrecision || value.m_precision < 0 || value.m_precision > 15 )
        {
            pluginLogError( "valueDisplay requires device.property.element keys and integer precision from 0 to 15" );
            return -1;
        }

        value.m_label = label.empty() ? parts[2] : QString::fromStdString( label );
        configured    = true;
    }

    if( !configured )
    {
        return 1;
    }

    if( !access.m_graphicsView )
    {
        return -1;
    }

    m_view       = access.m_graphicsView;
    m_dictionary = access.m_dictionary;

    if( m_dictionary )
    {
        for( const auto &value : m_values )
        {
            if( !value.m_key.empty() )
            {
                m_dictionary->try_emplace( value.m_key );
            }
        }
    }

    m_text = new QTextEdit( m_view );
    m_text->setObjectName( "valueDisplayText" );
    m_view->textEditSetup( m_text );
    m_text->setLineWrapMode( QTextEdit::NoWrap );

    QFont font( m_view->statusTextFontFamily() );
    font.setPixelSize( std::max( 1, static_cast<int>( m_view->statusTextFontSize() ) ) );
    m_text->setCurrentFont( font );
    m_text->document()->setDefaultFont( font );
    m_text->setTextColor( QColor( m_view->statusTextFontColor() ) );
    m_text->setTextBackgroundColor( QColor( 0, 0, 0, 150 ) );
    m_text->stackUnder( m_view->helpText() );

    m_view->installEventFilter( this );
    m_view->viewport()->installEventFilter( this );
    m_view->globalStatsText()->installEventFilter( this );
    m_view->saveBox()->installEventFilter( this );
    m_enableable = true;
    enableOverlay();
    return 0;
}

QString valueDisplay::valueText( const valueSlot &value )
{
    if( !m_dictionary )
    {
        return {};
    }

    auto entry = m_dictionary->find( value.m_key );

    if( entry == m_dictionary->end() )
    {
        return {};
    }

    char         blob[512];
    const size_t copied = entry->second.getBlob( blob, sizeof( blob ) );

    // Read the original terminator too: getBlobStr() would replace an unterminated final byte.
    if( copied <= 1 || copied == sizeof( blob ) || blob[copied - 1] != '\0' || std::memchr( blob, '\0', copied - 1 ) )
    {
        return {};
    }

    bool         valid  = false;
    const double number = QString::fromLatin1( blob, static_cast<int>( copied - 1 ) ).toDouble( &valid );

    if( !valid || !std::isfinite( number ) )
    {
        return {};
    }

    return QString::number( number, 'f', value.m_precision );
}

int valueDisplay::updateOverlay()
{
    if( !m_enabled || !m_view || !m_text )
    {
        return 0;
    }

    QStringList rows;

    for( const auto &value : m_values )
    {
        if( !value.m_key.empty() )
        {
            rows.push_back( value.m_label + ": " + valueText( value ) );
        }
    }

    const QString text = rows.join( '\n' );

    if( m_text->toPlainText() != text )
    {
        m_text->setPlainText( text );
    }

    layoutText();
    return 0;
}

void valueDisplay::layoutText()
{
    if( !m_enabled || !m_view || !m_text )
    {
        return;
    }

    constexpr int gap    = 6;
    const QRect   bounds = m_view->viewport()->geometry().adjusted( gap, gap, -gap, -gap );
    int           top    = std::max( bounds.top(), m_view->saveBox()->geometry().bottom() + gap );
    const auto   *stats  = m_view->globalStatsText();

    if( !stats->isHidden() )
    {
        top = std::max( top, stats->geometry().bottom() + gap );
    }

    const int height = bounds.bottom() - top + 1;

    if( bounds.width() <= 0 || height <= 0 )
    {
        m_text->hide();
        return;
    }

    const int   padding  = static_cast<int>( std::ceil( 2 * m_text->document()->documentMargin() ) ) + 2;
    const QSize textSize = QFontMetrics( m_text->currentFont() ).size( 0, m_text->toPlainText() );
    m_text->setGeometry( bounds.left(),
                         top,
                         std::min( bounds.width(), textSize.width() + padding ),
                         std::min( height, textSize.height() + padding ) );
    m_text->show();
}

bool valueDisplay::eventFilter( QObject *watched, QEvent *event )
{
    switch( event->type() )
    {
    case QEvent::Resize:
    case QEvent::Move:
    case QEvent::Show:
    case QEvent::Hide:
        layoutText();
        break;
    default:
        break;
    }

    return QObject::eventFilter( watched, event );
}

void valueDisplay::keyPressEvent( QKeyEvent *event )
{
    static_cast<void>( event );
}

bool valueDisplay::overlayEnabled()
{
    return m_enabled;
}

void valueDisplay::enableOverlay()
{
    if( m_enableable && m_view && m_text )
    {
        m_enabled = true;
        updateOverlay();
    }
}

void valueDisplay::disableOverlay()
{
    m_enabled = false;

    if( m_text )
    {
        m_text->hide();
    }
}

std::vector<std::string> valueDisplay::info()
{
    std::vector<std::string> result{ "Value display overlay" };

    for( const auto &value : m_values )
    {
        if( !value.m_key.empty() )
        {
            result.push_back( "  " + value.m_key );
        }
    }

    return result;
}
