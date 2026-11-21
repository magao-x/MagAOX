/** \file timingPanel.cpp
 * \brief Lay out and connect the timingGUI operator widgets.
 */
#include <cmath>
#include <limits>
#include <string_view>

#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>

#include "timingPanel.hpp"
#include "../xWidgets/statusEntry.hpp"
#include "../xWidgets/statusLabel.hpp"
#include "../xWidgets/toggleSlider.hpp"

namespace xqt
{

namespace
{
/// Decode an available timing value, preserving zero jitter and rejecting invalid sentinels.
std::optional<double> readTimingNumber( const pcf::IndiElement &element /**< [in] Timing value from INDI. */ );

std::optional<double> readTimingNumber( const pcf::IndiElement &element )
{
    bool         ok    = false;
    const double value = QString::fromStdString( element.get() ).toDouble( &ok );

    if( ok && std::isfinite( value ) && value >= 0 )
    {
        return value;
    }

    return std::nullopt;
}
} // namespace

timingPanel::timingPanel( QWidget *parent ) : xWidget( parent )
{
    setWindowTitle( QStringLiteral( "timingGUI" ) );
    setStyleSheet( QStringLiteral( "QGroupBox { border: none; }" ) );
    auto *layout = new QVBoxLayout( this );

    auto *timing = new QGroupBox( QStringLiteral( "Timing" ), this );
    auto *grid   = new QGridLayout( timing );
    setXwFont( timing );
    constexpr float timingFontScale = 1.2F;
    grid->setColumnMinimumWidth( 1, 85 );
    grid->setColumnMinimumWidth( 3, 85 );
    grid->setColumnStretch( 5, 1 );

    for( size_t index = 0; index < m_measurements.size(); ++index )
    {
        auto     &measurement = m_measurements[index];
        const int row         = int( index );
        auto     *label       = new QLabel( QString::fromLatin1( measurement.m_label ), timing );
        setXwFont( label, timingFontScale );
        grid->addWidget( label, row, 0 );

        measurement.m_value = new statusLabel( timing );
        setXwFont( measurement.m_value, timingFontScale );
        measurement.m_value->setAlignment( Qt::AlignRight | Qt::AlignVCenter );
        const QString device = QString::fromLatin1( measurement.m_device );
        measurement.m_value->setObjectName( device + QStringLiteral( ".value" ) );

        const bool latency = measurement.m_kind == MeasurementKind::latency;
        auto      *units   = new QLabel( latency ? QStringLiteral( "us" ) : QStringLiteral( "FPS" ), timing );
        setXwFont( units, timingFontScale );
        grid->addWidget( measurement.m_value, row, 1 );
        grid->addWidget( units, row, 2 );

        if( measurement.m_jitterElement )
        {
            measurement.m_jitter = new statusLabel( timing );
            setXwFont( measurement.m_jitter, timingFontScale );
            measurement.m_jitter->setAlignment( Qt::AlignRight | Qt::AlignVCenter );
            measurement.m_jitter->setObjectName( device + QStringLiteral( ".jitter" ) );
            auto *jitterUnits = new QLabel( QStringLiteral( "us rms" ), timing );
            setXwFont( jitterUnits, timingFontScale );
            const auto jitterDescription = QStringLiteral( "Jitter (standard deviation)" );
            measurement.m_jitter->setToolTip( jitterDescription );
            jitterUnits->setToolTip( jitterDescription );
            grid->addWidget( measurement.m_jitter, row, 3 );
            grid->addWidget( jitterUnits, row, 4 );
        }
    }

    layout->addWidget( timing );

    auto *synco       = new QGroupBox( QStringLiteral( "dmtweeter-synco" ), this );
    auto *syncoLayout = new QVBoxLayout( synco );
    setXwFont( synco );

    auto *syncoToggle = new QHBoxLayout;
    syncoLayout->addLayout( syncoToggle );
    syncoToggle->addStretch( 1 );
    addToggle( syncoToggle, "dmtweeter-synco", "enable", "Enabled", true );
    syncoToggle->addStretch( 1 );

    m_syncoStatus = new statusLabel( synco );
    m_syncoStatus->setObjectName( QStringLiteral( "dmtweeter-synco.status" ) );
    m_syncoStatus->setAlignment( Qt::AlignCenter );
    m_syncoStatus->setTextFormat( Qt::PlainText );
    syncoLayout->addWidget( m_syncoStatus );

    m_controls.push_back( { "dmtweeter-synco",
                            "health",
                            nullptr,
                            [this]( const pcf::IndiProperty &update )
                            {
                                if( update.getType() == pcf::IndiProperty::Text && update.find( "reason" ) )
                                {
                                    m_syncoReason = QString::fromStdString( update["reason"].get() ).trimmed();
                                    updateSyncoStatus();
                                }
                            },
                            [this]()
                            {
                                m_syncoReason.clear();
                                updateSyncoStatus();
                            } } );

    auto *syncoActions = new QHBoxLayout;
    syncoLayout->addLayout( syncoActions );
    addRequest( syncoActions, "dmtweeter-synco", "reacquire", "Reacquire", "Reset the synco timing estimate" );
    addRequest(
        syncoActions, "dmtweeter-synco", "zero", "Zero Channels", "Requires an open AO loop and disabled synco" );
    layout->addWidget( synco );

    auto *modulators = new QHBoxLayout;
    addModulator( modulators, "tweeterSpeck" );
    addModulator( modulators, "tweeterMod" );
    layout->addLayout( modulators );

    onDisconnect();
}

timingPanel::~timingPanel() noexcept
{
    detachSubscribersRecursive();
}

void timingPanel::addToggle(
    QBoxLayout *layout, const char *device, const char *property, const char *label, bool syncoEnable )
{
    auto *slider = new toggleSlider( device, property, label, layout->parentWidget() );
    slider->setObjectName(
        QStringLiteral( "%1.%2" ).arg( QString::fromLatin1( device ), QString::fromLatin1( property ) ) );
    slider->setStretch( 0, 0, 1, true );
    slider->setMaximumWidth( 230 );
    layout->addWidget( slider, 4 );

    m_controls.push_back( { device,
                            property,
                            slider,
                            [this, slider, syncoEnable]( const pcf::IndiProperty &update )
                            {
                                if( update.getType() != pcf::IndiProperty::Switch || !update.find( "toggle" ) )
                                {
                                    return;
                                }

                                const auto toggle = update["toggle"].getSwitchState();
                                slider->setEnabled( toggle == pcf::IndiElement::On || toggle == pcf::IndiElement::Off );

                                if( syncoEnable )
                                {
                                    setSyncoEnable( toggle );
                                }

                                if( syncoEnable && toggle == pcf::IndiElement::On &&
                                    update.getState() == pcf::IndiProperty::Busy )
                                {
                                    auto state = update;
                                    state.setState( pcf::IndiProperty::Ok );
                                    slider->handleSetProperty( state );
                                }
                                else
                                {
                                    slider->handleSetProperty( update );
                                }
                            },
                            [this, slider, syncoEnable]()
                            {
                                slider->onDisconnect();
                                slider->setEnabled( false );

                                if( syncoEnable )
                                {
                                    setSyncoEnable( pcf::IndiElement::UnknownSwitchState );
                                }
                            } } );
}

void timingPanel::addEntry(
    QBoxLayout *layout, const char *device, const char *property, const char *label, const char *units, int type )
{
    auto *entry = new statusEntry( device, property, type, label, units, layout->parentWidget() );
    entry->setObjectName(
        QStringLiteral( "%1.%2" ).arg( QString::fromLatin1( device ), QString::fromLatin1( property ) ) );
    entry->setStretch( 0, 2, 3 );
    layout->addWidget( entry );

    m_controls.push_back( { device,
                            property,
                            entry,
                            [entry]( const pcf::IndiProperty &update )
                            {
                                if( update.getType() == pcf::IndiProperty::Number )
                                {
                                    if( update.find( "current" ) )
                                    {
                                        entry->setEnabled( true );
                                    }

                                    entry->handleSetProperty( update );
                                }
                            },
                            [entry]()
                            {
                                entry->onDisconnect();
                                entry->setEnabled( false );
                            } } );
}

void timingPanel::addRequest(
    QBoxLayout *layout, const char *device, const char *property, const char *label, const char *tooltip )
{
    auto *button = new QPushButton( QString::fromLatin1( label ), layout->parentWidget() );
    button->setObjectName(
        QStringLiteral( "%1.%2" ).arg( QString::fromLatin1( device ), QString::fromLatin1( property ) ) );
    button->setToolTip( QString::fromLatin1( tooltip ) );
    setXwFont( button );
    layout->addWidget( button );

    connect( button,
             &QPushButton::clicked,
             this,
             [this, button, device, property]()
             {
                 if( button->isEnabled() )
                 {
                     pcf::IndiProperty request( pcf::IndiProperty::Switch, device, property );
                     request.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );
                     sendNewProperty( request );
                 }
             } );

    m_controls.push_back( { device,
                            property,
                            nullptr,
                            [button, label]( const pcf::IndiProperty &update )
                            {
                                const auto state = update.getState();
                                button->setEnabled( update.getType() == pcf::IndiProperty::Switch &&
                                                    update.find( "request" ) && state != pcf::IndiProperty::Busy );
                                QString text = QString::fromLatin1( label );

                                if( state == pcf::IndiProperty::Alert )
                                {
                                    text += QStringLiteral( " (failed)" );
                                }

                                button->setText( text );
                            },
                            [button, label]()
                            {
                                button->setText( QString::fromLatin1( label ) );
                                button->setEnabled( false );
                            } } );
}

void timingPanel::addModulator( QBoxLayout *layout, const char *device )
{
    auto *box      = new QGroupBox( QString::fromLatin1( device ), this );
    auto *controls = new QVBoxLayout( box );
    setXwFont( box );
    addEntry( controls, device, "amp", "Amplitude", "", statusEntry::FLOAT );
    addEntry( controls, device, "frequency", "Frequency", "Hz", statusEntry::FLOAT );

    if( std::string_view( device ) == "tweeterSpeck" )
    {
        addEntry( controls, device, "angle", "Angle", "deg", statusEntry::FLOAT );
    }
    else
    {
        addFileReadback( controls, device );
    }

    auto *modulation = new QHBoxLayout;
    controls->addLayout( modulation );
    addToggle( modulation, device, "modulating", "Modulating" );
    modulation->addStretch( 1 );
    addRequest( controls, device, "zero", "Zero", "Zero the contribution after stopping modulation" );
    layout->addWidget( box );
}

void timingPanel::addFileReadback( QBoxLayout *layout, const char *device )
{
    auto *label = new QLabel( QStringLiteral( "File" ), layout->parentWidget() );
    auto *file  = new statusLabel( layout->parentWidget() );
    setXwFont( label );
    file->setObjectName( QString::fromLatin1( device ) + QStringLiteral( ".file" ) );
    file->setTextFormat( Qt::PlainText );
    file->setSizePolicy( QSizePolicy::Ignored, QSizePolicy::Preferred );

    auto *row = new QHBoxLayout;
    row->setContentsMargins( 9, 9, 9, 9 );
    row->addWidget( label );
    row->addWidget( file, 1 );
    layout->addLayout( row );

    m_controls.push_back( { device,
                            "file",
                            nullptr,
                            [file]( const pcf::IndiProperty &update )
                            {
                                if( update.getType() == pcf::IndiProperty::Text && update.find( "current" ) )
                                {
                                    const auto path = QString::fromStdString( update["current"].get() );
                                    const auto name = QFileInfo( path ).fileName();
                                    file->setText( name.isEmpty() ? QStringLiteral( "---" ) : name );
                                    file->setToolTip( path );
                                }
                            },
                            [file]()
                            {
                                file->setText( QStringLiteral( "---" ) );
                                file->setToolTip( QString() );
                            } } );
}

void timingPanel::subscribe()
{
    if( !m_parent )
    {
        return;
    }

    for( const auto &control : m_controls )
    {
        if( control.m_widget )
        {
            registerSubscriber( control.m_widget );
        }

        m_parent->addSubscriberProperty( this, control.m_device, control.m_property );
    }

    for( const auto &measurement : m_measurements )
    {
        m_parent->addSubscriberProperty( this, measurement.m_device, measurement.m_property );
    }
}

void timingPanel::onDisconnect()
{
    for( const auto &control : m_controls )
    {
        control.m_clear();
    }

    for( auto &measurement : m_measurements )
    {
        clearMeasurement( measurement );
    }
}

void timingPanel::handleDefProperty( const pcf::IndiProperty &property )
{
    handleSetProperty( property );
}

void timingPanel::handleDelProperty( const pcf::IndiProperty &property )
{
    const auto removed = [&]( const char *device, const char *name )
    {
        return property.getDevice() == device &&
               ( property.getName().empty() || property.getName() == "fsm" || property.getName() == name );
    };

    for( const auto &control : m_controls )
    {
        if( removed( control.m_device, control.m_property ) )
        {
            control.m_clear();
        }
    }

    for( auto &measurement : m_measurements )
    {
        if( removed( measurement.m_device, measurement.m_property ) )
        {
            clearMeasurement( measurement );
        }
    }
}

void timingPanel::handleSetProperty( const pcf::IndiProperty &property )
{
    for( const auto &control : m_controls )
    {
        if( property.getDevice() == control.m_device && property.getName() == control.m_property )
        {
            control.m_update( property );
            return;
        }
    }

    for( auto &measurement : m_measurements )
    {
        if( property.getDevice() != measurement.m_device || property.getName() != measurement.m_property )
        {
            continue;
        }

        if( property.getType() != pcf::IndiProperty::Number || property.getState() == pcf::IndiProperty::Alert )
        {
            clearMeasurement( measurement );
            return;
        }

        if( measurement.m_kind == MeasurementKind::syncoRate && property.find( "active" ) )
        {
            const double steps = readTimingNumber( property["active"] ).value_or( 0 );
            m_syncoSteps = steps <= std::numeric_limits<int>::max() && steps == std::trunc( steps ) ? int( steps ) : 0;
            updateSyncoStatus();
        }

        const bool latency = measurement.m_kind == MeasurementKind::latency;
        if( latency && property.find( "valid" ) )
        {
            measurement.m_valid = readTimingNumber( property["valid"] ) == 1.0;

            if( !measurement.m_valid )
            {
                clearMeasurement( measurement );
                return;
            }
        }

        if( property.find( measurement.m_valueElement ) )
        {
            measurement.m_lastValue = readTimingNumber( property[measurement.m_valueElement] );

            if( measurement.m_kind == MeasurementKind::rate && measurement.m_lastValue.value_or( 0 ) <= 0 )
            {
                clearMeasurement( measurement );
                return;
            }
        }

        if( measurement.m_jitterElement && property.find( measurement.m_jitterElement ) )
        {
            measurement.m_lastJitter = readTimingNumber( property[measurement.m_jitterElement] );
        }

        showMeasurement( measurement );
        return;
    }
}

void timingPanel::showMeasurement( const Measurement &measurement )
{
    const QString missing   = QStringLiteral( "---" );
    const bool    latency   = measurement.m_kind == MeasurementKind::latency;
    const bool    available = latency ? measurement.m_valid : measurement.m_lastValue.has_value();
    const bool    valid =
        available && ( measurement.m_kind != MeasurementKind::syncoRate || m_syncoEnable == pcf::IndiElement::On );
    const double scale = latency ? 1e6 : 1;
    measurement.m_value->setText(
        valid && measurement.m_lastValue ? QString::number( *measurement.m_lastValue * scale, 'f', 2 ) : missing );

    if( measurement.m_jitter )
    {
        measurement.m_jitter->setText(
            valid && measurement.m_lastJitter ? QString::number( *measurement.m_lastJitter * 1e6, 'f', 2 ) : missing );
    }
}

void timingPanel::setSyncoEnable( pcf::IndiElement::SwitchStateType state )
{
    m_syncoEnable = state;

    for( const auto &measurement : m_measurements )
    {
        if( measurement.m_kind == MeasurementKind::syncoRate )
        {
            showMeasurement( measurement );
        }
    }

    updateSyncoStatus();
}

void timingPanel::updateSyncoStatus()
{
    QString text = QStringLiteral( "---" );

    if( m_syncoEnable == pcf::IndiElement::Off )
    {
        text = QStringLiteral( "Disabled" );
    }
    else if( m_syncoEnable == pcf::IndiElement::On )
    {
        text    = m_syncoReason.isEmpty() ? QStringLiteral( "Enabled" ) : m_syncoReason;
        text[0] = text.at( 0 ).toUpper();

        if( m_syncoSteps > 0 )
        {
            text += QStringLiteral( " - %1 %2" )
                        .arg( m_syncoSteps )
                        .arg( m_syncoSteps == 1 ? QStringLiteral( "step" ) : QStringLiteral( "steps" ) );
        }
    }

    m_syncoStatus->setText( text );
}

void timingPanel::clearMeasurement( Measurement &measurement )
{
    measurement.m_lastValue.reset();
    measurement.m_lastJitter.reset();
    measurement.m_valid = false;
    showMeasurement( measurement );

    if( measurement.m_kind == MeasurementKind::syncoRate )
    {
        m_syncoSteps = 0;
        updateSyncoStatus();
    }
}

} // namespace xqt
