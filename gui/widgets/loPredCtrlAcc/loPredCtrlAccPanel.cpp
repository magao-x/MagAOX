/** \file loPredCtrlAccPanel.cpp
 * \brief Lay out, route and gate the loPredCtrlAccGUI operator widgets.
 */
#include <algorithm>
#include <cmath>
#include <cstring>

#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>

#include "loPredCtrlAccPanel.hpp"
#include "../xWidgets/statusEntry.hpp"
#include "../xWidgets/statusLabel.hpp"
#include "../xWidgets/toggleSlider.hpp"

namespace xqt
{

namespace
{
const QString c_missing = QStringLiteral( "---" ); ///< Display of an unavailable readback.

/// Parse a finite number, or return false.
bool finiteNumber( const std::string &text, /**< [in] INDI value text. */
                   double            &number /**< [out] Parsed value. */ );

bool finiteNumber( const std::string &text, double &number )
{
    bool ok = false;
    number  = QString::fromStdString( text ).trimmed().toDouble( &ok );
    return ok && std::isfinite( number );
}

/// Operator-facing name of a ddspc_mode element.
QString modeName( const std::string &mode /**< [in] Element name. */ );

QString modeName( const std::string &mode )
{
    if( mode == "standard" )
    {
        return QStringLiteral( "Standard" );
    }

    if( mode == "accel_assisted" )
    {
        return QStringLiteral( "ACC Assisted" );
    }

    return mode.empty() ? c_missing : QString::fromStdString( mode );
}
} // namespace

loPredCtrlAccPanel::loPredCtrlAccPanel( const std::string &device, QWidget *parent )
    : xWidget( parent ), m_device( device ), m_clock(
                                                 []
                                                 {
                                                     return std::chrono::steady_clock::now();
                                                 } )
{
    m_confirmReset = [this]()
    {
        return QMessageBox::question( this,
                                      QStringLiteral( "Reset DDSPC models" ),
                                      QStringLiteral( "Reset both the standard and ACC-assisted models?\n\nAll learned "
                                                      "coefficients are discarded. A model must be retrained before "
                                                      "it can predict." ),
                                      QMessageBox::Yes | QMessageBox::Cancel,
                                      QMessageBox::Cancel ) == QMessageBox::Yes;
    };

    setWindowTitle( QString::fromStdString( m_device ) );
    setStyleSheet( QStringLiteral( "QGroupBox { font-weight: bold; }" ) );
    auto *top      = new QHBoxLayout( this );
    auto *controls = new QVBoxLayout;
    top->addLayout( controls, 1 );

    m_banner = new QLabel( this );
    m_banner->setObjectName( QStringLiteral( "prediction.banner" ) );
    m_banner->setAlignment( Qt::AlignCenter );
    m_banner->setWordWrap( true );
    m_banner->setMargin( 6 );
    setXwFont( m_banner, 1.3F );
    controls->addWidget( m_banner );

    auto *ttWarning = new QLabel( QStringLiteral( "Main-loop tip/tilt is always zero: loPredCtrlAcc publishes modes 0 "
                                                  "and 1 as zero to the AO loop. Turning prediction off does not "
                                                  "restore main-loop tip/tilt correction." ),
                                  this );
    ttWarning->setObjectName( QStringLiteral( "ttWarning" ) );
    ttWarning->setWordWrap( true );
    ttWarning->setStyleSheet( QStringLiteral( "color: #e0a030;" ) );
    setXwFont( ttWarning, 0.9F );
    controls->addWidget( ttWarning );

    auto *modeBox    = new QGroupBox( QStringLiteral( "Controller mode" ), this );
    auto *modeLayout = new QVBoxLayout( modeBox );
    auto *modeRow    = new QHBoxLayout;
    m_standard       = new QPushButton( QStringLiteral( "Standard" ), modeBox );
    m_assisted       = new QPushButton( QStringLiteral( "ACC Assisted" ), modeBox );
    m_standard->setObjectName( QStringLiteral( "ddspc_mode.standard" ) );
    m_assisted->setObjectName( QStringLiteral( "ddspc_mode.accel_assisted" ) );

    for( auto *button : { m_standard, m_assisted } )
    {
        button->setCheckable( true );
        setXwFont( button );
        modeRow->addWidget( button );
    }

    connect( m_standard,
             &QPushButton::clicked,
             this,
             [this]()
             {
                 sendMode( "standard" );
             } );
    connect( m_assisted,
             &QPushButton::clicked,
             this,
             [this]()
             {
                 sendMode( "accel_assisted" );
             } );
    modeLayout->addLayout( modeRow );
    m_interlock = new QLabel( modeBox );
    m_interlock->setObjectName( QStringLiteral( "interlock" ) );
    m_interlock->setWordWrap( true );
    setXwFont( m_interlock, 0.9F );
    modeLayout->addWidget( m_interlock );
    controls->addWidget( modeBox );

    auto *learnBox    = new QGroupBox( QStringLiteral( "Learning" ), this );
    auto *learnLayout = new QVBoxLayout( learnBox );
    m_learnStd        = addToggle( learnLayout,
                            "learn_std",
                            "Standard sequence",
                            "Original three-stage exploration. Turning it on also enables learning and prediction, "
                                   "so probes and feedback reach the DM." );
    m_learn = addToggle( learnLayout, "learn", "Learning", "Update the selected model; the other model stays frozen." );
    m_reset = new QPushButton( QStringLiteral( "Reset models..." ), learnBox );
    m_reset->setObjectName( QStringLiteral( "reset_model" ) );
    setXwFont( m_reset );
    connect( m_reset,
             &QPushButton::clicked,
             this,
             [this]()
             {
                 requestReset();
             } );
    learnLayout->addWidget( m_reset );
    controls->addWidget( learnBox );

    auto *predictBox    = new QGroupBox( QStringLiteral( "Prediction" ), this );
    auto *predictLayout = new QVBoxLayout( predictBox );
    m_predict           = addToggle( predictLayout,
                           "predict",
                           "Prediction",
                           "Apply DDSPC tip/tilt through the auxiliary DM channel. Off removes the contribution and "
                                     "waits for its zero to be confirmed." );
    m_selectedReady     = new QLabel( predictBox );
    m_selectedReady->setObjectName( QStringLiteral( "selectedReady" ) );
    setXwFont( m_selectedReady );
    predictLayout->addWidget( m_selectedReady );
    controls->addWidget( predictBox );

    auto *gainBox    = new QGroupBox( QStringLiteral( "Feedback gain" ), this );
    auto *gainLayout = new QVBoxLayout( gainBox );
    m_gainEntry      = new statusEntry( m_device, "gain", statusEntry::FLOAT, "Gain", "", gainBox );
    m_gainEntry->setObjectName( QStringLiteral( "gain.entry" ) );
    m_gainEntry->setStretch( 0, 1, 2 );
    m_gainEntry->setToolTip( QStringLiteral( "Added current-WFS feedback gain*y. Enter a target and press Return." ) );
    gainLayout->addWidget( m_gainEntry );
    m_gainSlider = new QSlider( Qt::Horizontal, gainBox );
    m_gainSlider->setObjectName( QStringLiteral( "gain.slider" ) );
    m_gainSlider->setToolTip( QStringLiteral( "Range and step from the gain property definition." ) );
    gainLayout->addWidget( m_gainSlider );
    connect( m_gainSlider,
             &QSlider::sliderMoved,
             this,
             [this]( int position )
             {
                 if( m_gainLimits )
                 {
                     m_gainEntry->setEditText( QString::number( m_gainMin + position * m_gainStep, 'f', 4 ) );
                 }
             } );
    connect( m_gainSlider,
             &QSlider::sliderReleased,
             this,
             [this]()
             {
                 sendGainFromSlider();
             } );
    controls->addWidget( gainBox );

    m_advancedButton = new QPushButton( QStringLiteral( "Advanced ▸" ), this );
    m_advancedButton->setObjectName( QStringLiteral( "advanced.toggle" ) );
    m_advancedButton->setCheckable( true );
    m_advancedButton->setFlat( true );
    controls->addWidget( m_advancedButton );
    m_advanced           = new QWidget( this );
    auto *advancedLayout = new QVBoxLayout( m_advanced );
    advancedLayout->setContentsMargins( 0, 0, 0, 0 );
    m_advanced->setVisible( false );
    connect( m_advancedButton,
             &QPushButton::toggled,
             this,
             [this]( bool shown )
             {
                 m_advanced->setVisible( shown );
                 m_advancedButton->setText( shown ? QStringLiteral( "Advanced ▾" ) : QStringLiteral( "Advanced ▸" ) );
             } );

    auto *explorationBox    = new QGroupBox( QStringLiteral( "Exploration" ), m_advanced );
    auto *explorationLayout = new QVBoxLayout( explorationBox );
    m_exploration =
        new statusEntry( m_device, "exploration_sequence", statusEntry::STRING, "Sequence", "", explorationBox );
    m_exploration->setObjectName( QStringLiteral( "exploration_sequence" ) );
    m_exploration->setStretch( 0, 1, 4 );
    m_exploration->setToolTip(
        QStringLiteral( "count,sigma,regularization triples, e.g. 1000,0.1,100,1000,0.1,3,1000,0.1,0.1. Used while "
                        "learning and predicting; enter and press Return." ) );
    explorationLayout->addWidget( m_exploration );
    advancedLayout->addWidget( explorationBox );

    auto *detailBox  = new QGroupBox( QStringLiteral( "Diagnostics" ), m_advanced );
    auto *detailGrid = new QGridLayout( detailBox );
    addElementRow( detailGrid, "Upstream enabled", "assisted_qualification", "upstream_enabled" );
    addElementRow( detailGrid, "Upstream locked", "assisted_qualification", "upstream_locked" );
    addElementRow( detailGrid, "Phase error [us]", "assisted_qualification", "upstream_phase_error_us" );
    addElementRow( detailGrid, "Read latency error [us]", "assisted_qualification", "upstream_read_latency_error_us" );
    addElementRow( detailGrid, "Upstream age [s]", "assisted_qualification", "upstream_age_s" );
    addElementRow( detailGrid, "ADC state", "assisted_qualification", "adc_state" );
    addElementRow( detailGrid, "ACC window", "assisted_qualification", "acc_window" );
    addElementRow( detailGrid, "Assisted model", "assisted_qualification", "model" );
    addElementRow( detailGrid, "Diagnostic mode", "assisted_qualification", "diagnostic_mode" );
    addElementRow( detailGrid, "Prediction rate [Hz]", "ddspc_performance", "predict_rate_hz" );
    addElementRow( detailGrid, "QR update rate [Hz]", "ddspc_performance", "update_rate_hz" );
    addElementRow( detailGrid, "Gain refresh rate [Hz]", "ddspc_performance", "refresh_rate_hz" );
    addElementRow( detailGrid, "Held slots", "ddspc_performance", "fast_holds" );
    addElementRow( detailGrid, "ACC lagging", "ddspc_performance", "acc_lagging" );
    addElementRow( detailGrid, "ACC anomalies", "ddspc_performance", "acc_anomalies" );
    addElementRow( detailGrid, "Min margin [us]", "ddspc_performance", "margin_us_min" );
    addElementRow( detailGrid, "ACC age at DM [us]", "ddspc_performance", "acc_age_apply_us_mean" );
    addElementRow( detailGrid, "Standard RMSE", "ddspc_performance", "standard_rmse" );
    addElementRow( detailGrid, "Assisted RMSE", "ddspc_performance", "assisted_rmse" );
    advancedLayout->addWidget( detailBox );
    controls->addWidget( m_advanced );
    controls->addStretch( 1 );

    auto *statusBox  = new QGroupBox( QStringLiteral( "Status" ), this );
    auto *statusGrid = new QGridLayout( statusBox );
    top->addWidget( statusBox, 1, Qt::AlignTop );

    addRow( statusGrid,
            "fsm",
            "Controller",
            [this]()
            {
                return QString::fromStdString( value( "fsm", "state" ) );
            } );
    addRow( statusGrid,
            "mode",
            "Mode",
            [this]()
            {
                return modeName( value( "ddspc_control", "requested_mode" ) );
            } );
    addElementRow( statusGrid, "Learning", "ddspc_control", "learning" );
    addElementRow( statusGrid, "Prediction", "ddspc_control", "prediction" );
    addRow( statusGrid,
            "standard",
            "Standard model",
            [this]()
            {
                const auto ready = value( "ddspc_control", "standard_ready" );
                if( ready.empty() )
                {
                    return c_missing;
                }
                return ready == "true"
                           ? QStringLiteral( "ready" )
                           : QStringLiteral( "not ready (%1)" )
                                 .arg( QString::fromStdString( value( "ddspc_control", "standard_state" ) ) );
            } );
    addRow( statusGrid,
            "assisted",
            "ACC-assisted model",
            [this]()
            {
                const auto ready = value( "ddspc_control", "assisted_ready" );
                if( ready.empty() )
                {
                    return c_missing;
                }
                return ready == "true"
                           ? QStringLiteral( "ready" )
                           : QStringLiteral( "not ready (%1)" )
                                 .arg( QString::fromStdString( value( "ddspc_control", "assisted_state" ) ) );
            } );
    addRow(
        statusGrid,
        "wfsRate",
        "WFS camera rate [Hz]",
        [this]()
        {
            const auto found = m_wfsTiming.m_elements.find( "acq_fps" );
            double     rate;
            return found != m_wfsTiming.m_elements.end() && finiteNumber( found->second.get(), rate ) && rate > 0
                       ? QString::number( rate, 'f', 1 )
                       : c_missing;
        },
        "camwfs.fg_timing.acq_fps: the source of aol1_modevalWFS. loPredCtrlAcc does not publish its own WFS rate." );
    addRow( statusGrid,
            "accRate",
            "ACC rate [Hz]",
            [this]()
            {
                double rate, valid;
                if( !finiteNumber( value( "adc_values", "rate_hz" ), rate ) )
                {
                    return c_missing;
                }
                const bool ok = finiteNumber( value( "adc_values", "valid" ), valid ) && valid == 1;
                return QString::number( rate, 'f', 1 ) + ( ok ? QString() : QStringLiteral( " (invalid)" ) );
            } );
    addElementRow( statusGrid, "ACC sync", "ddspc_control", "accel_sync" );
    addElementRow( statusGrid, "Assisted blockers", "assisted_qualification", "blockers" );
    addRow( statusGrid,
            "output",
            "Aux DM output",
            [this]()
            {
                const auto state = value( "safety", "output_state" );
                return state.empty() ? c_missing
                                     : QString::fromStdString( state ) + QStringLiteral( " (writes " ) +
                                           QString::fromStdString( value( "safety", "dm_write_enabled" ) ) +
                                           QStringLiteral( ")" );
            } );
    addElementRow( statusGrid, "Zero confirmed", "safety", "zero_confirmed" );
    addElementRow( statusGrid, "Aux publish rate [Hz]", "ddspc_performance", "publish_rate_hz" );
    addRow(
        statusGrid,
        "learnTime",
        "Learning time [us]",
        [this]()
        {
            double mean, p99, max;
            return finiteNumber( value( "ddspc_performance", "learn_us_mean" ), mean ) &&
                           finiteNumber( value( "ddspc_performance", "learn_us_p99" ), p99 ) &&
                           finiteNumber( value( "ddspc_performance", "learn_us_max" ), max )
                       ? QStringLiteral( "%1 mean, %2 p99, %3 max" )
                             .arg( mean, 0, 'f', 1 )
                             .arg( p99, 0, 'f', 1 )
                             .arg( max, 0, 'f', 1 )
                       : c_missing;
        },
        "Assisted-mode WFS-thread work per frame, which also delays the next main-loop modal frame." );
    addElementRow( statusGrid, "Learning overruns", "ddspc_performance", "learn_overruns" );
    addRow( statusGrid,
            "modalStream",
            "Main-loop modal output",
            [this]()
            {
                const auto name = value( "fg_shmimName", "name" );
                if( name.empty() )
                {
                    return c_missing;
                }
                const auto width = value( "fg_frameSize", "width" );
                return QString::fromStdString( name ) +
                       ( width.empty() ? QString() : QStringLiteral( " (%1 modes)" ).arg( width.c_str() ) );
            } );
    addRow( statusGrid,
            "transport",
            "Published / acks / skipped",
            [this]()
            {
                const auto published = value( "ddspc_performance", "transport_published" );
                return published.empty() ? c_missing
                                         : QStringLiteral( "%1 / %2 / %3" )
                                               .arg( published.c_str(),
                                                     value( "ddspc_performance", "transport_acks" ).c_str(),
                                                     value( "ddspc_performance", "transport_skipped" ).c_str() );
            } );
    addRow( statusGrid,
            "slots",
            "Slot shifts / step mismatch",
            [this]()
            {
                const auto shifts = value( "ddspc_performance", "slot_shifts" );
                return shifts.empty()
                           ? c_missing
                           : QStringLiteral( "%1 / %2" )
                                 .arg( shifts.c_str(), value( "ddspc_performance", "step_mismatch" ).c_str() );
            } );

    m_warnings = new QLabel( statusBox );
    m_warnings->setObjectName( QStringLiteral( "warnings" ) );
    m_warnings->setWordWrap( true );
    m_warnings->setStyleSheet( QStringLiteral( "color: #e0a030;" ) );
    statusGrid->addWidget( m_warnings, statusGrid->rowCount(), 0, 1, 2 );
    m_age = new QLabel( statusBox );
    m_age->setObjectName( QStringLiteral( "age" ) );
    m_age->setWordWrap( true );
    statusGrid->addWidget( m_age, statusGrid->rowCount(), 0, 1, 2 );

    m_staleTimer = new QTimer( this );
    connect( m_staleTimer,
             &QTimer::timeout,
             this,
             [this]()
             {
                 checkStale();
             } );
    m_staleTimer->start( 1000 );

    onDisconnect();
}

loPredCtrlAccPanel::~loPredCtrlAccPanel() noexcept
{
    detachSubscribersRecursive();
}

toggleSlider *
loPredCtrlAccPanel::addToggle( QBoxLayout *layout, const char *property, const char *label, const char *tooltip )
{
    auto *slider = new toggleSlider( m_device, property, label, layout->parentWidget() );
    slider->setObjectName( QString::fromLatin1( property ) );
    slider->setStretch( 0, 1, 1, true );
    slider->setToolTip( QString::fromLatin1( tooltip ) );
    layout->addWidget( slider );
    return slider;
}

void loPredCtrlAccPanel::addRow(
    QGridLayout *grid, const char *key, const char *label, std::function<QString()> text, const char *tooltip )
{
    const int row  = grid->rowCount();
    auto     *name = new QLabel( QString::fromUtf8( label ), grid->parentWidget() );
    setXwFont( name );
    auto *display = new statusLabel( grid->parentWidget() );
    display->setObjectName( QStringLiteral( "status." ) + QString::fromLatin1( key ) );
    display->setTextFormat( Qt::PlainText );
    display->setWordWrap( true );
    setXwFont( display );

    if( tooltip )
    {
        name->setToolTip( QString::fromUtf8( tooltip ) );
        display->setToolTip( QString::fromUtf8( tooltip ) );
    }

    grid->addWidget( name, row, 0, Qt::AlignTop );
    grid->addWidget( display, row, 1 );
    grid->setColumnStretch( 1, 1 );
    m_rows.push_back( { display, std::move( text ) } );
}

void loPredCtrlAccPanel::addElementRow( QGridLayout *grid,
                                        const char  *label,
                                        const char  *property,
                                        const char  *element )
{
    const std::string key = std::string( property ) + "." + element;
    addRow( grid,
            key.c_str(),
            label,
            [this, property, element]()
            {
                const auto text = value( property, element );
                return text.empty() ? c_missing : QString::fromStdString( text );
            } );
}

void loPredCtrlAccPanel::subscribe()
{
    if( !m_parent )
    {
        return;
    }

    // Standard widgets send through this panel; their readbacks are routed by handleSetProperty().
    for( multiIndiSubscriber *widget :
         std::initializer_list<multiIndiSubscriber *>{ m_learnStd, m_learn, m_predict, m_gainEntry, m_exploration } )
    {
        registerSubscriber( widget );
    }

    for( const auto *property : { "fsm",
                                  "ddspc_mode",
                                  "learn",
                                  "learn_std",
                                  "predict",
                                  "reset_model",
                                  "gain",
                                  "exploration_sequence",
                                  "ddspc_control",
                                  "ddspc_performance",
                                  "safety",
                                  "assisted_qualification",
                                  "adc_values",
                                  "fg_shmimName",
                                  "fg_frameSize" } )
    {
        m_parent->addSubscriberProperty( this, m_device, property );
    }

    m_parent->addSubscriberProperty( this, m_wfsDevice, "fg_timing" );
}

void loPredCtrlAccPanel::onConnect()
{
    setWindowTitle( QString::fromStdString( m_device ) );

    for( auto *slider : { m_learnStd, m_learn, m_predict } )
    {
        slider->onConnect();
    }

    m_gainEntry->onConnect();
    m_exploration->onConnect();
}

void loPredCtrlAccPanel::onDisconnect()
{
    setWindowTitle( QString::fromStdString( m_device ) + QStringLiteral( " (disconnected)" ) );
    m_readbacks.clear();
    m_wfsTiming = {};
    m_previousCounters.clear();
    m_increased.clear();
    m_gainLimits = false;
    m_haveUpdate = false;

    for( auto *slider : { m_learnStd, m_learn, m_predict } )
    {
        slider->onDisconnect();
    }

    m_gainEntry->onDisconnect();
    m_exploration->onDisconnect();
    refresh();
    checkStale();
}

void loPredCtrlAccPanel::handleDefProperty( const pcf::IndiProperty &property )
{
    handleSetProperty( property );
}

void loPredCtrlAccPanel::handleDelProperty( const pcf::IndiProperty &property )
{
    const auto &name = property.getName();

    if( property.getDevice() == m_wfsDevice && ( name.empty() || name == "fg_timing" ) )
    {
        m_wfsTiming = {};
    }

    if( property.getDevice() == m_device )
    {
        if( name.empty() )
        {
            // The whole device left the INDI server: nothing the panel shows is current any more.
            onDisconnect();
            setWindowTitle( QString::fromStdString( m_device ) + QStringLiteral( " (not running)" ) );
            return;
        }

        m_readbacks.erase( name );

        if( name == "learn" || name == "learn_std" || name == "predict" )
        {
            ( name == "learn" ? m_learn : name == "learn_std" ? m_learnStd : m_predict )->onDisconnect();
        }
        else if( name == "gain" )
        {
            m_gainEntry->onDisconnect();
            m_gainLimits = false;
        }
        else if( name == "exploration_sequence" )
        {
            m_exploration->onDisconnect();
        }
    }

    refresh();
}

void loPredCtrlAccPanel::handleSetProperty( const pcf::IndiProperty &property )
{
    const auto &name = property.getName();

    if( property.getDevice() == m_wfsDevice && name == "fg_timing" )
    {
        merge( m_wfsTiming, property );
        refresh();
        return;
    }

    if( property.getDevice() != m_device )
    {
        return;
    }

    merge( m_readbacks[name], property );

    if( name == "gain" )
    {
        readGainLimits( property );
    }
    else if( name == "ddspc_performance" )
    {
        trackCounters( property );
    }

    m_lastUpdate = m_clock();
    m_haveUpdate = true;
    refresh(); // Availability first, so the standard widgets accept their readbacks below.

    if( name == "learn" )
    {
        m_learn->handleSetProperty( property );
    }
    else if( name == "learn_std" )
    {
        m_learnStd->handleSetProperty( property );
    }
    else if( name == "predict" )
    {
        m_predict->handleSetProperty( property );
    }
    else if( name == "gain" )
    {
        m_gainEntry->handleSetProperty( property );
    }
    else if( name == "exploration_sequence" )
    {
        m_exploration->handleSetProperty( property );
    }

    checkStale();
}

void loPredCtrlAccPanel::setResetConfirmation( std::function<bool()> confirm )
{
    m_confirmReset = std::move( confirm );
}

void loPredCtrlAccPanel::setClock( clockFunction clock )
{
    m_clock = std::move( clock );
}

void loPredCtrlAccPanel::checkStale()
{
    if( !m_haveUpdate )
    {
        m_age->setText( QStringLiteral( "No telemetry from %1" ).arg( QString::fromStdString( m_device ) ) );
        m_age->setStyleSheet( QStringLiteral( "color: #e0a030;" ) );
        return;
    }

    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>( m_clock() - m_lastUpdate ).count();

    if( seconds >= c_staleSeconds )
    {
        m_age->setText( QStringLiteral( "No update for %1 s: values may be stale (%2 publishes only on change)" )
                            .arg( seconds )
                            .arg( QString::fromStdString( m_device ) ) );
        m_age->setStyleSheet( QStringLiteral( "color: #e0a030;" ) );
    }
    else
    {
        m_age->setText( QStringLiteral( "Last update %1 s ago" ).arg( seconds ) );
        m_age->setStyleSheet( QString() );
    }
}

std::string loPredCtrlAccPanel::value( const std::string &property, const std::string &element ) const
{
    const auto found = m_readbacks.find( property );

    if( found == m_readbacks.end() )
    {
        return {};
    }

    const auto el = found->second.m_elements.find( element );
    return el == found->second.m_elements.end() ? std::string() : el->second.get();
}

pcf::IndiElement::SwitchStateType loPredCtrlAccPanel::switchState( const std::string &property,
                                                                   const std::string &element ) const
{
    const auto found = m_readbacks.find( property );

    if( found == m_readbacks.end() || found->second.m_type != pcf::IndiProperty::Switch )
    {
        return pcf::IndiElement::UnknownSwitchState;
    }

    const auto el = found->second.m_elements.find( element );
    return el == found->second.m_elements.end() ? pcf::IndiElement::UnknownSwitchState : el->second.getSwitchState();
}

bool loPredCtrlAccPanel::operating() const
{
    return value( "fsm", "state" ) == "OPERATING";
}

QString loPredCtrlAccPanel::interlock() const
{
    if( !operating() )
    {
        return QStringLiteral( "Unavailable: %1 is not operating." ).arg( QString::fromStdString( m_device ) );
    }

    if( !m_readbacks.count( "ddspc_mode" ) )
    {
        return QStringLiteral( "Unavailable: ddspc_mode has not been received." );
    }

    const auto predict = m_readbacks.find( "predict" );

    if( switchState( "predict", "toggle" ) != pcf::IndiElement::Off ||
        ( predict != m_readbacks.end() && predict->second.m_state == pcf::IndiProperty::Busy ) )
    {
        return QStringLiteral( "Turn prediction off to change mode or reset." );
    }

    if( value( "safety", "zero_confirmed" ) != "true" )
    {
        return QStringLiteral( "Waiting for the auxiliary-channel zero to be confirmed." );
    }

    return {};
}

void loPredCtrlAccPanel::merge( Readback &readback, const pcf::IndiProperty &property )
{
    readback.m_type  = property.getType();
    readback.m_state = property.getState();

    for( const auto &[name, element] : property.getElements() )
    {
        readback.m_elements[name] = element;
    }
}

void loPredCtrlAccPanel::readGainLimits( const pcf::IndiProperty &property )
{
    for( const char *name : { "target", "current" } )
    {
        if( !property.find( name ) )
        {
            continue;
        }

        double      min, max, step;
        const auto &element = property[name];

        if( finiteNumber( element.getMin(), min ) && finiteNumber( element.getMax(), max ) &&
            finiteNumber( element.getStep(), step ) && min < max && step > 0 )
        {
            // At most 100000 slider positions; finer targets are still available from the numeric entry.
            m_gainMin    = min;
            m_gainMax    = max;
            m_gainStep   = std::max( step, ( max - min ) / 100000 );
            m_gainLimits = true;
            m_gainSlider->setRange( 0, int( std::lround( ( max - min ) / m_gainStep ) ) );
            m_gainSlider->setToolTip(
                QStringLiteral( "Gain from %1 to %2 (gain property limits)." ).arg( min ).arg( max ) );
            return;
        }
    }
}

void loPredCtrlAccPanel::trackCounters( const pcf::IndiProperty &property )
{
    m_increased.clear();

    for( const char *name :
         { "transport_skipped", "slot_shifts", "step_mismatch", "fast_holds", "acc_anomalies", "learn_overruns" } )
    {
        double current;

        if( !property.find( name ) || !finiteNumber( property[name].get(), current ) )
        {
            continue;
        }

        const auto previous = m_previousCounters.find( name );

        if( previous != m_previousCounters.end() && current > previous->second )
        {
            m_increased << QString::fromLatin1( name );
        }

        m_previousCounters[name] = current;
    }
}

void loPredCtrlAccPanel::sendMode( const char *selected )
{
    if( interlock().isEmpty() )
    {
        // Exactly one element on, as the controller's ddspc_mode callback requires.
        pcf::IndiProperty request( pcf::IndiProperty::Switch, m_device, "ddspc_mode" );
        request.setRule( pcf::IndiProperty::OneOfMany );

        for( const char *element : { "standard", "accel_assisted" } )
        {
            request.add( pcf::IndiElement(
                element, std::strcmp( element, selected ) ? pcf::IndiElement::Off : pcf::IndiElement::On ) );
        }

        sendNewProperty( request );
    }

    refresh(); // The buttons show the readback until the controller reports the new mode.
}

void loPredCtrlAccPanel::requestReset()
{
    if( !m_reset->isEnabled() || !m_confirmReset() || !interlock().isEmpty() || !m_readbacks.count( "reset_model" ) )
    {
        return; // Re-checked after the question: the state may have changed while it was open.
    }

    pcf::IndiProperty request( pcf::IndiProperty::Switch, m_device, "reset_model" );
    request.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );
    sendNewProperty( request );
}

void loPredCtrlAccPanel::sendGainFromSlider()
{
    m_gainEntry->stopEditing();

    if( !m_gainLimits || !m_gainSlider->isEnabled() )
    {
        return;
    }

    const double      target = std::clamp( m_gainMin + m_gainSlider->value() * m_gainStep, m_gainMin, m_gainMax );
    pcf::IndiProperty request( pcf::IndiProperty::Number, m_device, "gain" );
    request.add( pcf::IndiElement( "target", target ) );
    sendNewProperty( request );
}

void loPredCtrlAccPanel::refresh()
{
    const bool ops = operating();

    for( auto *slider : { m_learnStd, m_learn, m_predict } )
    {
        const auto state = switchState( slider->objectName().toStdString(), "toggle" );
        slider->setEnabled( ops && ( state == pcf::IndiElement::On || state == pcf::IndiElement::Off ) );
        slider->guiUpdate();
    }

    const QString blocked = interlock();
    m_interlock->setText( blocked.isEmpty() ? QStringLiteral( "Mode change and reset available." ) : blocked );
    const auto mode = value( "ddspc_control", "requested_mode" );
    m_standard->setChecked( switchState( "ddspc_mode", "standard" ) == pcf::IndiElement::On );
    m_assisted->setChecked( switchState( "ddspc_mode", "accel_assisted" ) == pcf::IndiElement::On );

    for( auto *button : { m_standard, m_assisted } )
    {
        button->setEnabled( blocked.isEmpty() );
        button->setToolTip( blocked );
    }

    m_reset->setEnabled( blocked.isEmpty() && m_readbacks.count( "reset_model" ) );
    m_reset->setToolTip( blocked.isEmpty() ? QStringLiteral( "Reset both models after confirmation." ) : blocked );

    const auto prediction = value( "ddspc_control", "prediction" );
    QString    banner     = QStringLiteral( "Controller unavailable" );
    QString    style      = QStringLiteral( "background: #555555; color: white;" );

    if( ops && !prediction.empty() )
    {
        if( prediction == "active" )
        {
            banner = QStringLiteral( "PREDICTION ACTIVE: DDSPC is correcting tip/tilt" );
            style  = QStringLiteral( "background: #1e7b34; color: white;" );
        }
        else if( prediction == "warming" || prediction == "awaiting_copy" )
        {
            banner = prediction == "warming" ? QStringLiteral( "PREDICTION ON: model warming (probes/feedback)" )
                                             : QStringLiteral( "PREDICTION ON: awaiting first DM copy" );
            style  = QStringLiteral( "background: #8a6d1a; color: white;" );
        }
        else if( prediction == "inhibited" )
        {
            banner = QStringLiteral( "PREDICTION INHIBITED: inputs not valid, tip/tilt uncorrected" );
            style  = QStringLiteral( "background: #9b2c2c; color: white;" );
        }
        else
        {
            banner = QStringLiteral( "PREDICTION OFF: tip/tilt is not corrected" );
            style  = QStringLiteral( "background: #9b2c2c; color: white;" );
        }
    }

    m_banner->setText( banner );
    m_banner->setStyleSheet( style );

    if( mode.empty() )
    {
        m_selectedReady->setText( QStringLiteral( "Selected model: ---" ) );
    }
    else
    {
        const bool ready = value( "ddspc_control", mode == "standard" ? "standard_ready" : "assisted_ready" ) == "true";
        m_selectedReady->setText(
            QStringLiteral( "Selected model (%1): %2" )
                .arg( modeName( mode ), ready ? QStringLiteral( "ready" ) : QStringLiteral( "not ready" ) ) );
    }

    const bool haveGain = ops && m_readbacks.count( "gain" );
    m_gainEntry->setEnabled( haveGain );
    m_gainSlider->setEnabled( haveGain && m_gainLimits );
    double current;

    if( m_gainLimits && !m_gainSlider->isSliderDown() && finiteNumber( value( "gain", "current" ), current ) )
    {
        const QSignalBlocker block( m_gainSlider );
        m_gainSlider->setValue(
            int( std::lround( ( std::clamp( current, m_gainMin, m_gainMax ) - m_gainMin ) / m_gainStep ) ) );
    }

    m_exploration->setEnabled( ops && m_readbacks.count( "exploration_sequence" ) );

    for( auto &row : m_rows )
    {
        const QString text =
            m_haveUpdate || row.m_value->objectName() == QStringLiteral( "status.wfsRate" ) ? row.m_text() : c_missing;

        if( row.m_value->text() != text )
        {
            row.m_value->setTextChanged( text );
        }
    }

    QStringList warnings;

    if( m_haveUpdate && !ops )
    {
        warnings << QStringLiteral( "Controller state is %1, not OPERATING" )
                        .arg( QString::fromStdString( value( "fsm", "state" ) ) );
    }

    const auto safety = m_readbacks.find( "safety" );

    if( safety != m_readbacks.end() && safety->second.m_state == pcf::IndiProperty::Alert )
    {
        warnings
            << QStringLiteral( "Output fault: %1" ).arg( QString::fromStdString( value( "safety", "output_state" ) ) );
    }

    const auto predict = m_readbacks.find( "predict" );

    if( predict != m_readbacks.end() && predict->second.m_state == pcf::IndiProperty::Alert )
    {
        warnings << QStringLiteral( "Prediction request rejected or faulted" );
    }

    if( !m_increased.isEmpty() )
    {
        warnings << QStringLiteral( "Increased: %1" ).arg( m_increased.join( QStringLiteral( ", " ) ) );
    }

    m_warnings->setText( warnings.join( QLatin1Char( '\n' ) ) );
}

} // namespace xqt
