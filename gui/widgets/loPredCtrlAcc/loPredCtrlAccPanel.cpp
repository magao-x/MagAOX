/** \file loPredCtrlAccPanel.cpp
 * \brief Lay out, route and gate the loPredCtrlAccGUI operator widgets.
 */
#include <cmath>
#include <cstring>

#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QStringList>
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

/// Explain a loPredCtrlAcc output fault, or return an empty string when the output state is not a fault.
/** safety.output_state is the fault code, or fault_zero_acknowledged once its zero is copied, followed by
 * "(<fault>: <condition> [<ACC window reason>])" when loPredCtrlAcc knows the condition behind it. The raw code is kept
 * in brackets so the explanation can be matched to the "DDSPC output fault" log entry.
 */
QString describeFault( const std::string &outputState /**< [in] safety.output_state. */ );

QString describeFault( const std::string &outputState )
{
    static const std::map<std::string, const char *> faults{
        { "input_or_model_qualification_lost", "the inputs or the model stopped qualifying" },
        { "command_discarded_zero_pending", "the DM scheduler discarded a DDSPC command" },
        { "scheduler_reset_zero_pending", "the DM scheduler (dmtweeter-synco) restarted" },
        { "ack_timeout_zero_unconfirmed", "the DM scheduler did not acknowledge a command in time" },
        { "nonfinite_or_command_limit", "a DDSPC command was non-finite or beyond its limit" },
        { "candidate_exception_zero_pending", "an internal error occurred while preparing a DM command" },
        { "copy_callback_exception_zero_pending", "an internal error occurred while recording a DM command" },
        { "auxiliary_unavailable", "the DDSPC DM channel was unavailable" },
        { "auxiliary_geometry", "the DDSPC DM channel has the wrong geometry" },
        { "syncopation_transport_unavailable", "the DM scheduler transport was unavailable" } };
    static const std::map<std::string, const char *> conditions{
        { "wfs_invalid", "the WFS measurement was invalid (non-finite)" },
        { "integrator_parameters",
          "the main loop's integrator parameters (holoop gain, mc, limit, aol1_m*fact) were "
          "unavailable" },
        { "command_invalid",
          "the selected model stopped qualifying (a WFS gap longer than max_fill broke its "
          "history, or a numerical fault)" },
        { "wfs_stale", "no WFS frame arrived within input_timeout" },
        { "acc_window", "the ACC window was unusable" },
        { "upstream_unlocked", "the ACC upstream timing lock was lost" } };
    static const std::map<std::string, const char *> windows{
        { "adc_stream_invalid", "ACC stream invalid" },
        { "awaiting_upstream_lock", "waiting for the upstream lock" },
        { "reader_backlog", "ACC reader fell behind" },
        { "history_filling", "ACC history still filling" },
        { "stale_sample", "newest ACC sample too old" },
        { "no_new_sample", "no new ACC samples for more than max_fill frames" },
        { "counter_gap", "more ACC samples missing than max_fill allows" },
        { "nonfinite_sample", "non-finite ACC sample" },
        { "adc_disabled", "ACC disabled" } };

    // "<head> (<detail>)", or the bare fault code.
    std::string head = outputState, detail;
    const auto  open = outputState.find( " (" );

    if( open != std::string::npos && outputState.back() == ')' )
    {
        head   = outputState.substr( 0, open );
        detail = outputState.substr( open + 2, outputState.size() - open - 3 );
    }

    if( head != "fault_zero_acknowledged" && !faults.count( head ) )
    {
        return {};
    }

    if( head == "fault_zero_acknowledged" && detail.empty() )
    {
        return QStringLiteral( "a fault (this controller version does not report its reason) [%1]" )
            .arg( QString::fromStdString( head ) );
    }

    const std::string code      = detail.empty() ? head : detail;
    const auto        colon     = code.find( ": " );
    const std::string fault     = code.substr( 0, colon );
    std::string       condition = colon == std::string::npos ? std::string() : code.substr( colon + 2 ), window;
    const auto        space     = condition.find( ' ' );

    if( space != std::string::npos )
    {
        window    = condition.substr( space + 1 );
        condition = condition.substr( 0, space );
    }

    const auto found = faults.find( fault );
    QString    text  = found == faults.end() ? QString::fromStdString( fault ) : QString::fromLatin1( found->second );

    if( const auto reason = conditions.find( condition ); reason != conditions.end() )
    {
        text = QString::fromLatin1( reason->second );
    }

    if( !window.empty() )
    {
        const auto reason = windows.find( window );
        text += QStringLiteral( ": " ) +
                ( reason == windows.end() ? QString::fromStdString( window ) : QString::fromLatin1( reason->second ) );
    }

    return text + QStringLiteral( " [" ) + QString::fromStdString( code ) + QLatin1Char( ']' );
}
} // namespace

loPredCtrlAccPanel::loPredCtrlAccPanel( const std::string &device, QWidget *parent )
    : xWidget( parent ), m_device( device )
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

    auto *modeBox    = new QGroupBox( QStringLiteral( "Controller mode" ), this );
    auto *modeLayout = new QVBoxLayout( modeBox );
    auto *modeRow    = new QHBoxLayout;
    m_standard       = new QPushButton( QStringLiteral( "Standard" ), modeBox );
    m_assisted       = new QPushButton( QStringLiteral( "ACC Assisted" ), modeBox );
    m_mainLoop       = new QPushButton( QStringLiteral( "Main Loop" ), modeBox );
    m_copy8k         = new QPushButton( QStringLiteral( "mfilt Copy 8 kHz" ), modeBox );
    m_copy2k         = new QPushButton( QStringLiteral( "mfilt Copy 2 kHz" ), modeBox );
    m_clone8k        = new QPushButton( QStringLiteral( "mfilt Clone 8 kHz" ), modeBox );
    m_clone2k        = new QPushButton( QStringLiteral( "mfilt Clone 2 kHz" ), modeBox );
    m_standard->setObjectName( QStringLiteral( "ddspc_mode.standard" ) );
    m_assisted->setObjectName( QStringLiteral( "ddspc_mode.accel_assisted" ) );
    m_mainLoop->setObjectName( QStringLiteral( "ddspc_mode.main_loop" ) );
    m_copy8k->setObjectName( QStringLiteral( "ddspc_mode.mfilt_copy_8k" ) );
    m_copy2k->setObjectName( QStringLiteral( "ddspc_mode.mfilt_copy_2k" ) );
    m_clone8k->setObjectName( QStringLiteral( "ddspc_mode.mfilt_clone_8k" ) );
    m_clone2k->setObjectName( QStringLiteral( "ddspc_mode.mfilt_clone_2k" ) );
    m_copy8k->setWhatsThis(
        QStringLiteral( "No DDSPC: the main loop's tip/tilt integrator, copied, drives the DDSPC DM "
                        "channel on every DM slot while main-loop tip/tilt is zero." ) );
    m_copy2k->setWhatsThis(
        QStringLiteral( "No DDSPC: the main loop's tip/tilt integrator, copied, drives the DDSPC DM "
                        "channel once per WFS frame while main-loop tip/tilt is zero." ) );
    m_clone8k->setWhatsThis(
        QStringLiteral( "No DDSPC: mfilt's tip/tilt state, picked up from aol1_modevalDM, drives the DDSPC DM channel "
                        "on every DM slot while main-loop tip/tilt is zero; the DM total stays mfilt's state." ) );
    m_clone2k->setWhatsThis(
        QStringLiteral( "No DDSPC: mfilt's tip/tilt state, picked up from aol1_modevalDM, drives the DDSPC DM channel "
                        "on step 0 only, with the main loop's write; step-0 sync is reported." ) );
    m_mainLoop->setWhatsThis( QStringLiteral( "Standard DDSPC sent through the main loop's mfilt, as loPredCtrl: no "
                                              "DDSPC DM channel, tip/tilt never zeroed." ) );

    auto *copyRow  = new QHBoxLayout; // The mfilt copies, below the controller modes.
    auto *cloneRow = new QHBoxLayout; // The mfilt clones, below the copies.

    for( auto *button : { m_standard, m_assisted, m_mainLoop, m_copy8k, m_copy2k, m_clone8k, m_clone2k } )
    {
        button->setCheckable( true );
        setXwFont( button );
        ( button == m_copy8k || button == m_copy2k     ? copyRow
          : button == m_clone8k || button == m_clone2k ? cloneRow
                                                       : modeRow )
            ->addWidget( button );
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
    connect( m_mainLoop,
             &QPushButton::clicked,
             this,
             [this]()
             {
                 sendMode( "main_loop" );
             } );
    connect( m_copy8k,
             &QPushButton::clicked,
             this,
             [this]()
             {
                 sendMode( "mfilt_copy_8k" );
             } );
    connect( m_copy2k,
             &QPushButton::clicked,
             this,
             [this]()
             {
                 sendMode( "mfilt_copy_2k" );
             } );
    connect( m_clone8k,
             &QPushButton::clicked,
             this,
             [this]()
             {
                 sendMode( "mfilt_clone_8k" );
             } );
    connect( m_clone2k,
             &QPushButton::clicked,
             this,
             [this]()
             {
                 sendMode( "mfilt_clone_2k" );
             } );
    modeLayout->addLayout( modeRow );
    modeLayout->addLayout( copyRow );
    modeLayout->addLayout( cloneRow );
    m_interlock = new QLabel( modeBox );
    m_interlock->setObjectName( QStringLiteral( "interlock" ) );
    m_interlock->setWordWrap( true );
    setXwFont( m_interlock, 0.9F );
    modeLayout->addWidget( m_interlock );
    controls->addWidget( modeBox );

    auto *ctrlBox    = new QGroupBox( QStringLiteral( "Control" ), this );
    auto *ctrlLayout = new QVBoxLayout( ctrlBox );
    m_learnStd       = addToggle( ctrlLayout,
                            "learn_std",
                            "Standard sequence",
                            "loPredCtrl's three exploration stages, then 1000 frames of learning without probes. "
                                  "Turning it on also enables learning and prediction, so probes and feedback reach the "
                                  "DM. When it ends, learning turns off and prediction continues." );
    m_learn = addToggle( ctrlLayout, "learn", "Learning", "Update the selected model; the other model stays frozen." );
    m_predict   = addToggle( ctrlLayout,
                           "predict",
                           "Prediction",
                           "Apply DDSPC tip/tilt through the auxiliary DM channel. Off removes the contribution and "
                             "waits for its zero to be confirmed." );
    m_gainEntry = new statusEntry( m_device, "gain", statusEntry::FLOAT, "Gain", "", ctrlBox );
    m_gainEntry->setObjectName( QStringLiteral( "gain.entry" ) );
    m_gainEntry->setStretch( 0, 1, 2 );
    m_gainEntry->setToolTip( QStringLiteral( "Added current-WFS feedback gain*y. Enter a target and press Return." ) );
    ctrlLayout->addWidget( m_gainEntry );
    m_learnStdExtraEntry =
        new statusEntry( m_device, "learn_std_extra", statusEntry::FLOAT, "Extra learning [s]", "", ctrlBox );
    m_learnStdExtraEntry->setObjectName( QStringLiteral( "learn_std_extra.entry" ) );
    m_learnStdExtraEntry->setStretch( 0, 1, 2 );
    m_learnStdExtraEntry->setToolTip(
        QStringLiteral( "Seconds the Standard sequence keeps learning after its last stage before learning turns off "
                        "(0: at its end). Enter seconds and press Return; it applies at once." ) );
    ctrlLayout->addWidget( m_learnStdExtraEntry );
    m_reset = new QPushButton( QStringLiteral( "Reset models..." ), ctrlBox );
    m_reset->setObjectName( QStringLiteral( "reset_model" ) );
    setXwFont( m_reset );
    connect( m_reset,
             &QPushButton::clicked,
             this,
             [this]()
             {
                 requestReset();
             } );
    ctrlLayout->addWidget( m_reset );
    controls->addWidget( ctrlBox );

    auto *recoverBox    = new QGroupBox( QStringLiteral( "DM recovery" ), this );
    auto *recoverLayout = new QHBoxLayout( recoverBox );
    m_zeroDm            = new QPushButton( QStringLiteral( "Zero DDSPC DM" ), recoverBox );
    m_releaseTT         = new QPushButton( QStringLiteral( "Return tip/tilt to main loop" ), recoverBox );
    m_zeroDm->setObjectName( QStringLiteral( "zero_dm" ) );
    m_releaseTT->setObjectName( QStringLiteral( "release_tt" ) );
    m_zeroDm->setToolTip(
        QStringLiteral( "Turn prediction off and write a fresh zero to the DDSPC auxiliary DM channel, "
                        "even if the last zero was already confirmed." ) );
    m_releaseTT->setToolTip(
        QStringLiteral( "Turn prediction off and pass main-loop tip/tilt through from the next WFS "
                        "frame, without waiting for the DDSPC zero to be confirmed. The main loop's "
                        "integrator is not changed." ) );

    for( auto *button : { m_zeroDm, m_releaseTT } )
    {
        setXwFont( button );
        recoverLayout->addWidget( button );
        connect( button,
                 &QPushButton::clicked,
                 this,
                 [this, button]()
                 {
                     sendRequest( button->objectName().toLatin1().constData() );
                 } );
    }

    controls->addWidget( recoverBox );
    controls->addStretch( 1 );

    auto *statusBox  = new QGroupBox( QStringLiteral( "Status" ), this );
    auto *statusGrid = new QGridLayout( statusBox );
    top->addWidget( statusBox, 1, Qt::AlignTop );

    addElementRow( statusGrid, "Controller", "fsm", "state" );

    for( const auto &[key, label] : std::initializer_list<std::pair<const char *, const char *>>{
             { "standard", "Standard model" }, { "assisted", "ACC-assisted model" } } )
    {
        const std::string name( key );
        addRow( statusGrid,
                key,
                label,
                [this, name]()
                {
                    const auto ready = value( "ddspc_control", name + "_ready" );

                    if( ready.empty() )
                    {
                        return c_missing;
                    }

                    return ready == "true"
                               ? QStringLiteral( "ready" )
                               : QStringLiteral( "not ready (%1)" )
                                     .arg( QString::fromStdString( value( "ddspc_control", name + "_state" ) ) );
                } );
    }

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
    addElementRow( statusGrid, "Aux DM output", "safety", "output_state" );
    addRow( statusGrid,
            "stopReason",
            "Last stop",
            [this]()
            {
                return stopReason();
            } );
    addRow( statusGrid,
            "mainLoopTT",
            "Main-loop tip/tilt",
            [this]()
            {
                const auto state = value( "ddspc_control", "main_loop_tt" );

                if( state == "zeroed" )
                {
                    return QStringLiteral( "zeroed (DDSPC only)" );
                }

                if( state == "passthrough" )
                {
                    return QStringLiteral( "passed through (main loop)" );
                }

                return state.empty() ? c_missing : QString::fromStdString( state );
            } );
    addElementRow( statusGrid, "Learning overruns", "ddspc_performance", "learn_overruns" );
    addRow( statusGrid,
            "cloneStep0",
            "Clone step 0 (in sync / missed)",
            [this]()
            {
                const auto inSync = value( "mfilt_clone", "step0_in_sync" );
                const auto missed = value( "mfilt_clone", "step0_missed" );

                return inSync.empty() || missed.empty() ? c_missing
                                                        : QString::fromStdString( inSync ) + QStringLiteral( " / " ) +
                                                              QString::fromStdString( missed );
            } );

    m_warnings = new QLabel( statusBox );
    m_warnings->setObjectName( QStringLiteral( "warnings" ) );
    m_warnings->setWordWrap( true );
    m_warnings->setStyleSheet( QStringLiteral( "color: #e0a030;" ) );
    statusGrid->addWidget( m_warnings, statusGrid->rowCount(), 0, 1, 2 );

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

void loPredCtrlAccPanel::addRow( QGridLayout *grid, const char *key, const char *label, std::function<QString()> text )
{
    const int row  = grid->rowCount();
    auto     *name = new QLabel( QString::fromUtf8( label ), grid->parentWidget() );
    setXwFont( name );
    auto *display = new statusLabel( grid->parentWidget() );
    display->setObjectName( QStringLiteral( "status." ) + QString::fromLatin1( key ) );
    display->setTextFormat( Qt::PlainText );
    display->setWordWrap( true );
    setXwFont( display );

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
    for( multiIndiSubscriber *widget : std::initializer_list<multiIndiSubscriber *>{
             m_learnStd, m_learn, m_predict, m_gainEntry, m_learnStdExtraEntry } )
    {
        registerSubscriber( widget );
    }

    for( const auto *property : { "fsm",
                                  "ddspc_mode",
                                  "learn",
                                  "learn_std",
                                  "predict",
                                  "reset_model",
                                  "zero_dm",
                                  "release_tt",
                                  "gain",
                                  "learn_std_extra",
                                  "ddspc_control",
                                  "ddspc_performance",
                                  "safety",
                                  "adc_values",
                                  "integrator",
                                  "mfilt_clone" } )
    {
        m_parent->addSubscriberProperty( this, m_device, property );
    }
}

void loPredCtrlAccPanel::onConnect()
{
    setWindowTitle( QString::fromStdString( m_device ) );

    for( auto *slider : { m_learnStd, m_learn, m_predict } )
    {
        slider->onConnect();
    }

    m_gainEntry->onConnect();
    m_learnStdExtraEntry->onConnect();
}

void loPredCtrlAccPanel::onDisconnect()
{
    setWindowTitle( QString::fromStdString( m_device ) + QStringLiteral( " (disconnected)" ) );
    m_readbacks.clear();

    for( auto *slider : { m_learnStd, m_learn, m_predict } )
    {
        slider->onDisconnect();
    }

    m_gainEntry->onDisconnect();
    m_learnStdExtraEntry->onDisconnect();
    refresh();
}

void loPredCtrlAccPanel::handleDefProperty( const pcf::IndiProperty &property )
{
    handleSetProperty( property );
}

void loPredCtrlAccPanel::handleDelProperty( const pcf::IndiProperty &property )
{
    if( property.getDevice() != m_device )
    {
        return;
    }

    const auto &name = property.getName();

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
    }
    else if( name == "learn_std_extra" )
    {
        m_learnStdExtraEntry->onDisconnect();
    }

    refresh();
}

void loPredCtrlAccPanel::handleSetProperty( const pcf::IndiProperty &property )
{
    if( property.getDevice() != m_device )
    {
        return;
    }

    const auto &name = property.getName();
    merge( m_readbacks[name], property );
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
    else if( name == "learn_std_extra" )
    {
        m_learnStdExtraEntry->handleSetProperty( property );
    }
}

void loPredCtrlAccPanel::setResetConfirmation( std::function<bool()> confirm )
{
    m_confirmReset = std::move( confirm );
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

QString loPredCtrlAccPanel::faultReason() const
{
    return describeFault( value( "safety", "output_state" ) );
}

QString loPredCtrlAccPanel::stopReason() const
{
    const auto state = value( "safety", "output_state" );

    if( state.empty() )
    {
        return c_missing;
    }

    if( const QString fault = faultReason(); !fault.isEmpty() )
    {
        return fault;
    }

    if( switchState( "predict", "toggle" ) == pcf::IndiElement::On )
    {
        return QStringLiteral( "running" );
    }

    return state == "disabled" ? QStringLiteral( "not started since the controller started" )
                               : QStringLiteral( "turned off by request (prediction off, Zero DDSPC DM, Return "
                                                 "tip/tilt, or a mode change)" );
}

QString loPredCtrlAccPanel::refusalReason() const
{
    // The controller's own conditions for accepting a request, in the order it checks them.
    double     enabled, ready;
    const bool mainLoop = switchState( "ddspc_mode", "main_loop" ) == pcf::IndiElement::On; // Needs no integrator.
    const bool copy     = switchState( "ddspc_mode", "mfilt_copy_8k" ) == pcf::IndiElement::On ||
                      switchState( "ddspc_mode", "mfilt_copy_2k" ) == pcf::IndiElement::On ||
                      switchState( "ddspc_mode", "mfilt_clone_8k" ) == pcf::IndiElement::On ||
                      switchState( "ddspc_mode", "mfilt_clone_2k" ) == pcf::IndiElement::On; // Always integrates.

    if( !mainLoop && ( copy || ( finiteNumber( value( "integrator", "enabled" ), enabled ) && enabled == 1 ) ) &&
        finiteNumber( value( "integrator", "ready" ), ready ) && ready == 0 )
    {
        return QStringLiteral( "the main loop's integrator parameters have not been received" );
    }

    const bool        standard = mainLoop || switchState( "ddspc_mode", "standard" ) == pcf::IndiElement::On;
    const std::string mode     = standard ? "standard" : "assisted";

    if( !copy && value( "ddspc_control", mode + "_ready" ) != "true" &&
        switchState( "learn", "toggle" ) != pcf::IndiElement::On )
    {
        return QStringLiteral( "the %1 model is not ready (%2) and learning is off" )
            .arg( standard ? QStringLiteral( "standard" ) : QStringLiteral( "ACC-assisted" ),
                  QString::fromStdString( value( "ddspc_control", mode + "_state" ) ) );
    }

    return QStringLiteral( "the DDSPC output was not available (%1)" )
        .arg( QString::fromStdString( value( "safety", "output_state" ) ) );
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

void loPredCtrlAccPanel::sendMode( const char *selected )
{
    if( interlock().isEmpty() )
    {
        // Exactly one element on, as the controller's ddspc_mode callback requires.
        pcf::IndiProperty request( pcf::IndiProperty::Switch, m_device, "ddspc_mode" );
        request.setRule( pcf::IndiProperty::OneOfMany );

        for( const char *element : { "standard",
                                     "accel_assisted",
                                     "main_loop",
                                     "mfilt_copy_8k",
                                     "mfilt_copy_2k",
                                     "mfilt_clone_8k",
                                     "mfilt_clone_2k" } )
        {
            if( switchState( "ddspc_mode", element ) == pcf::IndiElement::UnknownSwitchState )
            {
                continue; // An older controller without the main_loop, mfilt_copy_* or mfilt_clone_* elements.
            }

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

    sendRequest( "reset_model" );
}

void loPredCtrlAccPanel::sendRequest( const char *property )
{
    if( !operating() || !m_readbacks.count( property ) )
    {
        return;
    }

    pcf::IndiProperty request( pcf::IndiProperty::Switch, m_device, property );
    request.add( pcf::IndiElement( "request", pcf::IndiElement::On ) );
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

    m_gainEntry->setEnabled( ops && m_readbacks.count( "gain" ) );
    m_learnStdExtraEntry->setEnabled( ops && m_readbacks.count( "learn_std_extra" ) );

    const QString blocked = interlock();
    m_interlock->setText( blocked.isEmpty() ? QStringLiteral( "Mode change and reset available." ) : blocked );
    m_standard->setChecked( switchState( "ddspc_mode", "standard" ) == pcf::IndiElement::On );
    m_assisted->setChecked( switchState( "ddspc_mode", "accel_assisted" ) == pcf::IndiElement::On );
    m_mainLoop->setChecked( switchState( "ddspc_mode", "main_loop" ) == pcf::IndiElement::On );
    m_copy8k->setChecked( switchState( "ddspc_mode", "mfilt_copy_8k" ) == pcf::IndiElement::On );
    m_copy2k->setChecked( switchState( "ddspc_mode", "mfilt_copy_2k" ) == pcf::IndiElement::On );
    m_clone8k->setChecked( switchState( "ddspc_mode", "mfilt_clone_8k" ) == pcf::IndiElement::On );
    m_clone2k->setChecked( switchState( "ddspc_mode", "mfilt_clone_2k" ) == pcf::IndiElement::On );

    for( auto *button : { m_standard, m_assisted, m_mainLoop, m_copy8k, m_copy2k, m_clone8k, m_clone2k } )
    {
        button->setEnabled( blocked.isEmpty() );
        button->setToolTip( blocked.isEmpty() ? button->whatsThis() : blocked );
    }

    for( auto *button : { m_mainLoop, m_copy8k, m_copy2k, m_clone8k, m_clone2k } )
    {
        const auto element = button->objectName().toStdString().substr( std::strlen( "ddspc_mode." ) );

        if( switchState( "ddspc_mode", element ) == pcf::IndiElement::UnknownSwitchState )
        {
            button->setEnabled( false );
            button->setToolTip(
                QStringLiteral( "This controller has no %1 mode." ).arg( QString::fromStdString( element ) ) );
        }
    }

    m_reset->setEnabled( blocked.isEmpty() && m_readbacks.count( "reset_model" ) );
    m_reset->setToolTip( blocked.isEmpty() ? QStringLiteral( "Reset both models after confirmation." ) : blocked );

    // Recovery is always offered while operating: it only ever stops prediction.
    m_zeroDm->setEnabled( ops && m_readbacks.count( "zero_dm" ) );
    m_releaseTT->setEnabled( ops && m_readbacks.count( "release_tt" ) );

    const auto prediction = value( "ddspc_control", "prediction" );
    QString    banner     = QStringLiteral( "Controller unavailable" );
    QString    style      = QStringLiteral( "background: #555555; color: white;" );

    if( ops && !prediction.empty() )
    {
        const auto mode = value( "ddspc_control", "requested_mode" );

        if( prediction == "active" && ( mode == "mfilt_clone_8k" || mode == "mfilt_clone_2k" ) )
        {
            banner = QStringLiteral( "MFILT CLONE ACTIVE (%1): mfilt's tip/tilt state on the DDSPC DM channel; step 0 "
                                     "in sync %2, missed %3" )
                         .arg( mode == "mfilt_clone_8k" ? QStringLiteral( "8 kHz" ) : QStringLiteral( "2 kHz" ),
                               QString::fromStdString( value( "mfilt_clone", "step0_in_sync" ) ),
                               QString::fromStdString( value( "mfilt_clone", "step0_missed" ) ) );
            style = QStringLiteral( "background: #1e7b34; color: white;" );
        }
        else if( prediction == "active" && ( mode == "mfilt_copy_8k" || mode == "mfilt_copy_2k" ) )
        {
            banner = QStringLiteral( "MFILT COPY ACTIVE (%1): main-loop tip/tilt integrator on the DDSPC DM channel" )
                         .arg( mode == "mfilt_copy_8k" ? QStringLiteral( "8 kHz" ) : QStringLiteral( "2 kHz" ) );
            style = QStringLiteral( "background: #1e7b34; color: white;" );
        }
        else if( prediction == "active" && value( "ddspc_control", "main_loop_tt" ) == "predicted" )
        {
            banner = QStringLiteral( "PREDICTION ACTIVE: DDSPC tip/tilt through the main loop" );
            style  = QStringLiteral( "background: #1e7b34; color: white;" );
        }
        else if( prediction == "active" )
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
        else if( const QString fault = faultReason(); !fault.isEmpty() && prediction == "off" )
        {
            // Stopped by a fault: say why, whoever now corrects tip/tilt.
            banner = QStringLiteral( "PREDICTION STOPPED: %1" ).arg( fault );
            style  = QStringLiteral( "background: #9b2c2c; color: white;" );
        }
        else if( prediction == "inhibited" )
        {
            banner = QStringLiteral( "PREDICTION INHIBITED: inputs not valid, tip/tilt uncorrected" );
            style  = QStringLiteral( "background: #9b2c2c; color: white;" );
        }
        else if( value( "ddspc_control", "main_loop_tt" ) == "passthrough" )
        {
            banner = QStringLiteral( "PREDICTION OFF: main loop is correcting tip/tilt" );
            style  = QStringLiteral( "background: #2f5f8a; color: white;" );
        }
        else
        {
            // Main-loop tip/tilt stays zeroed until the DDSPC contribution's removal is confirmed.
            banner = QStringLiteral( "PREDICTION OFF: tip/tilt is not corrected" );
            style  = QStringLiteral( "background: #9b2c2c; color: white;" );
        }
    }

    m_banner->setText( banner );
    m_banner->setStyleSheet( style );

    for( auto &row : m_rows )
    {
        const QString text = row.m_text();

        if( row.m_value->text() != text )
        {
            row.m_value->setTextChanged( text );
        }
    }

    QStringList warnings;

    if( !m_readbacks.empty() && !ops )
    {
        warnings << QStringLiteral( "Controller state is %1, not OPERATING" )
                        .arg( QString::fromStdString( value( "fsm", "state" ) ) );
    }

    const auto safety = m_readbacks.find( "safety" );

    if( safety != m_readbacks.end() && safety->second.m_state == pcf::IndiProperty::Alert )
    {
        const QString fault = faultReason();
        warnings << QStringLiteral( "Output fault: %1" )
                        .arg( fault.isEmpty() ? QString::fromStdString( value( "safety", "output_state" ) ) : fault );
    }

    const auto predict = m_readbacks.find( "predict" );

    if( predict != m_readbacks.end() && predict->second.m_state == pcf::IndiProperty::Alert )
    {
        // A fault is explained above; otherwise the request itself was refused.
        warnings << ( faultReason().isEmpty()
                          ? QStringLiteral( "Prediction request refused: %1" ).arg( refusalReason() )
                          : QStringLiteral( "Prediction stopped by the output fault" ) );
    }

    m_warnings->setText( warnings.join( QLatin1Char( '\n' ) ) );
}

} // namespace xqt
