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

    auto *ctrlBox    = new QGroupBox( QStringLiteral( "Control" ), this );
    auto *ctrlLayout = new QVBoxLayout( ctrlBox );
    m_learnStd       = addToggle( ctrlLayout,
                            "learn_std",
                            "Standard sequence",
                            "Original three-stage exploration. Turning it on also enables learning and prediction, "
                                  "so probes and feedback reach the DM." );
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
    addElementRow( statusGrid, "Learning overruns", "ddspc_performance", "learn_overruns" );

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
    for( multiIndiSubscriber *widget :
         std::initializer_list<multiIndiSubscriber *>{ m_learnStd, m_learn, m_predict, m_gainEntry } )
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
                                  "ddspc_control",
                                  "ddspc_performance",
                                  "safety",
                                  "adc_values" } )
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

    const QString blocked = interlock();
    m_interlock->setText( blocked.isEmpty() ? QStringLiteral( "Mode change and reset available." ) : blocked );
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
        warnings
            << QStringLiteral( "Output fault: %1" ).arg( QString::fromStdString( value( "safety", "output_state" ) ) );
    }

    const auto predict = m_readbacks.find( "predict" );

    if( predict != m_readbacks.end() && predict->second.m_state == pcf::IndiProperty::Alert )
    {
        warnings << QStringLiteral( "Prediction request rejected or faulted" );
    }

    m_warnings->setText( warnings.join( QLatin1Char( '\n' ) ) );
}

} // namespace xqt
