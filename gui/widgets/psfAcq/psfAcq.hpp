/** \file psfAcq.hpp
 * \brief PSF acquisition panel embedded in cameraGUI: select a detected star, view its seeing, and acquire it.
 */
#ifndef psfAcq_hpp
#define psfAcq_hpp

#include <array>
#include <charconv>

#include <QSignalBlocker>
#include <QThread>
#include <QTimer>

#include "ui_psfAcq.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt
{

/// Select one of the stars detected by a psfAcq device, display its seeing, and request acquisition.
class psfAcq : public xWidget
{
    Q_OBJECT

  public:
    /// Number of `star_N` slots subscribed; must cover the device's `fitter.max_loops`.
    static constexpr int s_maxStars = 20;

  protected:
    /// Last-received values from one `star_N` property.
    struct starState
    {
        /// The property is currently defined by the device.
        bool m_valid{ false };

        /// The property carries a `seeing` element.
        bool m_hasSeeing{ false };

        /// Seeing of this star in arcseconds.
        float m_seeing{ 0 };

        /// Track id from `star_N.id`, stable across brightness-rank changes; -1 for an empty slot.
        int m_id{ -1 };
    };

    /// psfAcq device whose properties this widget subscribes to.
    std::string m_devName;

    /// Last reported device FSM state; controls command availability.
    std::string m_appState;

    /// Retained state of each `star_N` property, indexed by N.
    std::array<starState, s_maxStars> m_stars;

    /// Track id the device uses for its published seeing, from `seeing_star.current` or the last user selection; -1
    /// if unknown, which selects the brightest star.
    int m_seeingStar{ -1 };

    /// Periodic GUI refresh timer, owned by this widget.
    QTimer *m_updateTimer{ nullptr };

  public:
    /// Construct the PSF acquisition widget.
    explicit psfAcq( std::string    &devName /**< [in] psfAcq INDI device name. */,
                     QWidget        *Parent = 0 /**< [in] Optional owning Qt widget. */,
                     Qt::WindowFlags f      = Qt::WindowFlags() /**< [in] Qt window flags. */ );

    /// Destroy the widget and its Qt-owned timer.
    ~psfAcq();

    /// Subscribe to the device state, the seeing star, and every possible `star_N` property.
    void subscribe();

    /// Clear cached state and disable commands until fresh properties arrive.
    virtual void onDisconnect();

    /// Record a newly defined property; identical to a set.
    void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has been defined*/ );

    /// Invalidate a deleted star, or all state when the device goes away.
    void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has been deleted*/ );

    /// Cache the FSM state and star values on the GUI thread.
    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    /// Parse a `star_N` property name.
    /** \returns N if the name is `star_N` with 0 <= N < s_maxStars, -1 otherwise.
     */
    static int starIndex( const std::string &propName /**< [in] INDI property name. */ );

    /// Find the brightness rank of a track.
    /** \returns N of the `star_N` property carrying track id \p id, or -1 if that track is not listed.
     */
    int starRank( int id /**< [in] Track id from `star_N.id`. */ ) const;

  public slots:
    /// Refresh the star list, the selected star's seeing, and control availability.
    /** Stars are listed by rank and selected by track id, so a brightness-rank swap keeps the selected star.
     * The selection follows the device's seeing star, or the brightest star until the device reports one.
     */
    void updateGUI();

    /// Request acquisition of the selected star by its track id.
    void on_button_acquire_pressed();

    /// Ask the device to calculate seeing from the star the user selected, by its track id.
    void on_combo_star_activated( int index /**< [in] Selected combo box row. */ );

  private:
    /// Qt-generated controls owned by this widget.
    Ui::psfAcq ui;
};

psfAcq::psfAcq( std::string &devName, QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f ), m_devName{ devName }
{
    ui.setupUi( this );

    setXwFont( ui.label_title );
    setXwFont( ui.label_star );
    setXwFont( ui.combo_star );
    setXwFont( ui.label_seeing );
    setXwFont( ui.val_seeing );
    setXwFont( ui.button_acquire );

    ui.val_seeing->setProperty( "isStatus", true );

    m_updateTimer = new QTimer( this );

    connect( m_updateTimer, SIGNAL( timeout() ), this, SLOT( updateGUI() ) );
    m_updateTimer->start( 250 );

    onDisconnect();
}

psfAcq::~psfAcq()
{
}

void psfAcq::subscribe()
{
    if( !m_parent )
    {
        return;
    }

    m_parent->addSubscriberProperty( this, m_devName, "fsm" );

    m_parent->addSubscriberProperty( this, m_devName, "seeing_star" );

    for( int n = 0; n < s_maxStars; ++n )
    {
        m_parent->addSubscriberProperty( this, m_devName, "star_" + std::to_string( n ) );
    }
}

void psfAcq::onDisconnect()
{
    if( QThread::currentThread() != thread() )
    {
        QMetaObject::invokeMethod(
            this,
            [this]
            {
                onDisconnect();
            },
            Qt::QueuedConnection );
        return;
    }

    m_appState.clear();
    m_stars.fill( starState{} );
    m_seeingStar = -1;

    updateGUI();
}

void psfAcq::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    handleSetProperty( ipRecv );
}

void psfAcq::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_devName )
    {
        return;
    }

    if( ipRecv.getName() == "" || ipRecv.getName() == "fsm" )
    {
        onDisconnect();
        return;
    }

    int n = starIndex( ipRecv.getName() );

    if( n >= 0 )
    {
        m_stars[n] = starState{};
    }
}

void psfAcq::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_devName )
    {
        return;
    }

    if( ipRecv.getName() == "fsm" )
    {
        if( ipRecv.find( "state" ) )
        {
            m_appState = ipRecv["state"].get<std::string>();
        }

        return;
    }

    if( ipRecv.getName() == "seeing_star" )
    {
        if( ipRecv.find( "current" ) )
        {
            m_seeingStar = ipRecv["current"].get<int>();
        }

        return;
    }

    int n = starIndex( ipRecv.getName() );

    if( n < 0 )
    {
        return;
    }

    m_stars[n].m_valid = true;

    if( ipRecv.find( "seeing" ) )
    {
        m_stars[n].m_hasSeeing = true;
        m_stars[n].m_seeing    = ipRecv["seeing"].get<float>();
    }

    if( ipRecv.find( "id" ) )
    {
        m_stars[n].m_id = ipRecv["id"].get<int>();
    }
}

int psfAcq::starIndex( const std::string &propName )
{
    static const std::string prefix = "star_";

    if( propName.size() <= prefix.size() || propName.compare( 0, prefix.size(), prefix ) != 0 )
    {
        return -1;
    }

    int         n    = -1;
    const char *last = propName.data() + propName.size();
    auto [ptr, ec]   = std::from_chars( propName.data() + prefix.size(), last, n );

    if( ec != std::errc() || ptr != last || n < 0 || n >= s_maxStars )
    {
        return -1;
    }

    return n;
}

int psfAcq::starRank( int id ) const
{
    for( int n = 0; n < s_maxStars; ++n )
    {
        if( id >= 0 && m_stars[n].m_valid && m_stars[n].m_id == id )
        {
            return n;
        }
    }

    return -1;
}

void psfAcq::updateGUI()
{
    bool ready = ( m_appState == "READY" || m_appState == "OPERATING" );

    // Rebuild the list only when the ranked tracks change, keeping the selected track if it survives.
    bool listChanged = false;
    int  item        = 0;

    for( int n = 0; n < s_maxStars; ++n )
    {
        if( !m_stars[n].m_valid || m_stars[n].m_id < 0 )
        {
            continue;
        }

        if( item >= ui.combo_star->count() || ui.combo_star->itemData( item ).toInt() != m_stars[n].m_id )
        {
            listChanged = true;
            break;
        }

        ++item;
    }

    if( !listChanged && item != ui.combo_star->count() )
    {
        listChanged = true;
    }

    if( listChanged )
    {
        QSignalBlocker blocker( ui.combo_star );

        QVariant selected = ui.combo_star->currentData();

        ui.combo_star->clear();

        for( int n = 0; n < s_maxStars; ++n )
        {
            if( m_stars[n].m_valid && m_stars[n].m_id >= 0 )
            {
                ui.combo_star->addItem( QString( "Star %1 (#%2)" ).arg( n ).arg( m_stars[n].m_id ), m_stars[n].m_id );
            }
        }

        int idx = ui.combo_star->findData( selected );

        if( idx >= 0 )
        {
            ui.combo_star->setCurrentIndex( idx );
        }
    }

    // Seeing is calculated from the brightest star unless another track has been selected.
    int seeingIdx = ui.combo_star->findData( m_seeingStar );

    if( m_seeingStar < 0 && ui.combo_star->count() > 0 )
    {
        seeingIdx = 0;
    }

    if( seeingIdx >= 0 && seeingIdx != ui.combo_star->currentIndex() )
    {
        QSignalBlocker blocker( ui.combo_star );

        ui.combo_star->setCurrentIndex( seeingIdx );
    }

    bool haveStar = ( ui.combo_star->count() > 0 );

    ui.combo_star->setEnabled( ready && haveStar );
    ui.button_acquire->setEnabled( ready && haveStar );

    int n = haveStar ? starRank( ui.combo_star->currentData().toInt() ) : -1;

    // Seeing is shown as soon as the selected star is detected.
    if( n >= 0 && m_stars[n].m_hasSeeing )
    {
        ui.val_seeing->setEnabled( true );
        ui.val_seeing->setText( QString( "%1\"" ).arg( m_stars[n].m_seeing, 0, 'f', 2 ) );
    }
    else
    {
        ui.val_seeing->setEnabled( false );
        ui.val_seeing->setText( "---" );
    }
}

void psfAcq::on_button_acquire_pressed()
{
    // Do not acquire a track that is no longer listed; the device also rejects tracks it has lost.
    if( ui.combo_star->count() == 0 || starRank( ui.combo_star->currentData().toInt() ) < 0 )
    {
        return;
    }

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( m_devName );
    ip.setName( "acquire_star" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = ui.combo_star->currentData().toInt();

    sendNewProperty( ip );
}

void psfAcq::on_combo_star_activated( int index )
{
    if( index < 0 || index >= ui.combo_star->count() )
    {
        return;
    }

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( m_devName );
    ip.setName( "seeing_star" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = ui.combo_star->itemData( index ).toInt();

    // Hold the selection until the device reports it in `seeing_star.current`.
    m_seeingStar = ip["target"].get<int>();

    sendNewProperty( ip );
}

} // namespace xqt

#include "moc_psfAcq.cpp"

#endif
