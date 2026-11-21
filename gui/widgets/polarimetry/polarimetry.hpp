/** \file polarimetry.hpp
 * \brief Polarimetry control widget: HWP angle, QWP polarization and pupil tracking, linear stages, and the hwpSequencer.
 */

#ifndef polarimetry_hpp
#define polarimetry_hpp

#include <cmath>

#include <QStringList>
#include <QStyle>

#include "ui_polarimetry.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt
{

/// Polarimetry panel for `hwptrack`, `qwptrack`, `qwppupil`, the polarimetry stages, and `hwpsequence`.
/**
 * Shows the HWP angle and name, the FSM states of `hwptrack`, `qwppupil`, `qwptrack` and the stages, and
 * the hwpSequencer status, and sends start, stop, last-cycle and abort requests to `hwpsequence`. While `qwptrack` is tracking, the QWP fast-axis angles are shown
 * beside the QWP pol. tracking slider. While `qwppupil` is tracking, the ttmperi voltage offsets it applies
 * are shown beside the QWP pupil tracking slider, which may be toggled during a sequence. Sequence
 * parameters other than `num_cycles` are locked while sequencing, and `time_per_pos` is disabled in
 * frames mode (`frames_per_pos` > 0).
 */
class polarimetry : public xWidget
{
    Q_OBJECT

  protected:
    /** \name hwptrack State
     * @{
     */

    /// Last `hwptrack.hwp_position.current` [deg].
    float m_hwpSetAngle{ 0 };

    /// Last `hwptrack.hwp_position_name.value`.
    std::string m_hwpAngleName{ "" };

    /// Whether `hwptrack` is READY or OPERATING.
    bool m_hwptrackFsmOk{ false };

    ///@}

    /** \name hwpsequence State
     * @{
     */

    /// Whether `hwpsequence.sequence` is On.
    bool m_sequencing{ false };

    /// Whether `hwpsequence.abort` is On, i.e. an abort is in progress.
    bool m_aborting{ false };

    /// Last `hwpsequence.status.pos_index` (0-based).
    int m_hwpPosIndex{ 0 };

    /// Last `hwpsequence.status.cycle`.
    int m_curCycle{ 0 };

    /// Last `hwpsequence.num_cycles.current` (0 = until stopped).
    int m_numCycles{ 0 };

    /// Last `hwpsequence.frames_per_pos.current` (0 = time mode).
    int m_framesPerPos{ 0 };

    /// Last `hwpsequence.status.frames`: frames that landed at the previous position.
    int m_framesLast{ 0 };

    /// Number of positions in `hwpsequence.positions`, 0 if unknown.
    int m_numPositions{ 0 };

    /// Whether `hwpsequence` is READY or OPERATING.
    bool m_hwpseqFsmOk{ false };

    ///@}

    /** \name qwptrack State
     * @{
     */

    /// Whether `qwptrack.tracking` is On.
    bool m_qwpTracking{ false };

    /// Whether `qwptrack` is READY or OPERATING.
    bool m_qwptrackFsmOk{ false };

    /// Last `qwptrack.qwp1.current`, the QWP1 fast-axis angle [deg].
    float m_qwp1Angle{ 0 };

    /// Last `qwptrack.qwp2.current`, the QWP2 fast-axis angle [deg].
    float m_qwp2Angle{ 0 };

    ///@}

    /** \name qwppupil State
     * @{
     */

    /// Whether `qwppupil.tracking` is On.
    bool m_qwppupilTracking{ false };

    /// Whether `qwppupil` is READY or OPERATING.
    bool m_qwppupilFsmOk{ false };

    /// Last `qwppupil.offset.v1`, the ttmperi axis 1 voltage applied relative to the baseline [V].
    float m_qwppupilV1{ 0 };

    /// Last `qwppupil.offset.v2`, the ttmperi axis 2 voltage applied relative to the baseline [V].
    float m_qwppupilV2{ 0 };

    ///@}

    /// Set or clear the bold font on a label.
    void setBold( QLabel *label, /**< [in] the label to change */
                  bool    onoff  /**< [in] true for bold */
    );

    /// Lock or unlock a toggle slider against user input.
    /** The slider is not disabled, since a disabled toggleSlider appears "off" even when the toggle is On.
     */
    void lockSlider( toggleSlider *slider, /**< [in] the slider to change */
                     bool          locked  /**< [in] true to ignore mouse and keyboard input */
    );

    /// Set or clear the `isWarning` style property on a label and re-polish it.
    void setWarning( QLabel *label, /**< [in] the label to change */
                     bool    onoff  /**< [in] true to show the warning style */
    );

  public:
    /// Count the entries in a comma-separated positions list.
    /**
     * \returns the number of positions, or 0 if the list is empty or any entry is not a number.
     */
    static int countPositions( const std::string &positions /**< [in] the `hwpsequence.positions` text */ );

    /// Format the ttmperi voltage offsets applied by `qwppupil` for the label beside its slider.
    /**
     * Each value has one decimal and an explicit sign (`+` or U+2212), and is left-padded with figure
     * spaces to the width of a sign and three integer digits, so the text width stays constant.
     *
     * \returns the text, e.g. `ΔV1:   +1.2 V  ΔV2:  −12.4 V`
     */
    static QString pupilOffsetText( float v1, /**< [in] the ttmperi axis 1 offset [V] */
                                    float v2  /**< [in] the ttmperi axis 2 offset [V] */
    );

    /// Construct the panel and configure its child widgets.
    explicit polarimetry( QWidget        *Parent = 0,                /**< [in] the parent widget */
                          Qt::WindowFlags f      = Qt::WindowFlags() /**< [in] Qt window flags */
    );

    /// Unsubscribe from the parent publisher.
    ~polarimetry();

    /// Subscribe the panel and its child widgets to their INDI properties.
    void subscribe();

    /// Enable the panel and its children when the INDI connection is made.
    virtual void onConnect();

    /// Reset displays and disable controls when the INDI connection is lost.
    virtual void onDisconnect();

    /// Handle a property definition by treating it as a set.
    void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    /// Update the cached device state from a property and request a GUI update.
    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

  public slots:

    /// Update displays and control availability from the cached device state.
    void updateGUI();

    /// Request a sequence start (`hwpsequence.sequence` On).
    void on_buttonStartSequence_pressed();

    /// Request or cancel ending after the current cycle (`hwpsequence.last_cycle`).
    void on_buttonLastCycle_clicked( bool checked /**< [in] true to end after the current cycle */ );

    /// Request a graceful stop (`hwpsequence.sequence` Off).
    void on_buttonStopSequence_pressed();

    /// Request an immediate abort (`hwpsequence.abort` On); the in-flight frame is discarded.
    void on_buttonAbortSequence_pressed();

  signals:

    /// Request a GUI update on the GUI thread.
    void doUpdateGUI();

  private:
    /// Generated UI.
    Ui::polarimetry ui;
};

polarimetry::polarimetry( QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f )
{
    ui.setupUi( this );

    setWindowTitle( QString( "Polarimetry (disconnected)" ) );

    setXwFont( ui.labelHwptrack );
    setXwFont( ui.labelHwpseq );
    setXwFont( ui.labelQwpPupil );
    setXwFont( ui.labelStagePolLin );
    setXwFont( ui.labelQwpTrack );
    setXwFont( ui.labelStageQwpLin );

    setXwFont( ui.labelAngle );
    setXwFont( ui.labelName );
    setXwFont( ui.labelHwp );

    setXwFont( ui.hwpSetAngle );
    setXwFont( ui.labelQwpTracking );
    setXwFont( ui.labelQwpPupilTracking );
    setXwFont( ui.hwpAngleName );

    setXwFont( ui.labelCycleStatus );
    setXwFont( ui.hwpPosIndex );
    setXwFont( ui.labelCycleNum );
    setXwFont( ui.cycleNumStatus );
    setXwFont( ui.labelFramesStatus );
    setXwFont( ui.framesStatus );

    setXwFont( ui.buttonStartSequence );
    setXwFont( ui.buttonLastCycle );
    setXwFont( ui.buttonStopSequence );
    setXwFont( ui.buttonAbortSequence );
    setXwFont( ui.labelNumCyclesHint );
    setXwFont( ui.labelFramesHint );
    setXwFont( ui.labelTimeHint );

    ui.hwptrackFsm->highlightChanges( false );
    ui.hwpseqFsm->highlightChanges( false );
    ui.qwpPupilFsm->highlightChanges( false );
    ui.stagePolLinFsm->highlightChanges( false );
    ui.qwpTrackFsm->highlightChanges( false );
    ui.stageQwpLinFsm->highlightChanges( false );


    ui.labelHwptrack->setText( QString( "hwptrack" ) );
    ui.hwptrackFsm->device( "hwptrack" );

    ui.labelHwpseq->setText( QString( "hwpsequence" ) );
    ui.hwpseqFsm->device( "hwpsequence" );

    ui.labelQwpPupil->setText( QString( "qwppupil" ) );
    ui.qwpPupilFsm->device( "qwppupil" );

    ui.labelStagePolLin->setText( QString( "stagepollin" ) );
    ui.stagePolLinFsm->device( "stagepollin" );

    ui.labelQwpTracking->setVisible( false );
    ui.labelQwpTracking->setText( QString( "---" ) );

    ui.labelQwpPupilTracking->setVisible( false );
    ui.labelQwpPupilTracking->setText( QString( "---" ) );

    ui.labelQwpTrack->setText( QString( "qwptrack" ) );
    ui.qwpTrackFsm->device( "qwptrack" );

    ui.labelStageQwpLin->setText( QString( "stageqwplin" ) );
    ui.stageQwpLinFsm->device( "stageqwplin" );

    ui.buttonLastCycle->setCheckable( true );
    ui.buttonLastCycle->setProperty( "isHighlightButton", true );
    ui.buttonAbortSequence->setProperty( "isAbortButton", true );

    ui.labelFramesStatus->setVisible( false );
    ui.framesStatus->setVisible( false );

    ui.entryHwpAngle->setup( "hwptrack", "hwp_position", statusEntry::FLOAT, "HWP target", "°" );
    ui.entryHwpAngle->format( "%.01f" );
    ui.entryHwpAngle->setStretch( 0, 2, 1 );


    ui.comboHwpLin->setup( "stagepollin", "presetName", "", "HWP lin. stage", "" );
    ui.comboHwpLin->ctrlWidget( nullptr );

    ui.sliderQwpPupilTracking->setup( "qwppupil", "tracking", "toggle", "QWP pupil tracking" );
    ui.sliderQwpPupilTracking->setStretch( 0, 1, 3, true, false );

    ui.sliderQwpTracking->setup( "qwptrack", "tracking", "toggle", "QWP pol. tracking" );
    ui.sliderQwpTracking->setStretch( 0, 1, 3, true, false );

    ui.comboQwpLin->setup( "stageqwplin", "presetName", "", "QWP lin. stage", "" );
    ui.comboQwpLin->ctrlWidget( nullptr );

    ui.entryNumCycles->setup( "hwpsequence", "num_cycles", statusEntry::INT, "Num. cycles", "" );
    ui.entryNumCycles->setStretch( 0, 2, 1 );

    ui.entryFramesPerPos->setup( "hwpsequence", "frames_per_pos", statusEntry::INT, "Frames per pos.", "" );
    ui.entryFramesPerPos->setStretch( 0, 2, 1 );

    ui.entryTimePerPos->setup( "hwpsequence", "time_per_pos", statusEntry::FLOAT, "Time per pos.", "s" );
    ui.entryTimePerPos->format( "%.01f" );
    ui.entryTimePerPos->setStretch( 0, 2, 1 );

    connect( this, SIGNAL( doUpdateGUI() ), this, SLOT( updateGUI() ) );

    onDisconnect();
}

polarimetry::~polarimetry()
{
    if( m_parent )
    {
        m_parent->unsubscribe( this );
    }
}

void polarimetry::subscribe()
{
    if( !m_parent )
    {
        return;
    }

    m_parent->addSubscriberProperty( this, "hwptrack", "fsm" );
    m_parent->addSubscriberProperty( this, "hwptrack", "hwp_position" );
    m_parent->addSubscriberProperty( this, "hwptrack", "hwp_position_name" );

    m_parent->addSubscriberProperty( this, "hwpsequence", "fsm" );
    m_parent->addSubscriberProperty( this, "hwpsequence", "sequence" );
    m_parent->addSubscriberProperty( this, "hwpsequence", "last_cycle" );
    m_parent->addSubscriberProperty( this, "hwpsequence", "abort" );
    m_parent->addSubscriberProperty( this, "hwpsequence", "num_cycles" );
    m_parent->addSubscriberProperty( this, "hwpsequence", "frames_per_pos" );
    m_parent->addSubscriberProperty( this, "hwpsequence", "status" );
    m_parent->addSubscriberProperty( this, "hwpsequence", "positions" );

    m_parent->addSubscriberProperty( this, "qwptrack", "fsm" );
    m_parent->addSubscriberProperty( this, "qwptrack", "tracking" );
    m_parent->addSubscriberProperty( this, "qwptrack", "qwp1" );
    m_parent->addSubscriberProperty( this, "qwptrack", "qwp2" );

    m_parent->addSubscriberProperty( this, "qwppupil", "fsm" );
    m_parent->addSubscriberProperty( this, "qwppupil", "tracking" );
    m_parent->addSubscriberProperty( this, "qwppupil", "offset" );

    m_parent->addSubscriber( ui.hwptrackFsm );
    m_parent->addSubscriber( ui.hwpseqFsm );
    m_parent->addSubscriber( ui.qwpPupilFsm );
    m_parent->addSubscriber( ui.stagePolLinFsm );
    m_parent->addSubscriber( ui.qwpTrackFsm );
    m_parent->addSubscriber( ui.stageQwpLinFsm );
    m_parent->addSubscriber( ui.entryHwpAngle );
    m_parent->addSubscriber( ui.comboHwpLin );
    m_parent->addSubscriber( ui.sliderQwpPupilTracking );
    m_parent->addSubscriber( ui.sliderQwpTracking );
    m_parent->addSubscriber( ui.comboQwpLin );
    m_parent->addSubscriber( ui.entryNumCycles );
    m_parent->addSubscriber( ui.entryFramesPerPos );
    m_parent->addSubscriber( ui.entryTimePerPos );

    return;
}

void polarimetry::onConnect()
{

    setWindowTitle( QString( "Polarimetry" ) );
    ui.hwptrackFsm->onConnect();
    ui.hwpseqFsm->onConnect();
    ui.qwpPupilFsm->onConnect();
    ui.stagePolLinFsm->onConnect();
    ui.qwpTrackFsm->onConnect();
    ui.stageQwpLinFsm->onConnect();

    setBold( ui.hwpSetAngle, true );
    setBold( ui.hwpAngleName, true );

    ui.entryHwpAngle->onConnect();
    ui.comboHwpLin->onConnect();
    ui.sliderQwpPupilTracking->onConnect();
    ui.sliderQwpTracking->onConnect();
    ui.comboQwpLin->onConnect();
    ui.entryNumCycles->onConnect();
    ui.entryFramesPerPos->onConnect();
    ui.entryTimePerPos->onConnect();

    ui.entryHwpAngle->setEnabled( true );
    ui.comboHwpLin->setEnabled( true );
    ui.sliderQwpPupilTracking->setEnabled( true );
    ui.sliderQwpTracking->setEnabled( true );
    ui.comboQwpLin->setEnabled( true );
    ui.entryNumCycles->setEnabled( true );
    ui.labelNumCyclesHint->setEnabled( true );
    ui.entryFramesPerPos->setEnabled( true );
    ui.labelFramesHint->setEnabled( true );
    ui.entryTimePerPos->setEnabled( true );
    ui.labelTimeHint->setEnabled( true );

    ui.buttonStartSequence->setEnabled( true );
    ui.buttonLastCycle->setEnabled( true );
    ui.buttonStopSequence->setEnabled( true );
    ui.buttonAbortSequence->setEnabled( true );
}

void polarimetry::onDisconnect()
{
    setWindowTitle( QString( "Polarimetry (disconnected)" ) );

    ui.hwptrackFsm->onDisconnect();
    ui.hwpseqFsm->onDisconnect();
    ui.qwpPupilFsm->onDisconnect();
    ui.stagePolLinFsm->onDisconnect();
    ui.qwpTrackFsm->onDisconnect();
    ui.stageQwpLinFsm->onDisconnect();

    ui.hwpSetAngle->setText( QString( "---" ) );
    ui.hwpAngleName->setText( QString( "---" ) );
    setBold( ui.hwpSetAngle, false );
    setBold( ui.hwpAngleName, false );

    ui.labelQwpTracking->setText( QString( "---" ) );
    ui.labelQwpPupilTracking->setText( QString( "---" ) );

    ui.hwpPosIndex->setText( QString( "---" ) );
    ui.cycleNumStatus->setText( QString( "---" ) );
    setBold( ui.hwpPosIndex, false );
    setBold( ui.cycleNumStatus, false );

    ui.framesStatus->setText( QString( "---" ) );
    setWarning( ui.framesStatus, false );

    ui.entryHwpAngle->onDisconnect();
    ui.comboHwpLin->onDisconnect();
    ui.sliderQwpPupilTracking->onDisconnect();
    ui.sliderQwpTracking->onDisconnect();
    ui.comboQwpLin->onDisconnect();
    ui.entryNumCycles->onDisconnect();
    ui.entryFramesPerPos->onDisconnect();
    ui.entryTimePerPos->onDisconnect();

    ui.entryHwpAngle->setEnabled( false );
    ui.comboHwpLin->setEnabled( false );
    ui.sliderQwpPupilTracking->setEnabled( false );
    ui.sliderQwpTracking->setEnabled( false );
    ui.comboQwpLin->setEnabled( false );
    ui.entryNumCycles->setEnabled( false );
    ui.labelNumCyclesHint->setEnabled( false );
    ui.entryFramesPerPos->setEnabled( false );
    ui.labelFramesHint->setEnabled( false );
    ui.entryTimePerPos->setEnabled( false );
    ui.labelTimeHint->setEnabled( false );

    ui.buttonStartSequence->setEnabled( false );
    ui.buttonLastCycle->setEnabled( false );
    ui.buttonStopSequence->setEnabled( false );
    ui.buttonAbortSequence->setEnabled( false );

    multiIndiSubscriber::onDisconnect();
}

void polarimetry::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    return handleSetProperty( ipRecv );
}

void polarimetry::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() == "hwptrack" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                std::string fsmString = ipRecv["state"].get<std::string>();
                m_hwptrackFsmOk       = fsmString == "READY" || fsmString == "OPERATING";
            }
        }
        else if( ipRecv.getName() == "hwp_position" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_hwpSetAngle = ipRecv["current"].get<float>();
            }
        }
        else if( ipRecv.getName() == "hwp_position_name" )
        {
            if( ipRecv.find( "value" ) )
            {
                m_hwpAngleName = ipRecv["value"].get<std::string>();
            }
        }
    }
    else if( ipRecv.getDevice() == "hwpsequence" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                std::string fsmString = ipRecv["state"].get<std::string>();
                m_hwpseqFsmOk         = fsmString == "READY" || fsmString == "OPERATING";
            }
        }
        else if( ipRecv.getName() == "sequence" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                m_sequencing = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
            }
        }
        else if( ipRecv.getName() == "status" )
        {
            // status elements are published as doubles
            if( ipRecv.find( "pos_index" ) )
            {
                m_hwpPosIndex = static_cast<int>( ipRecv["pos_index"].get<double>() );
            }

            if( ipRecv.find( "cycle" ) )
            {
                m_curCycle = static_cast<int>( ipRecv["cycle"].get<double>() );
            }

            if( ipRecv.find( "frames" ) )
            {
                m_framesLast = static_cast<int>( ipRecv["frames"].get<double>() );
            }
        }
        else if( ipRecv.getName() == "positions" )
        {
            if( ipRecv.find( "value" ) )
            {
                m_numPositions = countPositions( ipRecv["value"].get<std::string>() );
            }
        }
        else if( ipRecv.getName() == "last_cycle" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                ui.buttonLastCycle->setChecked( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On );
            }
        }
        else if( ipRecv.getName() == "abort" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                m_aborting = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
            }
        }
        else if( ipRecv.getName() == "num_cycles" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_numCycles = ipRecv["current"].get<int>();
            }
        }
        else if( ipRecv.getName() == "frames_per_pos" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_framesPerPos = ipRecv["current"].get<int>();
            }
        }
    }
    else if( ipRecv.getDevice() == "qwptrack" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                std::string fsmString = ipRecv["state"].get<std::string>();
                m_qwptrackFsmOk       = fsmString == "READY" || fsmString == "OPERATING";
            }
        }
        else if( ipRecv.getName() == "tracking" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                m_qwpTracking = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
            }
        }
        else if( ipRecv.getName() == "qwp1" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_qwp1Angle = ipRecv["current"].get<float>();
            }
        }
        else if( ipRecv.getName() == "qwp2" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_qwp2Angle = ipRecv["current"].get<float>();
            }
        }
    }
    else if( ipRecv.getDevice() == "qwppupil" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                std::string fsmString = ipRecv["state"].get<std::string>();
                m_qwppupilFsmOk       = fsmString == "READY" || fsmString == "OPERATING";
            }
        }
        else if( ipRecv.getName() == "tracking" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                m_qwppupilTracking = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
            }
        }
        else if( ipRecv.getName() == "offset" )
        {
            if( ipRecv.find( "v1" ) )
            {
                m_qwppupilV1 = ipRecv["v1"].get<float>();
            }

            if( ipRecv.find( "v2" ) )
            {
                m_qwppupilV2 = ipRecv["v2"].get<float>();
            }
        }
    }
    else
    {
        return;
    }

    emit doUpdateGUI();
}

void polarimetry::updateGUI()
{
    if( m_hwptrackFsmOk )
    {
        ui.hwpSetAngle->setText( QString( "%1°" ).arg( m_hwpSetAngle, 0, 'f', 1 ) );
        ui.hwpAngleName->setText( QString( m_hwpAngleName.c_str() ) );
    }
    else
    {
        ui.hwpSetAngle->setText( QString( "---" ) );
        ui.hwpAngleName->setText( QString( "---" ) );
    }

    setBold( ui.hwpSetAngle, m_hwptrackFsmOk );
    setBold( ui.hwpAngleName, m_hwptrackFsmOk );

    // disable things that we shouldn't change while sequencing
    ui.entryHwpAngle->setEnabled( !m_sequencing && m_hwptrackFsmOk );
    lockSlider( ui.sliderQwpTracking, m_sequencing || !m_qwptrackFsmOk );

    // pupil tracking may be toggled while sequencing
    lockSlider( ui.sliderQwpPupilTracking, !m_qwppupilFsmOk );

    ui.comboHwpLin->setEnabled( !m_sequencing );

    ui.sliderQwpTracking->setLabelEnabled( !m_sequencing && m_qwptrackFsmOk );
    if( m_qwptrackFsmOk )
    {
        // pad with figure spaces (digit width) so the label width is constant for angles in [0, 180)
        ui.labelQwpTracking->setText( QString( "QWP1: %1°  QWP2: %2°" )
                                          .arg( m_qwp1Angle, 3, 'f', 0, QChar( 0x2007 ) )
                                          .arg( m_qwp2Angle, 3, 'f', 0, QChar( 0x2007 ) ) );
    }
    else
    {
        ui.labelQwpTracking->setText( QString( "---" ) );
    }

    ui.labelQwpTracking->setVisible( m_qwpTracking );
    ui.labelQwpTracking->setEnabled( !m_sequencing && m_qwptrackFsmOk );

    ui.comboQwpLin->setEnabled( !m_sequencing );

    ui.sliderQwpPupilTracking->setLabelEnabled( m_qwppupilFsmOk );

    if( m_qwppupilFsmOk )
    {
        ui.labelQwpPupilTracking->setText( pupilOffsetText( m_qwppupilV1, m_qwppupilV2 ) );
    }
    else
    {
        ui.labelQwpPupilTracking->setText( QString( "---" ) );
    }

    ui.labelQwpPupilTracking->setVisible( m_qwppupilTracking );

    // num_cycles may be changed during a sequence, it is checked at the end of each cycle
    ui.entryNumCycles->setEnabled( m_hwpseqFsmOk );
    ui.labelNumCyclesHint->setEnabled( m_hwpseqFsmOk );

    ui.entryFramesPerPos->setEnabled( !m_sequencing && m_hwpseqFsmOk );
    ui.labelFramesHint->setEnabled( !m_sequencing && m_hwpseqFsmOk );

    // time_per_pos is ignored in frames mode
    bool timeMode = m_framesPerPos == 0;

    ui.entryTimePerPos->setEnabled( !m_sequencing && m_hwpseqFsmOk && timeMode );
    ui.labelTimeHint->setEnabled( !m_sequencing && m_hwpseqFsmOk && timeMode );

    ui.buttonStartSequence->setVisible( !m_sequencing );
    ui.buttonStartSequence->setEnabled( !m_sequencing && m_hwpseqFsmOk && m_hwptrackFsmOk );

    ui.buttonStopSequence->setVisible( m_sequencing );
    ui.buttonStopSequence->setEnabled( m_sequencing && m_hwpseqFsmOk );

    ui.buttonLastCycle->setVisible( m_sequencing );
    ui.buttonLastCycle->setEnabled( m_sequencing && m_hwpseqFsmOk );

    ui.buttonAbortSequence->setVisible( m_sequencing );
    ui.buttonAbortSequence->setEnabled( m_sequencing && m_hwpseqFsmOk && !m_aborting );

    // the frames count is only checked in frames mode
    bool showFrames = m_sequencing && !timeMode;

    ui.labelFramesStatus->setVisible( showFrames );
    ui.framesStatus->setVisible( showFrames );

    // status.frames is 0 until the first position completes, so there is no count to show yet
    if( showFrames && !( m_curCycle <= 1 && m_hwpPosIndex == 0 ) )
    {
        ui.framesStatus->setText( QString( "%1 / %2" ).arg( m_framesLast ).arg( m_framesPerPos ) );
        setWarning( ui.framesStatus, m_framesLast != m_framesPerPos );
    }
    else
    {
        ui.framesStatus->setText( QString( "---" ) );
        setWarning( ui.framesStatus, false );
    }

    if( m_sequencing )
    {
        if( m_numPositions > 0 )
        {
            ui.hwpPosIndex->setText( QString( "%1 / %2" ).arg( m_hwpPosIndex + 1 ).arg( m_numPositions ) );
        }
        else
        {
            ui.hwpPosIndex->setText( QString( "%1" ).arg( m_hwpPosIndex + 1 ) );
        }

        if( m_numCycles > 0 )
        {
            ui.cycleNumStatus->setText( QString( "%1 / %2" ).arg( m_curCycle ).arg( m_numCycles ) );
        }
        else
        {
            ui.cycleNumStatus->setText( QString( "%1" ).arg( m_curCycle ) );
        }
    }
    else
    {
        ui.hwpPosIndex->setText( QString( "---" ) );

        ui.cycleNumStatus->setText( QString( "---" ) );
    }

    setBold( ui.hwpPosIndex, m_sequencing );
    setBold( ui.cycleNumStatus, m_sequencing );
    setBold( ui.framesStatus, showFrames );

} //updateGUI()

void polarimetry::on_buttonStartSequence_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( "hwpsequence" );
    ip.setName( "sequence" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"] = pcf::IndiElement::On;
    sendNewProperty( ip );

    emit doUpdateGUI();
    return;
}

void polarimetry::on_buttonStopSequence_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( "hwpsequence" );
    ip.setName( "sequence" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"] = pcf::IndiElement::Off;
    sendNewProperty( ip );

    emit doUpdateGUI();
    return;
}

void polarimetry::on_buttonAbortSequence_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( "hwpsequence" );
    ip.setName( "abort" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"] = pcf::IndiElement::On;
    sendNewProperty( ip );

    emit doUpdateGUI();
    return;
}

void polarimetry::on_buttonLastCycle_clicked( bool checked )
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( "hwpsequence" );
    ip.setName( "last_cycle" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"] = checked ? pcf::IndiElement::On : pcf::IndiElement::Off;
    sendNewProperty( ip );

    emit doUpdateGUI();
    return;
}

void polarimetry::setBold( QLabel *label, bool onoff )
{
    QFont font = label->font();
    font.setBold( onoff );
    label->setFont( font );
}

void polarimetry::lockSlider( toggleSlider *slider, bool locked )
{
    slider->setAttribute( Qt::WA_TransparentForMouseEvents, locked );
    slider->setFocusPolicy( locked ? Qt::NoFocus : Qt::StrongFocus );
}

void polarimetry::setWarning( QLabel *label, bool onoff )
{
    if( label->property( "isWarning" ).toBool() == onoff )
    {
        return;
    }

    label->setProperty( "isWarning", onoff );
    label->style()->unpolish( label );
    label->style()->polish( label );
}

int polarimetry::countPositions( const std::string &positions )
{
    QStringList list = QString( positions.c_str() ).split( ',' );

    int n = 0;

    for( const QString &pos : list )
    {
        bool ok = false;

        pos.trimmed().toDouble( &ok );

        if( !ok )
        {
            return 0;
        }

        ++n;
    }

    return n;
}

QString polarimetry::pupilOffsetText( float v1, float v2 )
{
    auto fmt = []( float v )
    {
        QString num = QString::number( std::fabs( v ), 'f', 1 );

        // a value that rounds to zero is shown as +0.0, not -0.0
        QChar sign = ( v < 0 && num.toDouble() != 0 ) ? QChar( 0x2212 ) : QChar( '+' );

        return ( sign + num ).rightJustified( 6, QChar( 0x2007 ) );
    };

    return QString( "ΔV1: %1 V  ΔV2: %2 V" ).arg( fmt( v1 ) ).arg( fmt( v2 ) );
}

} //namespace xqt

#include "moc_polarimetry.cpp"

#endif
