/** \file pupilGuide.hpp
 * \brief Alignment and pupil-guiding control widget for the MagAO-X GUI suite.
 */
#ifndef pupilGuide_hpp
#define pupilGuide_hpp

#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <unistd.h>

#include <QTimer>
#include <QWidget>

#include "ui_pupilGuide.h"

#include "../xWidgets/statusEntry.hpp"
#include "../xWidgets/xWidget.hpp"

#define MOVE_TTM ( 0 )
#define MOVE_TEL ( 1 )
#define MOVE_WOOF ( 2 )

#define CAMLENS_X ( 0 )
#define CAMLENS_Y ( 1 )
#define CAMLENS_BOTH ( 2 )

namespace xqt
{

/// Convert requested x/y offsets into woofer tip/tilt coordinates.
void wooferTipTilt( double &tip,  /**< [out] Woofer tip-mode offset. */
                    double &tilt, /**< [out] Woofer tilt-mode offset. */
                    double  x,    /**< [in] Requested horizontal offset. */
                    double  y /**< [in] Requested vertical offset. */ )
{
    double rot   = ( 180. + 29.0 ) * 3.14159 / 180.;
    double scale = -1.0;

    tip  = scale * ( x * cos( rot ) - y * sin( rot ) );
    tilt = scale * ( x * sin( rot ) + y * cos( rot ) );
}

/// Main pupil-guiding and alignment control panel.
/** The shared INDI connection controls transport availability; individual devices and
 * properties control only their dependent actions. Callbacks and command handlers
 * run on the Qt widget thread. Fresh valid observations restore controls without
 * replaying commands or requiring another connection callback.
 */
class pupilGuide : public xWidget
{
    Q_OBJECT

    /// Reference camera used for the pupil and periscope coordinate transforms.
    enum camera
    {
        FLOWFS,
        LLOWFS,
        CAMSCIS
    };

  protected:
    bool m_connected{ false }; ///< True while the shared INDI transport is connected.

    std::string m_appState; ///< Legacy application-state cache, cleared on connection loss.

    // --- modttm
    std::string m_modFsmState; ///< Last observed modulator FSM state.

    int m_modState{ 0 }; ///< Last observed modulator operating mode.

    double m_modCh1{ 0 }; ///< Legacy modulator channel-one cache.

    double m_modCh2{ 0 }; ///< Legacy modulator channel-two cache.

    double m_camwfsFreq{ 0 }; ///< Last received WFS camera frame rate.

    double m_modRad{ 0 }; ///< Legacy current modulation-radius cache.

    double m_modRad_tgt{ 0 }; ///< Legacy target modulation-radius cache.

    float m_stepSize{ 0.1 }; ///< Local step size for tip/tilt nudges.

    int m_tipmovewhat{ MOVE_WOOF }; ///< Selected tip/tilt actuator; preserved across device loss.

    float m_focusWooferStep{ 0.2 }; ///< Conversion from the focus step multiplier to woofer mode amplitude.

    float m_focusTelStep{ 100 }; ///< Conversion from the focus step multiplier to telescope focus offset.

    // --- TCS

    std::string m_tcsiState; ///< Last observed TCS interface FSM state.

    bool m_labMode{ true }; ///< Effective lab mode, true while TCS or its mode observation is unavailable.

    bool m_reportedLabMode{ true }; ///< Last valid TCS mode, used only while TCS is available.

    // --- woofer

    std::string m_dmWooferState; ///< Last observed woofer controller FSM state.

    std::string m_wooferModesState; ///< Last observed woofer modal-command FSM state.

    double m_tilt{ 0 }; ///< Last valid woofer tilt amplitude; usable only while its observation remains valid.

    double m_tip{ 0 }; ///< Last valid woofer tip amplitude; usable only while its observation remains valid.

    double m_focus{ 0 }; ///< Last valid woofer focus amplitude, rounded to INDI command precision.

    float m_focusStepSize{ 0.1 }; ///< Local step multiplier used by both focus actuators.

    // --- picoscix
    std::string m_picoState{ "UNKNOWN" }; ///< Last observed picomotor controller FSM state.

    int m_picoscixPos{ -1000000000 }; ///< Last valid picoscix position; the initial sentinel cannot be used for motion.

    int m_picoscix_stepSize{ 50 }; ///< Local picomotor nudge size in controller steps.

    std::string m_picoscix_gotoSelection; ///< Legacy picomotor preset-selection cache.

    // --- camwfs-fit
    std::string m_camwfsfitState; ///< Last observed pupil-fitting application FSM state.

    double m_med1{ 0 }; ///< Last valid median flux in fitted quadrant 1.

    double m_med2{ 0 }; ///< Last valid median flux in fitted quadrant 2.

    double m_med3{ 0 }; ///< Last valid median flux in fitted quadrant 3.

    double m_med4{ 0 }; ///< Last valid median flux in fitted quadrant 4.

    double m_x1{ 0 }; ///< Last valid horizontal center of fitted quadrant 1.

    double m_y1{ 0 }; ///< Last valid vertical center of fitted quadrant 1.

    double m_D1{ 0 }; ///< Last valid diameter of fitted quadrant 1.

    double m_setx1{ 0 }; ///< Last valid reference horizontal center for quadrant 1 deltas.

    double m_sety1{ 0 }; ///< Last valid reference vertical center for quadrant 1 deltas.

    double m_setD1{ 0 }; ///< Last valid reference diameter for quadrant 1 deltas.

    double m_x2{ 0 }; ///< Last valid horizontal center of fitted quadrant 2.

    double m_y2{ 0 }; ///< Last valid vertical center of fitted quadrant 2.

    double m_D2{ 0 }; ///< Last valid diameter of fitted quadrant 2.

    double m_setx2{ 0 }; ///< Last valid reference horizontal center for quadrant 2 deltas.

    double m_sety2{ 0 }; ///< Last valid reference vertical center for quadrant 2 deltas.

    double m_setD2{ 0 }; ///< Last valid reference diameter for quadrant 2 deltas.

    double m_x3{ 0 }; ///< Last valid horizontal center of fitted quadrant 3.

    double m_y3{ 0 }; ///< Last valid vertical center of fitted quadrant 3.

    double m_D3{ 0 }; ///< Last valid diameter of fitted quadrant 3.

    double m_setx3{ 0 }; ///< Last valid reference horizontal center for quadrant 3 deltas.

    double m_sety3{ 0 }; ///< Last valid reference vertical center for quadrant 3 deltas.

    double m_setD3{ 0 }; ///< Last valid reference diameter for quadrant 3 deltas.

    double m_x4{ 0 }; ///< Last valid horizontal center of fitted quadrant 4.

    double m_y4{ 0 }; ///< Last valid vertical center of fitted quadrant 4.

    double m_D4{ 0 }; ///< Last valid diameter of fitted quadrant 4.

    double m_setx4{ 0 }; ///< Last valid reference horizontal center for quadrant 4 deltas.

    double m_sety4{ 0 }; ///< Last valid reference vertical center for quadrant 4 deltas.

    double m_setD4{ 0 }; ///< Last valid reference diameter for quadrant 4 deltas.

    double m_threshold_current{ 0 }; ///< Last received pupil-fit threshold.

    double m_threshold_target{ 0 }; ///< Legacy target pupil-fit threshold cache.

    // -- camwfs-avg
    std::string m_camwfsavgState; ///< Last observed WFS averaging application FSM state.

    unsigned m_nAverage_current{ 0 }; ///< Last received WFS averaging count.

    unsigned m_nAverage_target{ 0 }; ///< Legacy target WFS averaging count cache.

    // -- dmtweeter
    std::string m_dmtweeterState; ///< Last observed tweeter controller FSM state.

    bool m_dmtweeterTestSet{ false }; ///< Last known tweeter test-pattern switch; validity is tracked separately.

    // -- dmncpc
    std::string m_dmncpcState; ///< Last observed NCPC controller FSM state.

    bool m_dmncpcTestSet{ false }; ///< Last known NCPC test-pattern switch; validity is tracked separately.

    // -- ttmpupil
    std::string m_pupFsmState; ///< Last observed pupil TTM FSM state.

    double m_pupCh1{ 0 }; ///< Last valid pupil TTM first-axis position.

    double m_pupCh2{ 0 }; ///< Last valid pupil TTM second-axis position.

    float m_pupStepSize{ 0.5 }; ///< Local pupil-steering nudge size.

    int m_pupCam{ CAMSCIS }; ///< Camera coordinate transform selected for pupil and periscope nudges.

    // -- ttmperi
    std::string m_ttmPeriFsmState; ///< Last observed periscope TTM FSM state.

    double m_ttmPeriCh1{ 0 }; ///< Last valid periscope first-axis voltage.

    double m_ttmPeriCh2{ 0 }; ///< Last valid periscope second-axis voltage.

    float m_ttmPeriStepSize{ 25 }; ///< Local periscope TTM voltage step.

    // -- Camera Lens
    std::string m_camlensxFsmState; ///< Last observed camera-lens X-stage FSM state.

    std::string m_camlensyFsmState; ///< Last observed camera-lens Y-stage FSM state.

    float m_camlensx_pos{ 0 }; ///< Last valid camera-lens X position used as the relative-motion baseline.

    float m_camlensy_pos{ 0 }; ///< Last valid camera-lens Y position used as the relative-motion baseline.

    float m_camlensStepSize{ 0.01 }; ///< Local camera-lens movement step in millimeters.

    // ****** Alignment ******** //

    // --- camwfs-align
    std::string m_camwfs_align_fsmState; ///< Last observed pupil-tracking controller FSM state.

    // --- twAlign-camwfs-ctrl
    std::string m_twAlign_camwfs_ctrl_fsmState; ///< Last observed actuator-alignment controller FSM state.

    // --- twAlign-camwfs-wfs
    std::string m_twAlign_camwfs_wfs_fsmState; ///< Last observed actuator-alignment sensor FSM state.

  public:
    /// Construct the pupil-guide control widget.
    pupilGuide( QWidget        *Parent = 0, /**< [in] Optional Qt parent owning this widget. */
                Qt::WindowFlags f      = Qt::WindowFlags() /**< [in] Qt window flags. */ );

    /// Destroy the pupil-guide control widget.
    ~pupilGuide();

    /// Register the widget and its child subscribers with the shared INDI manager.
    void subscribe();

    /// Apply connected-state GUI initialization after the shared INDI layer reconnects.
    virtual void onConnect();

    /// Apply disconnected-state GUI initialization and clear cached connection state.
    virtual void onDisconnect();

    /// Replace the observed definition of a known property and refresh dependent controls.
    void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] Property definition or value update. */ );

    /// Invalidate only the deleted property or device and its dependent controls.
    void handleDelProperty(
        const pcf::IndiProperty &ipRecv /**< [in] Deleted property; an empty name means the entire device. */ );

    /// Merge a partial property update and refresh dependent controls.
    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] Property definition or value update. */ );

    /// Apply modulator FSM availability to its controls and the selected motion buttons.
    void modGUISetEnable( bool enableModGUI, /**< [in] Modulator section availability. */
                          bool enableModArrows /**< [in] Selected tip/tilt actuator availability. */ );

    /// Set availability of the pupil-fitting readouts and related controls.
    void camwfsfitSetEnabled( bool enabled /**< [in] Whether fitting readouts are available. */ );

    /// Set availability of camera-lens axes and their shared scale control.
    void camlensSetEnabled( bool enabled, /**< [in] Whether the selected axes are available. */
                            int  whichcl = CAMLENS_BOTH /**< [in] CAMLENS_X, CAMLENS_Y, or CAMLENS_BOTH. */ );

    /// Set availability of the pupil-tracking section.
    void camwfs_align_setEnabled( bool enabled, /**< [in] Whether the section controls are available. */
                                  bool all /**< [in] Also update the section label. */ );

    /// Set availability of the actuator-alignment controller section.
    void twAlign_camwfs_ctrl_setEnabled( bool enabled, /**< [in] Whether the section controls are available. */
                                         bool all /**< [in] Also update the section label. */ );

    /// Set availability of the actuator-alignment sensor section.
    void twAlign_camwfs_wfs_setEnabled( bool enabled, /**< [in] Whether the section controls are available. */
                                        bool all /**< [in] Also update the section label. */ );

    /// Initialize availability of the combined alignment controls.
    void alignment_buttons_setEnabled( bool enabled, /**< [in] Whether the section controls are available. */
                                       bool all /**< [in] Also update the section label. */ );

  public slots:
    /// Refresh all controls from transport, FSM, and observed-property availability.
    void updateGUI();

    //----------- modttm

    /// Request modulation when the modulator command is available.
    void on_buttonMod_mod_pressed();

    /// Request the set position when the modulator command is available.
    void on_buttonMod_set_pressed();

    /// Request the rest position when the modulator command is available.
    void on_buttonMod_rest_pressed();

    /// Cycle the selected tip/tilt actuator, excluding the telescope in lab mode.
    void on_button_ttmtel_pressed();

    /// Nudge the selected tip/tilt actuator upward when its inputs are available.
    void on_button_tip_u_pressed();

    /// Nudge the selected tip/tilt actuator up and left when its inputs are available.
    void on_button_tip_ul_pressed();

    /// Nudge the selected tip/tilt actuator left when its inputs are available.
    void on_button_tip_l_pressed();

    /// Nudge the selected tip/tilt actuator down and left when its inputs are available.
    void on_button_tip_dl_pressed();

    /// Nudge the selected tip/tilt actuator downward when its inputs are available.
    void on_button_tip_d_pressed();

    /// Nudge the selected tip/tilt actuator down and right when its inputs are available.
    void on_button_tip_dr_pressed();

    /// Nudge the selected tip/tilt actuator right when its inputs are available.
    void on_button_tip_r_pressed();

    /// Nudge the selected tip/tilt actuator up and right when its inputs are available.
    void on_button_tip_ur_pressed();

    /// Cycle the local tip/tilt step size.
    void on_button_tip_scale_pressed();

    //------------- focus
    /// Increase focus on the selected actuator when its required inputs are available.
    void on_button_focus_p_pressed();

    /// Decrease focus on the selected actuator when its required inputs are available.
    void on_button_focus_m_pressed();

    /// Cycle the local focus step multiplier.
    void on_button_focus_scale_pressed();

    //------------- picoscix
    /// Move picoscix relative to a valid observed position without integer overflow.
    void move_picoscix( int delta /**< [in] Signed relative movement in picomotor steps. */ );

    /// Nudge picoscix in the positive direction by the selected step size.
    void on_picoscix_l_pressed();

    /// Cycle the local picomotor step size.
    void on_picoscix_scale_pressed();

    /// Nudge picoscix in the negative direction by the selected step size.
    void on_picoscix_r_pressed();

    /// Apply the selected beamsplitter offset and clear the preset selection.
    void on_picoscix_go_pressed();

    //----------- dmtweeter
    /// Toggle the tweeter test pattern from a known observed switch state.
    void on_buttonTweeterTest_set_pressed();

    //----------- dmncpc
    /// Toggle the NCPC test pattern from a known observed switch state.
    void on_buttonNCPCTest_set_pressed();

    //----------- ttmpupil
    /// Request release of the pupil TTM when that capability is available.
    void on_buttonPup_rest_pressed();

    /// Request initialization of the pupil TTM when that capability is available.
    void on_buttonPup_set_pressed();

    /// Cycle the camera used to choose pupil and periscope movement directions.
    void on_button_camera_pressed();

    /// Nudge the pupil TTM up and left when its inputs are available.
    void on_button_pup_ul_pressed();

    /// Nudge the pupil TTM down and left when its inputs are available.
    void on_button_pup_dl_pressed();

    /// Nudge the pupil TTM down and right when its inputs are available.
    void on_button_pup_dr_pressed();

    /// Nudge the pupil TTM up and right when its inputs are available.
    void on_button_pup_ur_pressed();

    /// Cycle the local pupil-steering step size.
    void on_button_pup_scale_pressed();

    //---------- TTM Peri
    /// Request the periscope TTM rest state when its switch is available.
    void on_button_ttmPeri_rest_pressed();

    /// Request the periscope TTM set state when its switch is available.
    void on_button_ttmPeri_set_pressed();

    /// Nudge the periscope TTM left when its inputs are available.
    void on_button_ttmPeri_l_pressed();

    /// Nudge the periscope TTM right when its inputs are available.
    void on_button_ttmPeri_r_pressed();

    /// Nudge the periscope TTM upward when its inputs are available.
    void on_button_ttmPeri_u_pressed();

    /// Nudge the periscope TTM downward when its inputs are available.
    void on_button_ttmPeri_d_pressed();

    /// Cycle the local periscope TTM step size.
    void on_button_ttmPeri_scale_pressed();

    /// Show or hide the individual pupil-fit coordinates.
    void toggleExpFit( bool visible /**< [in] Whether to show the individual pupil-fit coordinates. */ );

    /// Toggle the expanded pupil-fit coordinate display.
    void on_buttonExpFit_pressed();

    /// Nudge the camera lens upward when its inputs are available.
    void on_button_camlens_u_pressed();

    /// Nudge the camera lens left when its inputs are available.
    void on_button_camlens_l_pressed();

    /// Nudge the camera lens downward when its inputs are available.
    void on_button_camlens_d_pressed();

    /// Nudge the camera lens right when its inputs are available.
    void on_button_camlens_r_pressed();

    /// Cycle the local camera-lens step size.
    void on_button_camlens_scale_pressed();

    // ******** alignment *********//

    /// Start all three alignment participants only when every dependency is available.
    void on_button_startAlignment_pressed();

    /// Stop each available alignment target, preserving the lab-mode routing rule.
    void on_button_stopAlignment_pressed();

  private:
    /// Binding used to invalidate only the child belonging to a deleted device/property.
    struct childSubscription
    {
        multiIndiSubscriber *subscriber; ///< Child owned by the Qt widget tree.

        const char *device; ///< INDI device supplying the child.

        const char *property; ///< INDI property supplying the child.

        const char *element; ///< Required observed element for this child.
    };

    /// Observations for one of this panel's known INDI properties.
    struct propertyState
    {
        pcf::IndiProperty::Type type; ///< Expected protocol type from the producer contract.

        std::set<std::string> elements; ///< Observed element names, including empty command targets.

        std::set<std::string> valid; ///< Elements with valid values since the last invalidation.
    };

    /// Device/property observations; numeric caches are usable only with corresponding valid elements.
    std::map<std::pair<std::string, std::string>, propertyState> m_properties;

    /// Populate the panel's subscription and availability inventory.
    void initializeProperties();

    /// Check all required elements for a known property on the current connection.
    bool
    propertyAvailable( const std::string                  &device,   /**< [in] INDI device. */
                       const std::string                  &property, /**< [in] INDI property. */
                       std::initializer_list<const char *> elements, /**< [in] Required elements. */
                       bool values = true /**< [in] Require valid values, or just command element presence. */ ) const;

    /// Validate observed elements before updating the existing value caches.
    void receiveProperty( const pcf::IndiProperty &property, /**< [in] Incoming property. */
                          bool definition /**< [in] True replaces the previous definition; false merges an update. */ );

    /// Apply per-property constraints after the existing FSM-based GUI update.
    void updateAvailability();

    /// Refresh eligibility before a command handler uses its control.
    bool commandEnabled( QWidget *control /**< [in] Control whose command is requested. */ );

    /// Apply a callable to each cached device FSM.
    template <typename funcT>
    void forEachDeviceState( funcT &&func /**< [in] Callable receiving device name and state reference. */ );

    /// Clear affected child displays and pending edits without altering subscriptions.
    void resetChild( multiIndiSubscriber *sub /**< [in,out] Child to invalidate. */ );

    /// Applies a callable to each child binding subscribed directly through this GUI.
    template <typename funcT>
    void forEachChildSubscriber( funcT &&func /**< [in] Callable applied to each direct child subscriber. */ );

    /// Resets cached connection-dependent state so timer-driven GUI updates stay disconnected-safe.
    void resetConnectionState();

    Ui::pupilGuide ui; ///< Generated Qt UI facade for the pupil-guide widget layout.
};

pupilGuide::pupilGuide( QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f )
{
    char ss[64]; // for scale buttons

    ui.setupUi( this );

    ui.button_focus_scale->setProperty( "isScaleButton", true );
    ui.button_pup_scale->setProperty( "isScaleButton", true );
    ui.button_ttmPeri_scale->setProperty( "isScaleButton", true );

    //-----------modwfs controls ------------

    setXwFont( ui.label_modulation );

    ui.modwfs_fsm->device( "modwfs" );
    ui.modwfs_fsm->NOTHOMED( "RIP" );
    ui.modwfs_fsm->READY( "SET" );
    ui.modwfs_fsm->OPERATING( "MODULATING" );

    setXwFont( ui.label_modFreq );
    setXwFont( ui.label_modRad );

    setXwFont( ui.buttonMod_rest );
    setXwFont( ui.buttonMod_set );
    setXwFont( ui.buttonMod_mod );

    ui.modFreq_current->setup( "modwfs", "modFrequency", statusEntry::FLOAT, "", "" );
    ui.modFreq_current->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.modFreq_current->format( "%0.1f" );
    // ui.modFreq_current->onDisconnect();

    ui.modRad_current->setup( "modwfs", "modRadius", statusEntry::FLOAT, "", "" );
    ui.modRad_current->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.modRad_current->format( "%0.1f" );
    // ui.modRad_current->onDisconnect();

    ui.modCh1->setup( "fxngenmodwfs", "C1ofst", statusEntry::FLOAT, "Ch1", "V" );
    ui.modCh1->currEl( "value" );
    ui.modCh1->targEl( "value" );
    ui.modCh1->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.modCh1->format( "%0.2f" );
    // ui.modCh1->onDisconnect();

    ui.modCh2->setup( "fxngenmodwfs", "C2ofst", statusEntry::FLOAT, "Ch2", "V" );
    ui.modCh2->currEl( "value" );
    ui.modCh2->targEl( "value" );
    ui.modCh2->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.modCh2->format( "%0.2f" );
    // ui.modCh2->onDisconnect();

    //-----------tip alignment controls ------------

    setXwFont( ui.picoscix_label );

    ui.picoscix_pos->setup( "picomotors", "picoscix_pos", statusEntry::INT, "", "" );
    ui.picoscix_pos->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.picoscix_pos->format( "%d" );

    ui.picoscix_scale->setProperty( "isScaleButton", true );
    snprintf( ss, 5, "%0.2f", m_picoscix_stepSize / 1000. );
    ui.button_tip_scale->setText( ss );

    ui.picoscix_combo->addItem( "    " );
    ui.picoscix_combo->addItem( "65-35" );
    ui.picoscix_combo->addItem( "Ha-IR" );
    ui.picoscix_combo->setCurrentText( "    " );

    setXwFont( ui.picoscix_combo_label );

    //-----------orphans ------------
    m_tipmovewhat = MOVE_TTM; // This will make it be "move woofer" on startup
    on_button_ttmtel_pressed();

    ui.button_tip_scale->setProperty( "isScaleButton", true );
    snprintf( ss, 5, "%0.2f", m_stepSize );
    ui.button_tip_scale->setText( ss );

    snprintf( ss, 5, "%0.2f", m_focusStepSize );
    ui.button_focus_scale->setText( ss );

    //-----------picoscix controls ------------
    setXwFont( ui.label_tipAlignment );

    // orphans:

    snprintf( ss, 5, "%0.2f", m_pupStepSize );
    ui.button_pup_scale->setText( ss );

    snprintf( ss, 5, "%0.2f", m_camlensStepSize * 10 );
    ui.button_camlens_scale->setText( ss );

    snprintf( ss, 5, "%0.2f", m_ttmPeriStepSize / 100. );
    ui.button_ttmPeri_scale->setText( ss );

    setXwFont( ui.labelMedianFluxes );
    setXwFont( ui.med1 );
    setXwFont( ui.med2 );
    setXwFont( ui.med3 );
    setXwFont( ui.med4 );
    setXwFont( ui.setDelta );

    // tweeter controls
    setXwFont( ui.label_tweeter );
    setXwFont( ui.buttonTweeterTest_set );

    // ncpc controls
    setXwFont( ui.label_ncpc );
    setXwFont( ui.buttonNCPCTest_set );

    //-----------ttmpupil controls ------------
    setXwFont( ui.labelPupilSteering );
    setXwFont( ui.buttonPup_rest );
    setXwFont( ui.buttonPup_set );
    ui.pupState->device( "ttmpupil" );
    ui.pupState->NOTHOMED( "RIP" );
    ui.pupState->HOMING( "SETTING" );
    ui.pupState->READY( "SET" );

    ui.pupCh1->setup( "ttmpupil", "pos_1", statusEntry::FLOAT, "Ch 1", "V" );
    ui.pupCh1->setStretch( 1, 2, 4 );
    ui.pupCh1->highlightChanges( false );

    ui.pupCh2->setup( "ttmpupil", "pos_2", statusEntry::FLOAT, "Ch 2", "V" );
    ui.pupCh2->setStretch( 1, 2, 4 );
    ui.pupCh2->highlightChanges( false );

    //-----------ttmperi controls ------------
    setXwFont( ui.labelTTMPeri );
    setXwFont( ui.buttonPup_rest );
    setXwFont( ui.buttonPup_set );
    ui.ttmPeriState->device( "ttmperi" );
    ui.ttmPeriState->READY( "RIP" );
    ui.ttmPeriState->OPERATING( "SET" );

    ui.ttmPeriCh1->setup( "ttmperi", "axis1_voltage", statusEntry::FLOAT, "Ch 1", "V" );
    ui.ttmPeriCh1->setStretch( 1, 2, 4 );
    ui.ttmPeriCh1->highlightChanges( false );

    ui.ttmPeriCh2->setup( "ttmperi", "axis2_voltage", statusEntry::FLOAT, "Ch 2", "V" );
    ui.ttmPeriCh2->highlightChanges( false );
    ui.ttmPeriCh2->setStretch( 1, 2, 4 );

    /* pupil tracking loop */
    setXwFont( ui.label_pupTrackLoop );

    ui.pupTrackLoop_deltaX->setup( "camwfs-align", "deltas", statusEntry::FLOAT, "", "" );
    ui.pupTrackLoop_deltaX->currEl( "delta0" );
    ui.pupTrackLoop_deltaX->highlightChanges( false );
    ui.pupTrackLoop_deltaX->readOnly( true );
    ui.pupTrackLoop_deltaX->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.pupTrackLoop_deltaX->format( "%0.03f" );

    ui.pupTrackLoop_deltaY->setup( "camwfs-align", "deltas", statusEntry::FLOAT, "", "" );
    ui.pupTrackLoop_deltaY->currEl( "delta1" );
    ui.pupTrackLoop_deltaY->highlightChanges( false );
    ui.pupTrackLoop_deltaY->readOnly( true );
    ui.pupTrackLoop_deltaY->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.pupTrackLoop_deltaY->format( "%0.03f" );

    ui.pupTrackLoop_slider->setup( "camwfs-align", "loop_state", "toggle", "" );
    ui.pupTrackLoop_slider->setStretch( 0, 0, 10, true, true );

    ui.pupTrackLoop_gain->setup( "camwfs-align", "loop_gain", statusEntry::FLOAT, "loop gain", "" );
    ui.pupTrackLoop_gain->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.pupTrackLoop_gain->format( "%0.2f" );

    /* actuator alignment loop */
    setXwFont( ui.label_actAlignLoop );

    ui.actAlignLoop_deltaX->setup( "twAlign-camwfs-ctrl", "deltas", statusEntry::FLOAT, "", "" );
    ui.actAlignLoop_deltaX->currEl( "delta0" );
    ui.actAlignLoop_deltaX->highlightChanges( false );
    ui.actAlignLoop_deltaX->readOnly( true );
    ui.actAlignLoop_deltaX->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.actAlignLoop_deltaX->format( "%0.03f" );

    ui.actAlignLoop_deltaY->setup( "twAlign-camwfs-ctrl", "deltas", statusEntry::FLOAT, "", "" );
    ui.actAlignLoop_deltaY->currEl( "delta1" );
    ui.actAlignLoop_deltaY->highlightChanges( false );
    ui.actAlignLoop_deltaY->readOnly( true );
    ui.actAlignLoop_deltaY->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.actAlignLoop_deltaY->format( "%0.03f" );

    ui.actAlignLoop_slider->setup( "twAlign-camwfs-ctrl", "loop_state", "toggle", "" );
    ui.actAlignLoop_slider->setStretch( 0, 0, 10, true, true );

    ui.actAlignLoop_gain->setup( "twAlign-camwfs-ctrl", "loop_gain", statusEntry::FLOAT, "loop gain", "" );
    ui.actAlignLoop_gain->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.actAlignLoop_gain->format( "%0.2f" );

    /* actuator alignment sensor */
    setXwFont( ui.label_actAlignSensor );

    ui.actAlignSensor_slider->setup( "twAlign-camwfs-wfs", "continuous", "toggle", "" );
    ui.actAlignSensor_slider->setStretch( 0, 0, 10, true, true );

    ui.actAlignSensor_nAverage->setup( "twAlign-camwfs-wfs", "nPokeAverage", statusEntry::INT, "no. average", "" );
    ui.actAlignSensor_nAverage->setStretch( 1, 3, 6 );
    ui.actAlignSensor_nAverage->format( "%d" );

    ui.actAlignSensor_nImages->setup( "twAlign-camwfs-wfs", "nPokeImages", statusEntry::INT, "no. images", "" );
    ui.actAlignSensor_nImages->setStretch( 1, 3, 6 );
    ui.actAlignSensor_nImages->format( "%d" );

    ui.actAlignSensor_pokeAmp->setup( "twAlign-camwfs-wfs", "poke_amp", statusEntry::FLOAT, "poke amp.", "um" );
    ui.actAlignSensor_pokeAmp->setStretch( 1, 3, 6 );
    ui.actAlignSensor_pokeAmp->format( "%0.2f" );

    /* alignment start/stop */
    setXwFont( ui.label_alignment );
    setXwFont( ui.button_startAlignment );
    setXwFont( ui.button_stopAlignment );

    setXwFont( ui.labelPupilFitting ); //,1.2);

    setXwFont( ui.label_pupilPositions );
    setXwFont( ui.labelx );
    setXwFont( ui.labely );
    setXwFont( ui.labelD );
    setXwFont( ui.labelUR );
    setXwFont( ui.labelUL );
    setXwFont( ui.labelLR );
    setXwFont( ui.labelLL );
    setXwFont( ui.labelAvg );
    setXwFont( ui.coordUR_x );
    setXwFont( ui.coordUR_y );
    setXwFont( ui.coordUR_D );
    setXwFont( ui.coordUL_x );
    setXwFont( ui.coordUL_y );
    setXwFont( ui.coordUL_D );
    setXwFont( ui.coordLR_x );
    setXwFont( ui.coordLR_y );
    setXwFont( ui.coordLR_D );
    setXwFont( ui.coordLL_x );
    setXwFont( ui.coordLL_y );
    setXwFont( ui.coordLL_D );
    setXwFont( ui.coordAvg_x );
    setXwFont( ui.coordAvg_y );
    setXwFont( ui.coordAvg_D );

    setXwFont( ui.setDelta_pup );

    ui.fitThreshold->setup( "camwfs-fit", "threshold", statusEntry::FLOAT, "Thresh", "" );
    ui.fitThreshold->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.fitThreshold->format( "%0.3f" );

    ui.fitAvgTime->setup( "camwfs-avg", "avgTime", statusEntry::FLOAT, "Avg. T.", "s" );
    ui.fitAvgTime->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.fitAvgTime->format( "%0.3f" );

    /* Camera Lens */
    setXwFont( ui.label_camlens );
    setXwFont( ui.label_camlensX_fsm );
    setXwFont( ui.label_camlensY_fsm );
    ui.camlensX_fsm->device( "stagecamlensx" );
    ui.camlensY_fsm->device( "stagecamlensy" );

    ui.camlensX_pos->setup( "stagecamlensx", "position", statusEntry::FLOAT, "X", "mm" );
    ui.camlensX_pos->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.camlensX_pos->format( "%0.4f" );

    ui.camlensY_pos->setup( "stagecamlensy", "position", statusEntry::FLOAT, "Y", "mm" );
    ui.camlensY_pos->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.camlensY_pos->format( "%0.4f" );

    ui.button_camlens_scale->setProperty( "isScaleButton", true );

    // Set the pupil fit boxes to invisible at startup
    toggleExpFit( false );

    initializeProperties();
    onDisconnect();

    QTimer *timer = new QTimer( this );
    connect( timer, SIGNAL( timeout() ), this, SLOT( updateGUI() ) );
    timer->start( 250 );
}

pupilGuide::~pupilGuide()
{
}

template <typename funcT>
void pupilGuide::forEachChildSubscriber( funcT &&func )
{
    std::array<childSubscription, 30> subs{ {
        { ui.modwfs_fsm, "modwfs", "fsm", "state" },
        { ui.modFreq_current, "modwfs", "modFrequency", "current" },
        { ui.modRad_current, "modwfs", "modRadius", "current" },
        { ui.modCh1, "fxngenmodwfs", "C1ofst", "value" },
        { ui.modCh2, "fxngenmodwfs", "C2ofst", "value" },
        { ui.picoscix_pos, "picomotors", "picoscix_pos", "current" },
        { ui.pupState, "ttmpupil", "fsm", "state" },
        { ui.pupCh1, "ttmpupil", "pos_1", "current" },
        { ui.pupCh2, "ttmpupil", "pos_2", "current" },
        { ui.ttmPeriState, "ttmperi", "fsm", "state" },
        { ui.ttmPeriCh1, "ttmperi", "axis1_voltage", "current" },
        { ui.ttmPeriCh2, "ttmperi", "axis2_voltage", "current" },
        { ui.pupTrackLoop_deltaX, "camwfs-align", "deltas", "delta0" },
        { ui.pupTrackLoop_deltaY, "camwfs-align", "deltas", "delta1" },
        { ui.pupTrackLoop_slider, "camwfs-align", "loop_state", "toggle" },
        { ui.pupTrackLoop_gain, "camwfs-align", "loop_gain", "current" },
        { ui.actAlignLoop_deltaX, "twAlign-camwfs-ctrl", "deltas", "delta0" },
        { ui.actAlignLoop_deltaY, "twAlign-camwfs-ctrl", "deltas", "delta1" },
        { ui.actAlignLoop_slider, "twAlign-camwfs-ctrl", "loop_state", "toggle" },
        { ui.actAlignLoop_gain, "twAlign-camwfs-ctrl", "loop_gain", "current" },
        { ui.actAlignSensor_slider, "twAlign-camwfs-wfs", "continuous", "toggle" },
        { ui.actAlignSensor_nAverage, "twAlign-camwfs-wfs", "nPokeAverage", "current" },
        { ui.actAlignSensor_nImages, "twAlign-camwfs-wfs", "nPokeImages", "current" },
        { ui.actAlignSensor_pokeAmp, "twAlign-camwfs-wfs", "poke_amp", "current" },
        { ui.fitThreshold, "camwfs-fit", "threshold", "current" },
        { ui.fitAvgTime, "camwfs-avg", "avgTime", "current" },
        { ui.camlensX_fsm, "stagecamlensx", "fsm", "state" },
        { ui.camlensY_fsm, "stagecamlensy", "fsm", "state" },
        { ui.camlensX_pos, "stagecamlensx", "position", "current" },
        { ui.camlensY_pos, "stagecamlensy", "position", "current" },
    } };

    for( const auto &sub : subs )
    {
        func( sub );
    }
}

template <typename funcT>
void pupilGuide::forEachDeviceState( funcT &&func )
{
    func( "modwfs", m_modFsmState );
    func( "tcsi", m_tcsiState );
    func( "dmwoofer", m_dmWooferState );
    func( "wooferModes", m_wooferModesState );
    func( "picomotors", m_picoState );
    func( "camwfs-fit", m_camwfsfitState );
    func( "camwfs-avg", m_camwfsavgState );
    func( "dmtweeter", m_dmtweeterState );
    func( "dmncpc", m_dmncpcState );
    func( "ttmpupil", m_pupFsmState );
    func( "ttmperi", m_ttmPeriFsmState );
    func( "stagecamlensx", m_camlensxFsmState );
    func( "stagecamlensy", m_camlensyFsmState );
    func( "camwfs-align", m_camwfs_align_fsmState );
    func( "twAlign-camwfs-ctrl", m_twAlign_camwfs_ctrl_fsmState );
    func( "twAlign-camwfs-wfs", m_twAlign_camwfs_wfs_fsmState );
}

void pupilGuide::resetChild( multiIndiSubscriber *sub )
{
    if( auto *entry = dynamic_cast<statusEntry *>( sub ) )
    {
        entry->setEditText( "" );
        entry->stopEditing();
    }
    sub->onDisconnect();
}

void pupilGuide::initializeProperties()
{
    const auto add = [this]( const char *device, const char *property, pcf::IndiProperty::Type type )
    { m_properties.emplace( std::make_pair( device, property ), propertyState{ type, {}, {} } ); };
    add( "modwfs", "fsm", pcf::IndiProperty::Text );
    add( "modwfs", "modState", pcf::IndiProperty::Number );
    add( "camwfs", "fps", pcf::IndiProperty::Number );
    add( "tcsi", "fsm", pcf::IndiProperty::Text );
    add( "tcsi", "labMode", pcf::IndiProperty::Switch );
    add( "dmwoofer", "fsm", pcf::IndiProperty::Text );
    add( "wooferModes", "fsm", pcf::IndiProperty::Text );
    add( "wooferModes", "current_amps", pcf::IndiProperty::Number );
    add( "picomotors", "fsm", pcf::IndiProperty::Text );
    add( "picomotors", "picoscix_pos", pcf::IndiProperty::Number );
    add( "camwfs-fit", "fsm", pcf::IndiProperty::Text );
    add( "camwfs-fit", "quadrant1", pcf::IndiProperty::Number );
    add( "camwfs-fit", "quadrant2", pcf::IndiProperty::Number );
    add( "camwfs-fit", "quadrant3", pcf::IndiProperty::Number );
    add( "camwfs-fit", "quadrant4", pcf::IndiProperty::Number );
    add( "camwfs-fit", "threshold", pcf::IndiProperty::Number );
    add( "camwfs-avg", "fsm", pcf::IndiProperty::Text );
    add( "camwfs-avg", "nAverage", pcf::IndiProperty::Number );
    add( "ttmpupil", "fsm", pcf::IndiProperty::Text );
    add( "ttmpupil", "pos_1", pcf::IndiProperty::Number );
    add( "ttmpupil", "pos_2", pcf::IndiProperty::Number );
    add( "ttmperi", "fsm", pcf::IndiProperty::Text );
    add( "ttmperi", "axis1_voltage", pcf::IndiProperty::Number );
    add( "ttmperi", "axis2_voltage", pcf::IndiProperty::Number );
    add( "dmtweeter", "fsm", pcf::IndiProperty::Text );
    add( "dmtweeter", "test_set", pcf::IndiProperty::Switch );
    add( "dmncpc", "fsm", pcf::IndiProperty::Text );
    add( "dmncpc", "test_set", pcf::IndiProperty::Switch );
    add( "camwfs-align", "fsm", pcf::IndiProperty::Text );
    add( "camwfs-align", "loop_state", pcf::IndiProperty::Switch );
    add( "twAlign-camwfs-ctrl", "fsm", pcf::IndiProperty::Text );
    add( "twAlign-camwfs-ctrl", "loop_state", pcf::IndiProperty::Switch );
    add( "twAlign-camwfs-wfs", "fsm", pcf::IndiProperty::Text );
    add( "twAlign-camwfs-wfs", "continuous", pcf::IndiProperty::Switch );
    add( "stagecamlensx", "fsm", pcf::IndiProperty::Text );
    add( "stagecamlensy", "fsm", pcf::IndiProperty::Text );
    add( "stagecamlensx", "position", pcf::IndiProperty::Number );
    add( "stagecamlensy", "position", pcf::IndiProperty::Number );
    add( "modwfs", "offset", pcf::IndiProperty::Number );
    add( "wooferModes", "target_amps", pcf::IndiProperty::Number );
    add( "tcsi", "pyrNudge", pcf::IndiProperty::Number );
    add( "ttmpupil", "initDM", pcf::IndiProperty::Switch );
    add( "ttmpupil", "releaseDM", pcf::IndiProperty::Switch );
    add( "ttmperi", "set", pcf::IndiProperty::Switch );
    forEachChildSubscriber(
        [&add]( const childSubscription &sub )
        {
            const std::string element = sub.element;
            add( sub.device,
                 sub.property,
                 element == "state"    ? pcf::IndiProperty::Text
                 : element == "toggle" ? pcf::IndiProperty::Switch
                                       : pcf::IndiProperty::Number );
        } );
}

bool pupilGuide::propertyAvailable( const std::string                  &device,
                                    const std::string                  &property,
                                    std::initializer_list<const char *> elements,
                                    bool                                values ) const
{
    if( !m_connected )
        return false;
    const auto found = m_properties.find( { device, property } );
    if( found == m_properties.end() )
        return false;
    const auto &observed = values ? found->second.valid : found->second.elements;
    for( const auto *element : elements )
    {
        if( observed.count( element ) == 0 )
            return false;
    }
    return true;
}

bool pupilGuide::commandEnabled( QWidget *control )
{
    updateGUI();
    return m_connected && control->isEnabled();
}

void pupilGuide::resetConnectionState()
{
    for( auto &property : m_properties )
    {
        property.second.elements.clear();
        property.second.valid.clear();
    }
    m_reportedLabMode = true;
    forEachChildSubscriber( [this]( const childSubscription &sub ) { resetChild( sub.subscriber ); } );
    m_labMode                      = true;
    m_appState                     = "";
    m_modFsmState                  = "";
    m_modState                     = 0;
    m_tcsiState                    = "";
    m_dmWooferState                = "";
    m_wooferModesState             = "";
    m_picoState                    = "UNKNOWN";
    m_camwfsfitState               = "";
    m_camwfsavgState               = "";
    m_dmtweeterState               = "";
    m_dmncpcState                  = "";
    m_pupFsmState                  = "";
    m_ttmPeriFsmState              = "";
    m_camlensxFsmState             = "";
    m_camlensyFsmState             = "";
    m_camwfs_align_fsmState        = "";
    m_twAlign_camwfs_ctrl_fsmState = "";
    m_twAlign_camwfs_wfs_fsmState  = "";
}

void pupilGuide::subscribe()
{
    if( m_parent == nullptr )
    {
        return;
    }

    forEachChildSubscriber( [this]( const childSubscription &sub ) { m_parent->addSubscriber( sub.subscriber ); } );

    for( const auto &property : m_properties )
    {
        m_parent->addSubscriberProperty( this, property.first.first, property.first.second );
    }

    return;
}

void pupilGuide::onConnect()
{
    m_connected = true;
    setEnabled( true );
    ui.label_modulation->setEnabled( true );
    ui.labelPupilFitting->setEnabled( true );

    ui.label_tipAlignment->setEnabled( true );
    ui.button_ttmtel->setEnabled( true );

    ui.picoscix_label->setEnabled( true );
    ui.picoscix_l->setEnabled( true );
    ui.picoscix_scale->setEnabled( true );
    ui.picoscix_r->setEnabled( true );
    ui.picoscix_combo_label->setEnabled( true );
    ui.picoscix_combo->setEnabled( true );
    ui.picoscix_go->setEnabled( true );

    ui.button_camera->setEnabled( true );

    ui.label_tweeter->setEnabled( true );

    ui.labelPupilSteering->setEnabled( true );

    ui.label_ncpc->setEnabled( false );

    ui.labelTTMPeri->setEnabled( true );

    ui.label_pupilPositions->setEnabled( true );

    /* Camera Lens */
    ui.label_camlens->setEnabled( true );
    ui.label_camlensX_fsm->setEnabled( true );
    ui.label_camlensY_fsm->setEnabled( true );

    camwfs_align_setEnabled( true, true );
    twAlign_camwfs_ctrl_setEnabled( true, true );
    twAlign_camwfs_wfs_setEnabled( true, true );
    alignment_buttons_setEnabled( true, true );

    setWindowTitle( "Alignment" );
    updateGUI();
}

void pupilGuide::onDisconnect()
{
    m_connected = false;
    setEnabled( false );
    resetConnectionState();

    ui.label_modulation->setEnabled( false );

    ui.label_tipAlignment->setEnabled( false );
    ui.button_ttmtel->setEnabled( false );

    ui.picoscix_label->setEnabled( false );
    ui.picoscix_l->setEnabled( false );
    ui.picoscix_scale->setEnabled( false );
    ui.picoscix_r->setEnabled( false );
    ui.picoscix_combo_label->setEnabled( false );
    ui.picoscix_combo->setEnabled( false );
    ui.picoscix_go->setEnabled( false );

    ui.button_camera->setEnabled( false );

    ui.label_tweeter->setEnabled( false );

    ui.labelPupilSteering->setEnabled( false );

    ui.label_ncpc->setEnabled( false );

    ui.labelTTMPeri->setEnabled( false );

    ui.labelPupilFitting->setEnabled( false );

    ui.label_pupilPositions->setEnabled( false );

    /* Camera Lens */
    ui.label_camlens->setEnabled( false );
    ui.label_camlensX_fsm->setEnabled( false );
    ui.label_camlensY_fsm->setEnabled( false );

    camlensSetEnabled( false );

    camwfs_align_setEnabled( false, true );
    twAlign_camwfs_ctrl_setEnabled( false, true );
    twAlign_camwfs_wfs_setEnabled( false, true );
    alignment_buttons_setEnabled( false, true );

    setWindowTitle( "Alignment (disconnected)" );
}

void pupilGuide::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    receiveProperty( ipRecv, true );
}

void pupilGuide::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    const std::string dev        = ipRecv.getDevice();
    const std::string prop       = ipRecv.getName();
    const bool        deviceLost = prop.empty() || prop == "fsm";
    if( !deviceLost && m_properties.count( { dev, prop } ) == 0 )
        return;
    for( auto &property : m_properties )
    {
        if( property.first.first == dev && ( deviceLost || property.first.second == prop ) )
        {
            property.second.elements.clear();
            property.second.valid.clear();
        }
    }

    if( deviceLost )
    {
        forEachDeviceState(
            [&dev]( const char *device, std::string &state )
            {
                if( dev == device )
                    state.clear();
            } );
    }
    if( dev == "modwfs" && ( deviceLost || prop == "modState" ) )
        m_modState = 0;
    if( dev == "tcsi" && ( deviceLost || prop == "labMode" ) )
        m_labMode = true;
    if( dev == "dmtweeter" && ( deviceLost || prop == "test_set" ) )
        m_dmtweeterTestSet = false;
    if( dev == "dmncpc" && ( deviceLost || prop == "test_set" ) )
        m_dmncpcTestSet = false;

    forEachChildSubscriber(
        [this, &dev, &prop, deviceLost]( const childSubscription &sub )
        {
            if( dev == sub.device && ( deviceLost || prop == sub.property ) )
                resetChild( sub.subscriber );
        } );
    updateGUI();
}

void pupilGuide::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    receiveProperty( ipRecv, false );
}

void pupilGuide::receiveProperty( const pcf::IndiProperty &property, bool definition )
{
    if( !m_connected )
        return;
    const auto found = m_properties.find( { property.getDevice(), property.getName() } );
    if( found == m_properties.end() )
        return;
    auto &observed = found->second;
    if( definition || property.getType() != observed.type )
    {
        observed.elements.clear();
        observed.valid.clear();
    }
    pcf::IndiProperty ipRecv( observed.type, property.getDevice(), property.getName() );
    if( property.getType() == observed.type )
    {
        for( const auto &entry : property.getElements() )
        {
            const auto &name      = entry.first;
            const auto &element   = entry.second;
            auto        validated = element;
            observed.elements.insert( name );
            bool valid = false;
            if( observed.type == pcf::IndiProperty::Number )
            {
                double             value = 0;
                std::istringstream input( element.get() );
                if( input >> value )
                {
                    input >> std::ws;
                    valid            = input.eof() && std::isfinite( value );
                    const auto &prop = property.getName();
                    if( prop == "modState" || prop == "picoscix_pos" || prop == "nPokeImages" ||
                        prop == "nPokeAverage" )
                        valid = valid && std::trunc( value ) == value && value >= std::numeric_limits<int>::min() &&
                                value <= std::numeric_limits<int>::max();
                    if( prop == "nAverage" )
                        valid = valid && std::trunc( value ) == value && value >= 0 &&
                                value <= std::numeric_limits<unsigned>::max();
                    if( property.getDevice() == "wooferModes" && prop == "current_amps" && name == "0002" )
                        valid = valid && std::fabs( value ) <= std::numeric_limits<double>::max() / 1e5;
                    if( prop == "position" )
                        valid = valid && std::fabs( value ) <= std::numeric_limits<float>::max();
                    // Normalize integer values before the legacy cache reads them with stream extraction.
                    if( valid && ( prop == "modState" || prop == "picoscix_pos" || prop == "nPokeImages" ||
                                   prop == "nPokeAverage" ) )
                        validated.set( static_cast<int>( value ) );
                    if( valid && prop == "nAverage" )
                        validated.set( static_cast<unsigned>( value ) );
                }
            }
            else if( observed.type == pcf::IndiProperty::Switch )
                valid = element.getSwitchState() == pcf::IndiElement::On ||
                        element.getSwitchState() == pcf::IndiElement::Off;
            else if( observed.type == pcf::IndiProperty::Text )
                valid = !element.get().empty();

            if( valid )
            {
                observed.valid.insert( name );
                ipRecv.add( validated );
            }
            else
                observed.valid.erase( name );
        }
    }
    const bool tcsWasConnected = m_tcsiState == "CONNECTED";
    if( property.getName() == "fsm" && observed.valid.count( "state" ) == 0 )
    {
        forEachDeviceState(
            [&property]( const char *device, std::string &state )
            {
                if( property.getDevice() == device )
                    state.clear();
            } );
    }
    forEachChildSubscriber(
        [this, &property]( const childSubscription &sub )
        {
            if( property.getDevice() == sub.device && property.getName() == sub.property &&
                !propertyAvailable( sub.device, sub.property, { sub.element } ) )
                resetChild( sub.subscriber );
        } );
    std::string dev = ipRecv.getDevice();

    if( dev == "modwfs" )
    {
        if( ipRecv.getName() == "modState" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_modState = ipRecv["current"].get<int>();
            }
        }
        else if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_modFsmState = ipRecv["state"].get<std::string>();
            }
        }
    }
    else if( dev == "camwfs" )
    {
        if( ipRecv.getName() == "fps" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_camwfsFreq = ipRecv["current"].get<double>();
            }
        }
    }
    else if( dev == "tcsi" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_tcsiState = ipRecv["state"].get<std::string>();
            }
        }
        else if( ipRecv.getName() == "labMode" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
                {
                    m_reportedLabMode = true;
                }
                else
                {
                    m_reportedLabMode = false;
                }
            }
        }
    }
    else if( dev == "dmwoofer" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_dmWooferState = ipRecv["state"].get<std::string>();
            }
        }
    }
    else if( dev == "wooferModes" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_wooferModesState = ipRecv["state"].get<std::string>();
            }
        }
        else if( ipRecv.getName() == "current_amps" )
        {
            if( ipRecv.find( "0000" ) )
            {
                m_tip = ipRecv["0000"].get<double>();
            }
            if( ipRecv.find( "0001" ) )
            {
                m_tilt = ipRecv["0001"].get<double>();
            }
            if( ipRecv.find( "0002" ) )
            {
                // round to avoid INDI f.p. precision probs.
                m_focus = std::round( ipRecv["0002"].get<double>() * 1e5 ) / 1e5;
            }
        }
    }
    else if( dev == "picomotors" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_picoState = ipRecv["state"].get<std::string>();
            }
        }
        else if( ipRecv.getName() == "picoscix_pos" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_picoscixPos = ipRecv["current"].get<int>();
            }
        }
    }
    else if( dev == "camwfs-avg" )
    {
        if( ipRecv.getName() == "nAverage" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_nAverage_current = ipRecv["current"].get<unsigned>();
            }
        }
        else if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_camwfsavgState = ipRecv["state"].get<std::string>();
            }
        }
    }
    else if( dev == "camwfs-fit" )
    {

        if( ipRecv.getName() == "quadrant1" )
        {
            if( ipRecv.find( "med" ) )
            {
                m_med1 = ipRecv["med"].get<double>();
            }

            if( ipRecv.find( "x" ) )
            {
                m_x1 = ipRecv["x"].get<double>();
            }

            if( ipRecv.find( "y" ) )
            {
                m_y1 = ipRecv["y"].get<double>();
            }

            if( ipRecv.find( "D" ) )
            {
                m_D1 = ipRecv["D"].get<double>();
            }

            if( ipRecv.find( "set-x" ) )
            {
                m_setx1 = ipRecv["set-x"].get<double>();
            }

            if( ipRecv.find( "set-y" ) )
            {
                m_sety1 = ipRecv["set-y"].get<double>();
            }

            if( ipRecv.find( "set-D" ) )
            {
                m_setD1 = ipRecv["set-D"].get<double>();
            }
        }
        else if( ipRecv.getName() == "quadrant2" )
        {
            if( ipRecv.find( "med" ) )
            {
                m_med2 = ipRecv["med"].get<double>();
            }

            if( ipRecv.find( "x" ) )
            {
                m_x2 = ipRecv["x"].get<double>();
            }

            if( ipRecv.find( "y" ) )
            {
                m_y2 = ipRecv["y"].get<double>();
            }

            if( ipRecv.find( "D" ) )
            {
                m_D2 = ipRecv["D"].get<double>();
            }

            if( ipRecv.find( "set-x" ) )
            {
                m_setx2 = ipRecv["set-x"].get<double>();
            }

            if( ipRecv.find( "set-y" ) )
            {
                m_sety2 = ipRecv["set-y"].get<double>();
            }

            if( ipRecv.find( "set-D" ) )
            {
                m_setD2 = ipRecv["set-D"].get<double>();
            }
        }
        else if( ipRecv.getName() == "quadrant3" )
        {
            if( ipRecv.find( "med" ) )
            {
                m_med3 = ipRecv["med"].get<double>();
            }

            if( ipRecv.find( "x" ) )
            {
                m_x3 = ipRecv["x"].get<double>();
            }

            if( ipRecv.find( "y" ) )
            {
                m_y3 = ipRecv["y"].get<double>();
            }

            if( ipRecv.find( "D" ) )
            {
                m_D3 = ipRecv["D"].get<double>();
            }

            if( ipRecv.find( "set-x" ) )
            {
                m_setx3 = ipRecv["set-x"].get<double>();
            }

            if( ipRecv.find( "set-y" ) )
            {
                m_sety3 = ipRecv["set-y"].get<double>();
            }

            if( ipRecv.find( "set-D" ) )
            {
                m_setD3 = ipRecv["set-D"].get<double>();
            }
        }
        else if( ipRecv.getName() == "quadrant4" )
        {
            if( ipRecv.find( "med" ) )
            {
                m_med4 = ipRecv["med"].get<double>();
            }

            if( ipRecv.find( "x" ) )
            {
                m_x4 = ipRecv["x"].get<double>();
            }

            if( ipRecv.find( "y" ) )
            {
                m_y4 = ipRecv["y"].get<double>();
            }

            if( ipRecv.find( "D" ) )
            {
                m_D4 = ipRecv["D"].get<double>();
            }

            if( ipRecv.find( "set-x" ) )
            {
                m_setx4 = ipRecv["set-x"].get<double>();
            }

            if( ipRecv.find( "set-y" ) )
            {
                m_sety4 = ipRecv["set-y"].get<double>();
            }

            if( ipRecv.find( "set-D" ) )
            {
                m_setD4 = ipRecv["set-D"].get<double>();
            }
        }
        else if( ipRecv.getName() == "threshold" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_threshold_current = ipRecv["current"].get<double>();
            }
        }
        else if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_camwfsfitState = ipRecv["state"].get<std::string>();
            }
        }
    }
    else if( dev == "ttmpupil" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_pupFsmState = ipRecv["state"].get<std::string>();
            }
        }
        else if( ipRecv.getName() == "pos_1" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_pupCh1 = ipRecv["current"].get<double>();
            }
        }
        else if( ipRecv.getName() == "pos_2" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_pupCh2 = ipRecv["current"].get<double>();
            }
        }
    }
    else if( dev == "ttmperi" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_ttmPeriFsmState = ipRecv["state"].get<std::string>();
            }
        }
        else if( ipRecv.getName() == "axis1_voltage" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_ttmPeriCh1 = ipRecv["current"].get<double>();
            }
        }
        else if( ipRecv.getName() == "axis2_voltage" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_ttmPeriCh2 = ipRecv["current"].get<double>();
            }
        }
    }
    else if( dev == "stagecamlensx" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_camlensxFsmState = ipRecv["state"].get<std::string>();
            }
        }
        else if( ipRecv.getName() == "position" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_camlensx_pos = ipRecv["current"].get<float>();
            }
        }
    }
    else if( dev == "stagecamlensy" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_camlensyFsmState = ipRecv["state"].get<std::string>();
            }
        }
        else if( ipRecv.getName() == "position" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_camlensy_pos = ipRecv["current"].get<float>();
            }
        }
    }
    else if( dev == "dmtweeter" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_dmtweeterState = ipRecv["state"].get<std::string>();
            }
        }
        else if( ipRecv.getName() == "test_set" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                if( ipRecv["toggle"] == pcf::IndiElement::On )
                    m_dmtweeterTestSet = true;
                else
                    m_dmtweeterTestSet = false;
            }
        }
    }
    else if( dev == "dmncpc" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_dmncpcState = ipRecv["state"].get<std::string>();
            }
        }
        else if( ipRecv.getName() == "test_set" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                if( ipRecv["toggle"] == pcf::IndiElement::On )
                    m_dmncpcTestSet = true;
                else
                    m_dmncpcTestSet = false;
            }
        }
    }
    else if( dev == "camwfs-align" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_camwfs_align_fsmState = ipRecv["state"].get<std::string>();
            }
        }
    }
    else if( dev == "twAlign-camwfs-ctrl" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_twAlign_camwfs_ctrl_fsmState = ipRecv["state"].get<std::string>();
            }
        }
    }
    else if( dev == "twAlign-camwfs-wfs" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_twAlign_camwfs_wfs_fsmState = ipRecv["state"].get<std::string>();
            }
        }
    }
    if( dev == "tcsi" && ipRecv.getName() == "fsm" && tcsWasConnected && m_tcsiState != "CONNECTED" )
    {
        auto &mode = m_properties.at( { "tcsi", "labMode" } );
        mode.elements.clear();
        mode.valid.clear();
        m_reportedLabMode = true;
    }
    updateGUI();
    return;
}

void pupilGuide::modGUISetEnable( bool enableModGUI, bool enableModArrows )
{
    if( enableModGUI )
    {
        ui.label_modulation->setEnabled( true );
        ui.modwfs_fsm->setEnabled( true );
        ui.modwfs_fsm->setEnabled( true );
        if( m_modState == 3 || m_modState == 4 )
        {
            ui.label_modFreq->setEnabled( true );
            ui.modFreq_current->setEnabled( true );
            ui.label_modRad->setEnabled( true );
            ui.modRad_current->setEnabled( true );

            ui.modCh1->setEnabled( true );
            ui.modCh2->setEnabled( true );
        }
        else
        {
            ui.label_modFreq->setEnabled( false );
            ui.modFreq_current->setEnabled( false );
            ui.label_modRad->setEnabled( false );
            ui.modRad_current->setEnabled( false );

            ui.modCh1->setEnabled( false );
            ui.modCh2->setEnabled( false );
        }

        if( enableModArrows )
        {
            ui.button_tip_ul->setEnabled( true );
            ui.button_tip_u->setEnabled( true );
            ui.button_tip_ur->setEnabled( true );
            ui.button_tip_l->setEnabled( true );
            ui.button_tip_scale->setEnabled( true );
            ui.button_tip_r->setEnabled( true );
            ui.button_tip_dl->setEnabled( true );
            ui.button_tip_d->setEnabled( true );
            ui.button_tip_dr->setEnabled( true );

            if( m_tipmovewhat == MOVE_TEL || m_tipmovewhat == MOVE_WOOF )
            {
                ui.button_focus_p->setEnabled( true );
                ui.button_focus_scale->setEnabled( true );
                ui.button_focus_m->setEnabled( true );
            }
            else
            {
                ui.button_focus_p->setEnabled( false );
                ui.button_focus_scale->setEnabled( false );
                ui.button_focus_m->setEnabled( false );
            }
        }
        else
        {
            ui.button_tip_ul->setEnabled( false );
            ui.button_tip_u->setEnabled( false );
            ui.button_tip_ur->setEnabled( false );
            ui.button_tip_l->setEnabled( false );
            ui.button_tip_scale->setEnabled( false );
            ui.button_tip_r->setEnabled( false );
            ui.button_tip_dl->setEnabled( false );
            ui.button_tip_d->setEnabled( false );
            ui.button_tip_dr->setEnabled( false );

            ui.button_focus_p->setEnabled( false );
            ui.button_focus_scale->setEnabled( false );
            ui.button_focus_m->setEnabled( false );
        }
    }
    else
    {
        if( m_modFsmState != "POWEROFF" && m_modFsmState != "CONFIGURING" )
        {
            ui.label_modulation->setEnabled( false );
        }
        else
        {
            ui.label_modulation->setEnabled( true );
        }
        ui.modwfs_fsm->setEnabled( false );
        ui.label_modFreq->setEnabled( false );
        ui.modFreq_current->setEnabled( false );
        ui.label_modRad->setEnabled( false );
        ui.modRad_current->setEnabled( false );
        ui.buttonMod_rest->setEnabled( false );
        ui.buttonMod_set->setEnabled( false );
        ui.buttonMod_mod->setEnabled( false );
        ui.modCh1->setEnabled( false );
        ui.modCh2->setEnabled( false );

        if( !enableModArrows )
        {
            ui.button_tip_ul->setEnabled( false );
            ui.button_tip_u->setEnabled( false );
            ui.button_tip_ur->setEnabled( false );
            ui.button_tip_l->setEnabled( false );
            ui.button_tip_scale->setEnabled( false );
            ui.button_tip_r->setEnabled( false );
            ui.button_tip_dl->setEnabled( false );
            ui.button_tip_d->setEnabled( false );
            ui.button_tip_dr->setEnabled( false );

            ui.button_focus_p->setEnabled( false );
            ui.button_focus_scale->setEnabled( false );
            ui.button_focus_m->setEnabled( false );
        }
        else
        {
            ui.button_tip_ul->setEnabled( true );
            ui.button_tip_u->setEnabled( true );
            ui.button_tip_ur->setEnabled( true );
            ui.button_tip_l->setEnabled( true );
            ui.button_tip_scale->setEnabled( true );
            ui.button_tip_r->setEnabled( true );
            ui.button_tip_dl->setEnabled( true );
            ui.button_tip_d->setEnabled( true );
            ui.button_tip_dr->setEnabled( true );

            if( m_tipmovewhat != MOVE_TTM )
            {
                ui.button_focus_p->setEnabled( true );
                ui.button_focus_scale->setEnabled( true );
                ui.button_focus_m->setEnabled( true );
            }
        }
    }
}

void pupilGuide::camwfsfitSetEnabled( bool enabled )
{
    ui.labelMedianFluxes->setEnabled( enabled );
    ui.med1->setEnabled( enabled );
    ui.med2->setEnabled( enabled );
    ui.med3->setEnabled( enabled );
    ui.med4->setEnabled( enabled );
    ui.setDelta->setEnabled( enabled );
    ui.fitThreshold->setEnabled( enabled );

    if( enabled == false )
    {
        ui.med1->setText( "" );
        ui.med2->setText( "" );
        ui.med3->setText( "" );
        ui.med4->setText( "" );
    }

    ui.coordLL_D->setEnabled( enabled );
    ui.coordLR_D->setEnabled( enabled );
    ui.coordUL_D->setEnabled( enabled );
    ui.coordUR_D->setEnabled( enabled );
    ui.coordLL_x->setEnabled( enabled );
    ui.coordLR_x->setEnabled( enabled );
    ui.coordUL_x->setEnabled( enabled );
    ui.coordUR_x->setEnabled( enabled );
    ui.coordLL_y->setEnabled( enabled );
    ui.coordLR_y->setEnabled( enabled );
    ui.coordUL_y->setEnabled( enabled );
    ui.coordUR_y->setEnabled( enabled );
    ui.coordAvg_D->setEnabled( enabled );
    ui.coordAvg_x->setEnabled( enabled );
    ui.coordAvg_y->setEnabled( enabled );
    ui.setDelta_pup->setEnabled( enabled );
    ui.labelx->setEnabled( enabled );
    ui.labely->setEnabled( enabled );
    ui.labelD->setEnabled( enabled );
    ui.labelUR->setEnabled( enabled );
    ui.labelUL->setEnabled( enabled );
    ui.labelLR->setEnabled( enabled );
    ui.labelLL->setEnabled( enabled );
    ui.labelAvg->setEnabled( enabled );
}

void pupilGuide::camlensSetEnabled( bool enabled, int whichcl )
{
    if( whichcl == CAMLENS_BOTH )
    {
        ui.button_camlens_scale->setEnabled( enabled );
    }
    else
    {
        ui.button_camlens_scale->setEnabled( true );
    }

    if( whichcl == CAMLENS_X || whichcl == CAMLENS_BOTH )
    {
        ui.camlensX_fsm->setEnabled( enabled );
        ui.camlensX_pos->setEnabled( enabled );
        ui.button_camlens_l->setEnabled( enabled );
        ui.button_camlens_r->setEnabled( enabled );
    }

    if( whichcl == CAMLENS_Y || whichcl == CAMLENS_BOTH )
    {
        ui.camlensY_fsm->setEnabled( enabled );
        ui.camlensY_pos->setEnabled( enabled );
        ui.button_camlens_u->setEnabled( enabled );
        ui.button_camlens_d->setEnabled( enabled );
    }
}

void pupilGuide::camwfs_align_setEnabled( bool enabled, bool all )
{
    if( all )
    {
        ui.label_pupTrackLoop->setEnabled( enabled );
    }
    ui.pupTrackLoop_deltaX->setEnabled( enabled );
    ui.pupTrackLoop_deltaY->setEnabled( enabled );
    ui.pupTrackLoop_slider->setEnabled( enabled );
    ui.pupTrackLoop_gain->setEnabled( enabled );
}

void pupilGuide::twAlign_camwfs_ctrl_setEnabled( bool enabled, bool all )
{
    if( all )
    {
        ui.label_actAlignLoop->setEnabled( enabled );
    }
    ui.actAlignLoop_deltaX->setEnabled( enabled );
    ui.actAlignLoop_deltaY->setEnabled( enabled );
    ui.actAlignLoop_slider->setEnabled( enabled );
    ui.actAlignLoop_gain->setEnabled( enabled );
}

void pupilGuide::twAlign_camwfs_wfs_setEnabled( bool enabled, bool all )
{
    if( all )
    {
        ui.label_actAlignSensor->setEnabled( enabled );
    }
    ui.actAlignSensor_slider->setEnabled( enabled );
    ui.actAlignSensor_nAverage->setEnabled( enabled );
    ui.actAlignSensor_nImages->setEnabled( enabled );
    ui.actAlignSensor_pokeAmp->setEnabled( enabled );
}

void pupilGuide::alignment_buttons_setEnabled( bool enabled, bool all )
{
    if( all )
    {
        ui.label_alignment->setEnabled( enabled );
    }
    ui.button_startAlignment->setEnabled( enabled );
    ui.button_stopAlignment->setEnabled( enabled );
}

void pupilGuide::updateAvailability()
{
    const auto limit = []( QWidget *control, bool available )
    { control->setEnabled( control->isEnabled() && available ); };
    m_labMode =
        m_tcsiState != "CONNECTED" || !propertyAvailable( "tcsi", "labMode", { "toggle" } ) || m_reportedLabMode;

    const bool woofer = m_dmWooferState == "OPERATING" && m_wooferModesState == "READY";
    bool       tip    = false;
    bool       focus  = false;
    if( m_tipmovewhat == MOVE_WOOF )
    {
        tip = woofer && propertyAvailable( "wooferModes", "current_amps", { "0000", "0001" } ) &&
              propertyAvailable( "wooferModes", "target_amps", { "0000", "0001" }, false );
        focus = woofer && propertyAvailable( "wooferModes", "current_amps", { "0002" } ) &&
                propertyAvailable( "wooferModes", "target_amps", { "0002" }, false );
    }
    else if( m_tipmovewhat == MOVE_TTM )
    {
        tip = ( m_modFsmState == "READY" || m_modFsmState == "OPERATING" ) &&
              propertyAvailable( "modwfs", "offset", { "x", "y" }, false );
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        tip   = !m_labMode && propertyAvailable( "tcsi", "pyrNudge", { "x", "y" }, false );
        focus = !m_labMode && propertyAvailable( "tcsi", "pyrNudge", { "z" }, false );
    }
    for( auto *button : { ui.button_tip_ul,
                          ui.button_tip_u,
                          ui.button_tip_ur,
                          ui.button_tip_l,
                          ui.button_tip_scale,
                          ui.button_tip_r,
                          ui.button_tip_dl,
                          ui.button_tip_d,
                          ui.button_tip_dr } )
        button->setEnabled( tip );
    for( auto *button : { ui.button_focus_p, ui.button_focus_scale, ui.button_focus_m } )
        button->setEnabled( focus );

    const bool mod = propertyAvailable( "modwfs", "modState", { "current" } ) &&
                     propertyAvailable( "modwfs", "modState", { "target" }, false );
    for( auto *button : { ui.buttonMod_mod, ui.buttonMod_set, ui.buttonMod_rest } )
        limit( button, mod );

    const bool pico = propertyAvailable( "picomotors", "picoscix_pos", { "current" } ) &&
                      propertyAvailable( "picomotors", "picoscix_pos", { "target" }, false );
    for( QWidget *control : std::array<QWidget *, 5>{
             ui.picoscix_l, ui.picoscix_r, ui.picoscix_scale, ui.picoscix_combo, ui.picoscix_go } )
        limit( control, pico );
    limit( ui.buttonTweeterTest_set, propertyAvailable( "dmtweeter", "test_set", { "toggle" } ) );
    limit( ui.buttonNCPCTest_set, propertyAvailable( "dmncpc", "test_set", { "toggle" } ) );

    const bool pupil = propertyAvailable( "ttmpupil", "pos_1", { "current" } ) &&
                       propertyAvailable( "ttmpupil", "pos_2", { "current" } ) &&
                       propertyAvailable( "ttmpupil", "pos_1", { "target" }, false ) &&
                       propertyAvailable( "ttmpupil", "pos_2", { "target" }, false );
    for( auto *button :
         { ui.button_pup_ul, ui.button_pup_ur, ui.button_pup_dl, ui.button_pup_dr, ui.button_pup_scale } )
        limit( button, pupil );
    limit( ui.buttonPup_set, propertyAvailable( "ttmpupil", "initDM", { "request" } ) );
    limit( ui.buttonPup_rest, propertyAvailable( "ttmpupil", "releaseDM", { "request" } ) );
    limit( ui.button_ttmPeri_set, propertyAvailable( "ttmperi", "set", { "toggle" } ) );
    limit( ui.button_ttmPeri_rest, propertyAvailable( "ttmperi", "set", { "toggle" } ) );
    const bool periX = propertyAvailable( "ttmperi", "axis1_voltage", { "current" } ) &&
                       propertyAvailable( "ttmperi", "axis1_voltage", { "target" }, false );
    const bool periY = propertyAvailable( "ttmperi", "axis2_voltage", { "current" } ) &&
                       propertyAvailable( "ttmperi", "axis2_voltage", { "target" }, false );
    limit( ui.button_ttmPeri_l, periX );
    limit( ui.button_ttmPeri_r, periX );
    limit( ui.button_ttmPeri_u, periY );
    limit( ui.button_ttmPeri_d, periY );
    limit( ui.button_ttmPeri_scale, periX || periY );
    const bool lensX = m_camlensxFsmState == "READY" &&
                       propertyAvailable( "stagecamlensx", "position", { "current" } ) &&
                       propertyAvailable( "stagecamlensx", "position", { "target" }, false );
    const bool lensY = m_camlensyFsmState == "READY" &&
                       propertyAvailable( "stagecamlensy", "position", { "current" } ) &&
                       propertyAvailable( "stagecamlensy", "position", { "target" }, false );
    limit( ui.button_camlens_l, lensX );
    limit( ui.button_camlens_r, lensX );
    limit( ui.button_camlens_u, lensY );
    limit( ui.button_camlens_d, lensY );
    ui.button_camlens_scale->setEnabled( lensX || lensY );

    forEachChildSubscriber(
        [this, &limit]( const childSubscription &sub )
        {
            auto *control   = dynamic_cast<QWidget *>( sub.subscriber );
            bool  available = propertyAvailable( sub.device, sub.property, { sub.element } );
            auto *entry     = dynamic_cast<statusEntry *>( sub.subscriber );
            if( entry && !entry->readOnly() )
                available =
                    available && propertyAvailable( sub.device,
                                                    sub.property,
                                                    { std::string( sub.element ) == "value" ? "value" : "target" },
                                                    false );
            limit( control, available );
            if( entry )
                entry->updateGUI();
            if( auto *slider = dynamic_cast<toggleSlider *>( sub.subscriber ) )
                slider->guiUpdate();
        } );

    const bool sensor     = propertyAvailable( "twAlign-camwfs-wfs", "continuous", { "toggle" } );
    const bool controller = propertyAvailable( "twAlign-camwfs-ctrl", "loop_state", { "toggle" } );
    const bool tracking   = propertyAvailable( "camwfs-align", "loop_state", { "toggle" } );
    const auto ready      = []( const std::string &state ) { return state == "READY" || state == "OPERATING"; };
    ui.button_startAlignment->setEnabled( sensor && controller && tracking && ready( m_twAlign_camwfs_wfs_fsmState ) &&
                                          ready( m_twAlign_camwfs_ctrl_fsmState ) && ready( m_camwfs_align_fsmState ) );
    ui.button_stopAlignment->setEnabled( sensor || controller || ( m_labMode && tracking ) );

    // Derived fit values are displays, but must not present missing observations as zero or stale data.
    const std::array<QLineEdit *, 4>                medians{ ui.med1, ui.med2, ui.med3, ui.med4 };
    const std::array<std::array<QLineEdit *, 4>, 3> coordinates{
        { { ui.coordLL_x, ui.coordLR_x, ui.coordUL_x, ui.coordUR_x },
          { ui.coordLL_y, ui.coordLR_y, ui.coordUL_y, ui.coordUR_y },
          { ui.coordLL_D, ui.coordLR_D, ui.coordUL_D, ui.coordUR_D } } };
    const std::array<QLineEdit *, 3>  averages{ ui.coordAvg_x, ui.coordAvg_y, ui.coordAvg_D };
    const std::array<const char *, 3> axes{ "x", "y", "D" };
    const auto                        display = [&limit]( QLineEdit *field, bool available )
    {
        limit( field, available );
        if( !field->isEnabled() )
            field->clear();
    };
    bool allMedians = true;
    for( int q = 0; q < 4; ++q )
        allMedians = allMedians && propertyAvailable( "camwfs-fit", "quadrant" + std::to_string( q + 1 ), { "med" } );
    for( int q = 0; q < 4; ++q )
        display( medians[q],
                 ui.setDelta->isChecked()
                     ? allMedians
                     : propertyAvailable( "camwfs-fit", "quadrant" + std::to_string( q + 1 ), { "med" } ) );
    for( int axis = 0; axis < 3; ++axis )
    {
        bool       all       = true;
        const auto reference = std::string( "set-" ) + axes[axis];
        for( int q = 0; q < 4; ++q )
        {
            const auto property = "quadrant" + std::to_string( q + 1 );
            bool       available =
                propertyAvailable( "camwfs-fit", property, { axes[axis] } ) &&
                ( !ui.setDelta_pup->isChecked() || propertyAvailable( "camwfs-fit", property, { reference.c_str() } ) );
            display( coordinates[axis][q], available );
            all = all && available;
        }
        display( averages[axis], all );
    }
}

void pupilGuide::updateGUI()
{
    if( !m_connected )
        return;

    //--------- Modulation

    bool enableModGUI    = true;
    bool enableModArrows = true;

    char str[16];
    if( m_modFsmState == "NOTHOMED" )
    {
        if( m_tipmovewhat == MOVE_TTM )
        {
            enableModArrows = false;
        }
    }
    else if( ( m_modFsmState != "READY" ) && ( m_modFsmState != "OPERATING" ) )
    {
        enableModGUI = false;
        if( m_tipmovewhat == MOVE_TTM )
        {
            enableModArrows = false;
        }
    }

    // If moving woofer and either woofer or wooferModes aren't ready we disable the arrows
    if( m_tipmovewhat == MOVE_WOOF && ( m_dmWooferState != "OPERATING" || m_wooferModesState != "READY" ) )
    {
        enableModArrows = false;
    }

    // If moving telescope and tcsi isn't connected we disable the arrows
    if( m_tipmovewhat == MOVE_TEL && ( m_tcsiState != "CONNECTED" ) )
    {
        enableModArrows = false;
    }

    modGUISetEnable( enableModGUI, enableModArrows );

    if( m_modState == 3 && enableModGUI )
    {
        ui.buttonMod_rest->setEnabled( true );
        ui.buttonMod_set->setEnabled( false );
        ui.buttonMod_mod->setEnabled( true );
    }
    else if( m_modState == 4 && enableModGUI )
    {
        ui.buttonMod_rest->setEnabled( true );
        ui.buttonMod_set->setEnabled( true );
        ui.buttonMod_mod->setEnabled( true );
    }
    else
    {
        if( enableModGUI )
        {
            ui.buttonMod_rest->setEnabled( true );
            ui.buttonMod_set->setEnabled( true );
            ui.buttonMod_mod->setEnabled( false );
        }
    }

    // ------picoscis
    if( m_picoState != "READY" )
    {
        ui.picoscix_pos->setEnabled( false );
        ui.picoscix_l->setEnabled( false );
        ui.picoscix_scale->setEnabled( false );
        ui.picoscix_r->setEnabled( false );
        ui.picoscix_combo->setEnabled( false );
        ui.picoscix_go->setEnabled( false );
    }
    else
    {
        ui.picoscix_pos->setEnabled( true );
        ui.picoscix_l->setEnabled( true );
        ui.picoscix_scale->setEnabled( true );
        ui.picoscix_r->setEnabled( true );
        ui.picoscix_combo->setEnabled( true );
        ui.picoscix_go->setEnabled( true );
    }

    // ------Pupil Fitting

    if( !( m_camwfsfitState == "READY" || m_camwfsfitState == "OPERATING" ) )
    {
        camwfsfitSetEnabled( false );
    }
    else
    {
        camwfsfitSetEnabled( true );

        double m1, m2, m3, m4;

        if( ui.setDelta->checkState() == Qt::Checked )
        {
            double ave = 0.25 * ( m_med1 + m_med2 + m_med3 + m_med4 );
            m1         = m_med1 - ave;
            m2         = m_med2 - ave;
            m3         = m_med3 - ave;
            m4         = m_med4 - ave;
        }
        else
        {
            m1 = m_med1;
            m2 = m_med2;
            m3 = m_med3;
            m4 = m_med4;
        }

        snprintf( str, 16, "%0.1f", m1 );
        ui.med1->setText( str );

        snprintf( str, 16, "%0.1f", m2 );
        ui.med2->setText( str );

        snprintf( str, 16, "%0.1f", m3 );
        ui.med3->setText( str );

        snprintf( str, 16, "%0.1f", m4 );
        ui.med4->setText( str );

        double x1 = m_x1;
        double y1 = m_y1;
        double D1 = m_D1;
        double x2 = m_x2;
        double y2 = m_y2;
        double D2 = m_D2;
        double x3 = m_x3;
        double y3 = m_y3;
        double D3 = m_D3;
        double x4 = m_x4;
        double y4 = m_y4;
        double D4 = m_D4;

        if( ui.setDelta_pup->checkState() == Qt::Checked )
        {
            x1 -= m_setx1;
            y1 -= m_sety1;
            D1 -= m_setD1;

            x2 -= m_setx2;
            y2 -= m_sety2;
            D2 -= m_setD2;

            x3 -= m_setx3;
            y3 -= m_sety3;
            D3 -= m_setD3;

            x4 -= m_setx4;
            y4 -= m_sety4;
            D4 -= m_setD4;
        }

        snprintf( str, 16, "%0.2f", D1 );
        ui.coordLL_D->setText( str );

        snprintf( str, 16, "%0.2f", D2 );
        ui.coordLR_D->setText( str );

        snprintf( str, 16, "%0.2f", D3 );
        ui.coordUL_D->setText( str );

        snprintf( str, 16, "%0.2f", D4 );
        ui.coordUR_D->setText( str );

        snprintf( str, 16, "%0.2f", x1 );
        ui.coordLL_x->setText( str );

        snprintf( str, 16, "%0.2f", x2 );
        ui.coordLR_x->setText( str );

        snprintf( str, 16, "%0.2f", x3 );
        ui.coordUL_x->setText( str );

        snprintf( str, 16, "%0.2f", x4 );
        ui.coordUR_x->setText( str );

        snprintf( str, 16, "%0.2f", y1 );
        ui.coordLL_y->setText( str );

        snprintf( str, 16, "%0.2f", y2 );
        ui.coordLR_y->setText( str );

        snprintf( str, 16, "%0.2f", y3 );
        ui.coordUL_y->setText( str );

        snprintf( str, 16, "%0.2f", y4 );
        ui.coordUR_y->setText( str );

        snprintf( str, 16, "%0.2f", 0.25 * ( D1 + D2 + D3 + D4 ) );
        ui.coordAvg_D->setText( str );

        snprintf( str, 16, "%0.2f", 0.25 * ( x1 + x2 + x3 + x4 ) );
        ui.coordAvg_x->setText( str );

        snprintf( str, 16, "%0.2f", 0.25 * ( y1 + y2 + y3 + y4 ) );
        ui.coordAvg_y->setText( str );
    }

    // ------ camwfs averaging
    if( m_camwfsavgState == "READY" || m_camwfsavgState == "OPERATING" )
    {
        ui.fitAvgTime->setEnabled( true );
    }
    else
    {
        ui.fitAvgTime->setEnabled( false );
    }

    // ------ dmtweeter

    if( m_dmtweeterState == "READY" || m_dmtweeterState == "OPERATING" )
    {
        ui.buttonTweeterTest_set->setEnabled( true );
        if( m_dmtweeterTestSet )
        {
            ui.buttonTweeterTest_set->setText( "zero test" );
        }
        else
        {
            ui.buttonTweeterTest_set->setText( "set test" );
        }
    }
    else
    {
        ui.buttonTweeterTest_set->setEnabled( false );
        ui.buttonTweeterTest_set->setText( "set test" );
    }

    // ------ dmncpc

    if( m_dmncpcState == "READY" || m_dmncpcState == "OPERATING" )
    {
        ui.buttonNCPCTest_set->setEnabled( true );

        if( m_dmncpcTestSet )
        {
            ui.buttonNCPCTest_set->setText( "zero test" );
        }
        else
        {
            ui.buttonNCPCTest_set->setText( "set test" );
        }
    }
    else
    {
        ui.buttonNCPCTest_set->setEnabled( false );
        ui.buttonNCPCTest_set->setText( "set test" );
    }

    // ------ Pupil Steering
    bool enablePupFSM       = true;
    bool enablePupFSMArrows = true;

    if( m_pupFsmState == "READY" )
    {
        ui.pupState->setEnabled( true );
        ui.pupCh1->setEnabled( true );
        ui.pupCh2->setEnabled( true );
        ui.buttonPup_set->setEnabled( false );
        ui.buttonPup_rest->setEnabled( true );
    }
    else if( m_pupFsmState == "NOTHOMED" )
    {
        ui.pupState->setEnabled( true );
        ui.pupCh1->setEnabled( false );
        ui.pupCh2->setEnabled( false );
        ui.buttonPup_set->setEnabled( true );
        ui.buttonPup_rest->setEnabled( false );
        enablePupFSMArrows = false;
    }
    else if( m_pupFsmState == "HOMING" )
    {
        ui.pupState->setEnabled( true );
        ui.pupCh1->setEnabled( false );
        ui.pupCh2->setEnabled( false );
        ui.buttonPup_set->setEnabled( false );
        ui.buttonPup_rest->setEnabled( true );
        enablePupFSMArrows = false;
    }
    else
    {
        enablePupFSM = false;
        if( m_pupFsmState == "" )
        {
            ui.pupState->setEnabled( false );
        }
        else
        {
            ui.pupState->setEnabled( true );
        }
    }

    if( enablePupFSM )
    {
        if( enablePupFSMArrows )
        {
            ui.button_pup_ul->setEnabled( true );
            ui.button_pup_ur->setEnabled( true );
            ui.button_pup_scale->setEnabled( true );
            ui.button_pup_dl->setEnabled( true );
            ui.button_pup_dr->setEnabled( true );
        }
        else
        {
            ui.button_pup_ul->setEnabled( false );
            ui.button_pup_ur->setEnabled( false );
            ui.button_pup_scale->setEnabled( false );
            ui.button_pup_dl->setEnabled( false );
            ui.button_pup_dr->setEnabled( false );
        }
    }
    else
    {

        ui.buttonPup_set->setEnabled( false );
        ui.buttonPup_rest->setEnabled( false );
        ui.pupCh1->setEnabled( false );
        ui.pupCh2->setEnabled( false );

        ui.button_pup_ul->setEnabled( false );
        ui.button_pup_ur->setEnabled( false );
        ui.button_pup_scale->setEnabled( false );
        ui.button_pup_dl->setEnabled( false );
        ui.button_pup_dr->setEnabled( false );
    }

    // ------ TTM Peri
    bool enableTTMPeriFSM       = true;
    bool enableTTMPeriFSMArrows = true;

    if( m_ttmPeriFsmState == "READY" )
    {
        ui.ttmPeriState->setEnabled( true );
        ui.ttmPeriCh1->setEnabled( false );
        ui.ttmPeriCh2->setEnabled( false );
        ui.button_ttmPeri_set->setEnabled( true );
        ui.button_ttmPeri_rest->setEnabled( false );

        enableTTMPeriFSMArrows = false;
    }
    else if( m_ttmPeriFsmState == "OPERATING" )
    {
        ui.ttmPeriState->setEnabled( true );
        ui.ttmPeriCh1->setEnabled( true );
        ui.ttmPeriCh2->setEnabled( true );
        ui.button_ttmPeri_set->setEnabled( false );
        ui.button_ttmPeri_rest->setEnabled( true );
        enableTTMPeriFSMArrows = true;
    }
    else
    {
        enableTTMPeriFSM = false;

        if( m_ttmPeriFsmState == "" )
        {
            ui.ttmPeriState->setEnabled( false );
        }
        else
        {
            ui.ttmPeriState->setEnabled( true );
        }

        ui.ttmPeriCh1->setEnabled( false );
        ui.ttmPeriCh2->setEnabled( false );
        ui.button_ttmPeri_set->setEnabled( false );
        ui.button_ttmPeri_rest->setEnabled( false );
    }

    if( enableTTMPeriFSM )
    {
        if( enableTTMPeriFSMArrows )
        {
            ui.button_ttmPeri_l->setEnabled( true );
            ui.button_ttmPeri_r->setEnabled( true );
            ui.button_ttmPeri_scale->setEnabled( true );
            ui.button_ttmPeri_u->setEnabled( true );
            ui.button_ttmPeri_d->setEnabled( true );
        }
        else
        {
            ui.button_ttmPeri_l->setEnabled( false );
            ui.button_ttmPeri_r->setEnabled( false );
            ui.button_ttmPeri_scale->setEnabled( false );
            ui.button_ttmPeri_u->setEnabled( false );
            ui.button_ttmPeri_d->setEnabled( false );
        }
    }
    else
    {
        ui.button_ttmPeri_l->setEnabled( false );
        ui.button_ttmPeri_r->setEnabled( false );
        ui.button_ttmPeri_scale->setEnabled( false );
        ui.button_ttmPeri_u->setEnabled( false );
        ui.button_ttmPeri_d->setEnabled( false );
    }

    // --- camera lens

    if( ( m_camlensxFsmState == "READY" || m_camlensxFsmState == "OPERATING" ) &&
        ( m_camlensyFsmState == "READY" || m_camlensyFsmState == "OPERATING" ) )
    {
        camlensSetEnabled( true );
    }
    else if( ( m_camlensxFsmState == "READY" || m_camlensxFsmState == "OPERATING" ) &&
             !( m_camlensyFsmState == "READY" || m_camlensyFsmState == "OPERATING" ) )
    {
        camlensSetEnabled( true, CAMLENS_X );
        camlensSetEnabled( false, CAMLENS_Y );
    }
    else if( !( m_camlensxFsmState == "READY" || m_camlensxFsmState == "OPERATING" ) &&
             ( m_camlensyFsmState == "READY" || m_camlensyFsmState == "OPERATING" ) )
    {
        camlensSetEnabled( false, CAMLENS_X );

        camlensSetEnabled( true, CAMLENS_Y );
    }
    else
    {
        camlensSetEnabled( false );
    }

    if( m_camwfs_align_fsmState != "READY" && m_camwfs_align_fsmState != "OPERATING" )
    {
        camwfs_align_setEnabled( false, false );
    }
    else
    {
        camwfs_align_setEnabled( true, true );
    }

    if( m_twAlign_camwfs_ctrl_fsmState != "READY" && m_twAlign_camwfs_ctrl_fsmState != "OPERATING" )
    {
        twAlign_camwfs_ctrl_setEnabled( false, false );
    }
    else
    {
        twAlign_camwfs_ctrl_setEnabled( true, true );
    }

    if( m_twAlign_camwfs_wfs_fsmState != "READY" && m_twAlign_camwfs_wfs_fsmState != "OPERATING" )
    {
        twAlign_camwfs_wfs_setEnabled( false, false );
    }
    else
    {
        twAlign_camwfs_wfs_setEnabled( true, true );
    }

    updateAvailability();
} // updateGUI()

// ------------- modttm

void pupilGuide::on_buttonMod_mod_pressed()
{
    if( !commandEnabled( ui.buttonMod_mod ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "modwfs" );
    ip.setName( "modState" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = 4;

    sendNewProperty( ip );
}

void pupilGuide::on_buttonMod_set_pressed()
{
    if( !commandEnabled( ui.buttonMod_set ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "modwfs" );
    ip.setName( "modState" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = 3;
    sendNewProperty( ip );
}

void pupilGuide::on_buttonMod_rest_pressed()
{
    if( !commandEnabled( ui.buttonMod_rest ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "modwfs" );
    ip.setName( "modState" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = 1;

    sendNewProperty( ip );
}

void pupilGuide::on_button_ttmtel_pressed()
{
    if( m_tipmovewhat == MOVE_TTM )
    {
        m_tipmovewhat = MOVE_WOOF;
        ui.button_ttmtel->setText( "move woofer" );
    }
    else if( m_tipmovewhat == MOVE_WOOF && !m_labMode )
    {
        m_tipmovewhat = MOVE_TEL;
        ui.button_ttmtel->setText( "move telescope" );
    }
    else
    {
        m_tipmovewhat = MOVE_TTM;
        ui.button_ttmtel->setText( "move ttm" );
    }
}

void pupilGuide::on_button_tip_u_pressed()
{
    if( !commandEnabled( ui.button_tip_u ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = m_stepSize;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = 0;
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, 0, m_stepSize );

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0000" ) );
        ip.add( pcf::IndiElement( "0001" ) );
        ip["0000"] = m_tip + tip;
        ip["0001"] = m_tilt + tilt;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = m_stepSize * 5.;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = 0;
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_button_tip_ul_pressed()
{
    if( !commandEnabled( ui.button_tip_ul ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = m_stepSize / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = -m_stepSize / sqrt( 2. );
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, m_stepSize / sqrt( 2. ), m_stepSize / sqrt( 2. ) );

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0000" ) );
        ip.add( pcf::IndiElement( "0001" ) );
        ip["0000"] = m_tip + tip;
        ip["0001"] = m_tilt + tilt;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = m_stepSize * 5. / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = -m_stepSize * 5. / sqrt( 2. );
    }

    sendNewProperty( ip );
}

void pupilGuide::on_button_tip_l_pressed()
{
    if( !commandEnabled( ui.button_tip_l ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = 0;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = -m_stepSize;
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, m_stepSize, 0 );

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0000" ) );
        ip.add( pcf::IndiElement( "0001" ) );
        ip["0000"] = m_tip + tip;
        ip["0001"] = m_tilt + tilt;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = 0;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = -m_stepSize * 5.;
    }

    sendNewProperty( ip );
}

void pupilGuide::on_button_tip_dl_pressed()
{
    if( !commandEnabled( ui.button_tip_dl ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = -m_stepSize / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = -m_stepSize / sqrt( 2. );
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, m_stepSize / sqrt( 2. ), -m_stepSize / sqrt( 2. ) );

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0000" ) );
        ip.add( pcf::IndiElement( "0001" ) );
        ip["0000"] = m_tip + tip;
        ip["0001"] = m_tilt + tilt;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = -m_stepSize * 5. / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = -m_stepSize * 5. / sqrt( 2. );
    }

    sendNewProperty( ip );
}

void pupilGuide::on_button_tip_d_pressed()
{
    if( !commandEnabled( ui.button_tip_d ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = -m_stepSize;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = 0;
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, 0, -m_stepSize );

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0000" ) );
        ip.add( pcf::IndiElement( "0001" ) );
        ip["0000"] = m_tip + tip;
        ip["0001"] = m_tilt + tilt;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = -m_stepSize * 5.;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = 0;
    }

    sendNewProperty( ip );
}

void pupilGuide::on_button_tip_dr_pressed()
{
    if( !commandEnabled( ui.button_tip_dr ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = -m_stepSize / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_stepSize / sqrt( 2. );
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, -m_stepSize / sqrt( 2. ), -m_stepSize / sqrt( 2. ) );

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0000" ) );
        ip.add( pcf::IndiElement( "0001" ) );
        ip["0000"] = m_tip + tip;
        ip["0001"] = m_tilt + tilt;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = -m_stepSize * 5. / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_stepSize * 5. / sqrt( 2. );
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_button_tip_r_pressed()
{
    if( !commandEnabled( ui.button_tip_r ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = 0;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_stepSize;
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, -m_stepSize, 0 );

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0000" ) );
        ip.add( pcf::IndiElement( "0001" ) );
        ip["0000"] = m_tip + tip;
        ip["0001"] = m_tilt + tilt;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = 0;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_stepSize * 5.;
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_button_tip_ur_pressed()
{
    if( !commandEnabled( ui.button_tip_ur ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = m_stepSize / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_stepSize / sqrt( 2. );
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, -m_stepSize / sqrt( 2. ), m_stepSize / sqrt( 2. ) );

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0000" ) );
        ip.add( pcf::IndiElement( "0001" ) );
        ip["0000"] = m_tip + tip;
        ip["0001"] = m_tilt + tilt;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = m_stepSize * 5. / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_stepSize * 5. / sqrt( 2. );
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_button_tip_scale_pressed()
{
    if( ( (int)( 100 * m_stepSize ) ) == 100 )
    {
        m_stepSize = 0.5;
    }
    else if( ( (int)( 100 * m_stepSize ) ) == 50 )
    {
        m_stepSize = 0.1;
    }
    else if( ( (int)( 100 * m_stepSize ) ) == 10 )
    {
        m_stepSize = 0.05;
    }
    else if( ( (int)( 100 * m_stepSize ) ) == 5 )
    {
        m_stepSize = 0.01;
    }
    else if( ( (int)( 100 * m_stepSize ) ) == 1 )
    {
        m_stepSize = 1.0;
    }

    char ss[5];
    snprintf( ss, 5, "%0.2f", m_stepSize );
    ui.button_tip_scale->setText( ss );
}

void pupilGuide::on_button_focus_p_pressed()
{
    if( !commandEnabled( ui.button_focus_p ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_WOOF )
    {

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0002" ) );

        // round to avoid INDI f.p. precision probs
        ip["0002"] = std::round( ( m_focus + m_focusStepSize * m_focusWooferStep ) * 1e5 ) / 1e5;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "z" ) );
        ip["z"] = m_focusStepSize * m_focusTelStep;
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_button_focus_m_pressed()
{
    if( !commandEnabled( ui.button_focus_m ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_WOOF )
    {

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0002" ) );

        // round to avoid INDI f.p. precision probs
        ip["0002"] = std::round( ( m_focus - m_focusStepSize * m_focusWooferStep ) * 1e5 ) / 1e5;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "z" ) );
        ip["z"] = -m_focusStepSize * m_focusTelStep;
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_button_focus_scale_pressed()
{
    if( ( (int)( 100 * m_focusStepSize ) ) == 100 )
    {
        m_focusStepSize = 0.5;
    }
    else if( ( (int)( 100 * m_focusStepSize ) ) == 50 )
    {
        m_focusStepSize = 0.1;
    }
    else if( ( (int)( 100 * m_focusStepSize ) ) == 10 )
    {
        m_focusStepSize = 0.05;
    }
    else if( ( (int)( 100 * m_focusStepSize ) ) == 5 )
    {
        m_focusStepSize = 0.01;
    }
    else if( ( (int)( 100 * m_focusStepSize ) ) == 1 )
    {
        m_focusStepSize = 1.0;
    }

    char ss[5];
    snprintf( ss, 5, "%0.2f", m_focusStepSize );
    ui.button_focus_scale->setText( ss );
}

//----------- picoscix

void pupilGuide::move_picoscix( int delta )
{
    if( !commandEnabled( ui.picoscix_l ) || m_picoscixPos < -1000000 )
    {
        return;
    }

    const long long target = static_cast<long long>( m_picoscixPos ) + delta;
    if( target < std::numeric_limits<int>::min() || target > std::numeric_limits<int>::max() )
        return;
    int newpos = static_cast<int>( target );

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "picomotors" );
    ip.setName( "picoscix_pos" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = newpos;

    sendNewProperty( ip );
}

void pupilGuide::on_picoscix_l_pressed()
{
    move_picoscix( +m_picoscix_stepSize );
}

void pupilGuide::on_picoscix_scale_pressed()
{
    if( m_picoscix_stepSize == 1000 )
    {
        m_picoscix_stepSize = 500;
    }
    else if( m_picoscix_stepSize == 500 )
    {
        m_picoscix_stepSize = 100;
    }
    else if( m_picoscix_stepSize == 100 )
    {
        m_picoscix_stepSize = 50;
    }
    else
    {
        m_picoscix_stepSize = 1000;
    }

    char ss[5];
    snprintf( ss, 5, "%0.2f", m_picoscix_stepSize / 1000. );
    ui.picoscix_scale->setText( ss );
}

void pupilGuide::on_picoscix_r_pressed()
{
    move_picoscix( -m_picoscix_stepSize );
}

void pupilGuide::on_picoscix_go_pressed()
{
    QString select = ui.picoscix_combo->currentText();

    if( select == "65-35" )
    {
        move_picoscix( -7000 );
    }

    if( select == "Ha-IR" )
    {
        move_picoscix( 7000 );
    }

    ui.picoscix_combo->setCurrentText( "    " );
}

//----------- dmtweeter

void pupilGuide::on_buttonTweeterTest_set_pressed()
{
    if( !commandEnabled( ui.buttonTweeterTest_set ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "dmtweeter" );
    ip.setName( "test_set" );
    ip.add( pcf::IndiElement( "toggle" ) );

    if( m_dmtweeterTestSet )
    {
        ip["toggle"].setSwitchState( pcf::IndiElement::Off );
    }
    else
    {
        ip["toggle"].setSwitchState( pcf::IndiElement::On );
    }

    sendNewProperty( ip );
}

//----------- dmtweeter

void pupilGuide::on_buttonNCPCTest_set_pressed()
{
    if( !commandEnabled( ui.buttonNCPCTest_set ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "dmncpc" );
    ip.setName( "test_set" );
    ip.add( pcf::IndiElement( "toggle" ) );

    if( m_dmncpcTestSet )
    {
        ip["toggle"].setSwitchState( pcf::IndiElement::Off );
    }
    else
    {
        ip["toggle"].setSwitchState( pcf::IndiElement::On );
    }

    sendNewProperty( ip );
}

//----------- ttmpupil

void pupilGuide::on_buttonPup_rest_pressed()
{
    if( !commandEnabled( ui.buttonPup_rest ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmpupil" );
    ip.setName( "releaseDM" );
    ip.add( pcf::IndiElement( "request" ) );
    ip["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ip );
}

void pupilGuide::on_buttonPup_set_pressed()
{
    if( !commandEnabled( ui.buttonPup_set ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmpupil" );
    ip.setName( "initDM" );
    ip.add( pcf::IndiElement( "request" ) );
    ip["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ip );
}

void pupilGuide::on_button_camera_pressed()
{
    if( m_pupCam == FLOWFS )
    {
        m_pupCam = CAMSCIS;
        ui.button_camera->setText( "camsci1/2" );
    }
    else
    {
        m_pupCam = FLOWFS;
        ui.button_camera->setText( "flowfs" );
    }
}

void pupilGuide::on_button_pup_ul_pressed()
{
    if( !commandEnabled( ui.button_pup_ul ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmpupil" );
    ip.setName( "pos_1" );
    ip.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip["target"] = m_pupCh1 + m_pupStepSize / sqrt( 2 );
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip["target"] = m_pupCh1 + m_pupStepSize / sqrt( 2 );
    }

    sendNewProperty( ip );

    pcf::IndiProperty ip2( pcf::IndiProperty::Number );

    ip2.setDevice( "ttmpupil" );
    ip2.setName( "pos_2" );
    ip2.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip2["target"] = m_pupCh2 + m_pupStepSize / sqrt( 2 );
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip2["target"] = m_pupCh2 + m_pupStepSize / sqrt( 2 );
    }

    sendNewProperty( ip2 );
}

void pupilGuide::on_button_pup_dl_pressed()
{
    if( !commandEnabled( ui.button_pup_dl ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmpupil" );
    ip.setName( "pos_1" );
    ip.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip["target"] = m_pupCh1 + m_pupStepSize / sqrt( 2 );
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip["target"] = m_pupCh1 - m_pupStepSize / sqrt( 2 );
    }

    sendNewProperty( ip );

    pcf::IndiProperty ip2( pcf::IndiProperty::Number );

    ip2.setDevice( "ttmpupil" );
    ip2.setName( "pos_2" );
    ip2.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip2["target"] = m_pupCh2 - m_pupStepSize / sqrt( 2 );
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip2["target"] = m_pupCh2 + m_pupStepSize / sqrt( 2 );
    }

    sendNewProperty( ip2 );
}

void pupilGuide::on_button_pup_dr_pressed()
{
    if( !commandEnabled( ui.button_pup_dr ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmpupil" );
    ip.setName( "pos_1" );
    ip.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip["target"] = m_pupCh1 - m_pupStepSize / sqrt( 2 );
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip["target"] = m_pupCh1 - m_pupStepSize / sqrt( 2 );
    }

    sendNewProperty( ip );

    pcf::IndiProperty ip2( pcf::IndiProperty::Number );

    ip2.setDevice( "ttmpupil" );
    ip2.setName( "pos_2" );
    ip2.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip2["target"] = m_pupCh2 - m_pupStepSize / sqrt( 2 );
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip2["target"] = m_pupCh2 - m_pupStepSize / sqrt( 2 );
    }

    sendNewProperty( ip2 );
}

void pupilGuide::on_button_pup_ur_pressed()
{
    if( !commandEnabled( ui.button_pup_ur ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmpupil" );
    ip.setName( "pos_1" );
    ip.add( pcf::IndiElement( "target" ) );
    if( m_pupCam == FLOWFS )
    {
        ip["target"] = m_pupCh1 - m_pupStepSize / sqrt( 2 );
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip["target"] = m_pupCh1 + m_pupStepSize / sqrt( 2 );
    }

    sendNewProperty( ip );

    pcf::IndiProperty ip2( pcf::IndiProperty::Number );

    ip2.setDevice( "ttmpupil" );
    ip2.setName( "pos_2" );
    ip2.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip2["target"] = m_pupCh2 + m_pupStepSize / sqrt( 2 );
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip2["target"] = m_pupCh2 - m_pupStepSize / sqrt( 2 );
    }

    sendNewProperty( ip2 );
}

void pupilGuide::on_button_pup_scale_pressed()
{
    if( ( (int)( 100 * m_pupStepSize ) ) == 100 )
    {
        m_pupStepSize = 0.5;
    }
    else if( ( (int)( 100 * m_pupStepSize ) ) == 50 )
    {
        m_pupStepSize = 0.1;
    }
    else if( ( (int)( 100 * m_pupStepSize ) ) == 10 )
    {
        m_pupStepSize = 0.05;
    }
    else if( ( (int)( 100 * m_pupStepSize ) ) == 5 )
    {
        m_pupStepSize = 0.01;
    }
    else if( ( (int)( 100 * m_pupStepSize ) ) == 1 )
    {
        m_pupStepSize = 1.0;
    }

    char ss[5];
    snprintf( ss, 5, "%0.2f", m_pupStepSize );
    ui.button_pup_scale->setText( ss );
}

void pupilGuide::on_button_ttmPeri_rest_pressed()
{
    if( !commandEnabled( ui.button_ttmPeri_rest ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmperi" );
    ip.setName( "set" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"].setSwitchState( pcf::IndiElement::Off );

    sendNewProperty( ip );
}

void pupilGuide::on_button_ttmPeri_set_pressed()
{
    if( !commandEnabled( ui.button_ttmPeri_set ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmperi" );
    ip.setName( "set" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ip );
}

void pupilGuide::on_button_ttmPeri_l_pressed()
{
    if( !commandEnabled( ui.button_ttmPeri_l ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmperi" );
    ip.setName( "axis1_voltage" );
    ip.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip["target"] = m_ttmPeriCh1 + m_ttmPeriStepSize;
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip["target"] = m_ttmPeriCh1 + m_ttmPeriStepSize;
    }

    sendNewProperty( ip );
}

void pupilGuide::on_button_ttmPeri_r_pressed()
{
    if( !commandEnabled( ui.button_ttmPeri_r ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmperi" );
    ip.setName( "axis1_voltage" );
    ip.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip["target"] = m_ttmPeriCh1 - m_ttmPeriStepSize;
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip["target"] = m_ttmPeriCh1 - m_ttmPeriStepSize;
    }

    sendNewProperty( ip );
}

void pupilGuide::on_button_ttmPeri_u_pressed()
{
    if( !commandEnabled( ui.button_ttmPeri_u ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmperi" );
    ip.setName( "axis2_voltage" );
    ip.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip["target"] = m_ttmPeriCh2 + m_ttmPeriStepSize;
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip["target"] = m_ttmPeriCh2 + m_ttmPeriStepSize;
    }

    sendNewProperty( ip );
}

void pupilGuide::on_button_ttmPeri_d_pressed()
{
    if( !commandEnabled( ui.button_ttmPeri_d ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmperi" );
    ip.setName( "axis2_voltage" );
    ip.add( pcf::IndiElement( "target" ) );

    if( m_pupCam == FLOWFS )
    {
        ip["target"] = m_ttmPeriCh2 - m_ttmPeriStepSize;
    }
    else if( m_pupCam == LLOWFS )
    {
    }
    else if( m_pupCam == CAMSCIS )
    {
        ip["target"] = m_ttmPeriCh2 - m_ttmPeriStepSize;
    }

    sendNewProperty( ip );
}

void pupilGuide::on_button_ttmPeri_scale_pressed()
{
    if( ( (int)( m_ttmPeriStepSize ) ) == 50 )
    {
        m_ttmPeriStepSize = 25;
    }
    else if( ( (int)( m_ttmPeriStepSize ) ) == 25 )
    {
        m_ttmPeriStepSize = 10;
    }
    else if( ( (int)( m_ttmPeriStepSize ) ) == 10 )
    {
        m_ttmPeriStepSize = 1;
    }
    else
    {
        m_ttmPeriStepSize = 50;
    }

    char ss[5];
    snprintf( ss, 5, "%0.2f", m_ttmPeriStepSize / 100. );
    ui.button_ttmPeri_scale->setText( ss );
}

void pupilGuide::toggleExpFit( bool st )
{

    ui.labelD->setVisible( st );

    ui.labelUR->setVisible( st );
    ui.coordUR_x->setVisible( st );
    ui.coordUR_y->setVisible( st );
    ui.coordUR_D->setVisible( st );

    ui.labelUL->setVisible( st );
    ui.coordUL_x->setVisible( st );
    ui.coordUL_y->setVisible( st );
    ui.coordUL_D->setVisible( st );

    ui.labelLR->setVisible( st );
    ui.coordLR_x->setVisible( st );
    ui.coordLR_y->setVisible( st );
    ui.coordLR_D->setVisible( st );

    ui.labelLL->setVisible( st );
    ui.coordLL_x->setVisible( st );
    ui.coordLL_y->setVisible( st );
    ui.coordLL_D->setVisible( st );

    ui.coordAvg_D->setVisible( st );

    if( st )
    {
        ui.buttonExpFit->setIcon( QIcon( ":/icons/keyboard_double_arrow_up.png" ) );
    }
    else
    {
        ui.buttonExpFit->setIcon( QIcon( ":/icons/keyboard_double_arrow_down.png" ) );
    }
}

void pupilGuide::on_buttonExpFit_pressed()
{
    bool st = !ui.labelD->isVisible();
    toggleExpFit( st );
}

void pupilGuide::on_button_camlens_u_pressed()
{
    if( !commandEnabled( ui.button_camlens_u ) )
        return;
    if( m_camlensyFsmState != "READY" )
        return;

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "stagecamlensy" );
    ip.setName( "position" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = m_camlensy_pos - m_camlensStepSize;

    sendNewProperty( ip );
}

void pupilGuide::on_button_camlens_l_pressed()
{
    if( !commandEnabled( ui.button_camlens_l ) )
        return;
    if( m_camlensxFsmState != "READY" )
        return;

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "stagecamlensx" );
    ip.setName( "position" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = m_camlensx_pos - m_camlensStepSize;

    sendNewProperty( ip );
}

void pupilGuide::on_button_camlens_d_pressed()
{
    if( !commandEnabled( ui.button_camlens_d ) )
        return;
    if( m_camlensyFsmState != "READY" )
    {
        return;
    }

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "stagecamlensy" );
    ip.setName( "position" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = m_camlensy_pos + m_camlensStepSize;

    sendNewProperty( ip );
}

void pupilGuide::on_button_camlens_r_pressed()
{
    if( !commandEnabled( ui.button_camlens_r ) )
        return;
    if( m_camlensxFsmState != "READY" )
    {
        return;
    }

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "stagecamlensx" );
    ip.setName( "position" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = m_camlensx_pos + m_camlensStepSize;
    sendNewProperty( ip );
}

void pupilGuide::on_button_camlens_scale_pressed()
{
    if( ( (int)( 1000 * m_camlensStepSize + 0.5 ) ) == 5 )
    {
        m_camlensStepSize = 0.05;
    }
    else if( ( (int)( 1000 * m_camlensStepSize + 0.5 ) ) == 50 )
    {
        m_camlensStepSize = 0.025;
    }
    else if( ( (int)( 1000 * m_camlensStepSize + 0.5 ) ) == 25 )
    {
        m_camlensStepSize = 0.01;
    }
    else if( ( (int)( 1000 * m_camlensStepSize + 0.5 ) ) == 10 )
    {
        m_camlensStepSize = 0.005;
    }

    char ss[5];
    snprintf( ss, 5, "%0.2f", m_camlensStepSize * 10 );
    ui.button_camlens_scale->setText( ss );
}

void pupilGuide::on_button_startAlignment_pressed()
{
    if( !commandEnabled( ui.button_startAlignment ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "twAlign-camwfs-wfs" );
    ip.setName( "continuous" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"] = pcf::IndiElement::On;

    sendNewProperty( ip );

    ip.setDevice( "twAlign-camwfs-ctrl" );
    ip.setName( "loop_state" );
    ip["toggle"] = pcf::IndiElement::On;

    sendNewProperty( ip );

    ip.setDevice( "camwfs-align" );
    ip.setName( "loop_state" );
    ip["toggle"] = pcf::IndiElement::On;

    sendNewProperty( ip );
}

void pupilGuide::on_button_stopAlignment_pressed()
{
    if( !commandEnabled( ui.button_stopAlignment ) )
        return;
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"] = pcf::IndiElement::Off;
    for( const auto &target : { std::make_pair( "twAlign-camwfs-wfs", "continuous" ),
                                std::make_pair( "twAlign-camwfs-ctrl", "loop_state" ),
                                std::make_pair( "camwfs-align", "loop_state" ) } )
    {
        if( std::string( target.first ) == "camwfs-align" && !m_labMode )
            continue;
        if( !propertyAvailable( target.first, target.second, { "toggle" } ) )
            continue;
        ip.setDevice( target.first );
        ip.setName( target.second );
        sendNewProperty( ip );
    }
}

} // namespace xqt

#include "moc_pupilGuide.cpp"

#endif
