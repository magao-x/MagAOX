#ifndef pupilGuide_hpp
#define pupilGuide_hpp

#include <cmath>
#include <unistd.h>

#include <QWidget>
#include <QMutex>
#include <QTimer>

#include "ui_pupilGuide.h"

#include "../xWidgets/xWidget.hpp"
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

void wooferTipTilt( double &tip, double &tilt, double x, double y )
{
    double rot   = ( 180. + 29.0 ) * 3.14159 / 180.;
    double scale = -1.0;

    tip  = scale * ( x * cos( rot ) - y * sin( rot ) );
    tilt = scale * ( x * sin( rot ) + y * cos( rot ) );
}

class pupilGuide : public xWidget
{
    Q_OBJECT

  public:
    pupilGuide( QWidget *Parent = 0, Qt::WindowFlags f = Qt::WindowFlags() );

    ~pupilGuide();

    // INDI Interface
    void subscribe();

    virtual void onConnect();
    virtual void onDisconnect();

    void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );
    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

  protected:
    // === Acquisition and Alignment === //
    // *** Acquistion *** //
    // --- Target --- //
    std::string m_tcsiFsmState;
    std::string m_tcsiCatObj;
    bool        m_tcsiLabMode{ true };

    std::string m_observersFsmState;
    std::string m_observersTarget;

  protected slots:
    /// Synchronize the observer's target with the tcsi's catObj.
    void on_target_sync_pressed();

    // --- Telescope --- //

    // Send the acquire from guider request
    void on_telacq_acqfromguider_pressed();

    // Send the acquisition focus offset
    void on_telacq_acqfocus_pressed();

    // --- camacq --- //
    void on_camacq_acqstar_activated( int index );

    void on_camacq_acq_go_pressed();

    void on_camacq_seestar_activated( int index );

    void on_camacq_see_go_pressed();

    void on_camacq_acq_restart_pressed();

    void on_camacq_see_measure_pressed();

  protected:
    // *** PWFS Alignment *** //
    // --- modwfs --- //

    std::string m_modwfsFsmState;

    int m_modState{ 0 };

    double m_modFreq{ 0 };
    double m_modFreqTarget{ 0 };

    double m_modRad{ 0 };
    double m_modRad_tgt{ 0 };

    double m_camwfsFreq{ 0 };

  protected slots:

    void on_modwfs_rest_pressed();
    void on_modwfs_set_pressed();
    void on_modwfs_modulate_pressed();

  protected:
    // --- pwfsacq --- //
    std::string m_dmWooferFsmState;
    std::string m_wooferModesFsmState;

    double m_tilt{ 0 };  ///< current value of tilt mode from wooferModes
    double m_tip{ 0 };   ///< current value of tip mode from wooferModes
    double m_focus{ 0 }; ///< current value of focus mode from wooferModes

    int m_tipmovewhat{ MOVE_TTM };

    float m_pwfsacqScale{ 0.5 }; // 0.5 means that it will be 0.1 on initialization

    float m_pwfsacqFocScale{ 0.1 };

  protected slots:

    void on_pwfsacq_what_pressed();

    void on_pwfsacq_u_pressed();
    void on_pwfsacq_ul_pressed();
    void on_pwfsacq_l_pressed();
    void on_pwfsacq_dl_pressed();
    void on_pwfsacq_d_pressed();
    void on_pwfsacq_dr_pressed();
    void on_pwfsacq_r_pressed();
    void on_pwfsacq_ur_pressed();
    void on_pwfsacq_scale_pressed();

    void on_pwfsfoc_p_pressed();
    void on_pwfsfoc_m_pressed();
    void on_pwfsfoc_scale_pressed();

    // **** Pupil Alignment **** //

    // --- Act Align Loop ---- //

    // --- Act Align Sensor --- //
    // --- Pupil Tracking Loop --- //
    // --- Auto Pupil Alignment --- //

    // void on_autoalign_start_pressed();
    // void on_autoalign_stop_pressed();

    // ******** alignment *********//

    // ===== Manual Alignment ===== //
    // **** F-test **** //
    // --- Tweeter --- //

  protected slots:
    void on_tweeter_set_pressed();

  protected:
    // --- TTM Pupil Buttons --- //

  protected slots:
    void on_ttmpupil_ul_pressed();
    void on_ttmpupil_dl_pressed();
    void on_ttmpupil_dr_pressed();
    void on_ttmpupil_ur_pressed();
    void on_ttmpupil_scale_pressed();

  protected:
    // --- TTM Pupil --- //

  protected slots:
    void on_ttmpupil_rest_pressed();
    void on_ttmpupil_set_pressed();

  protected:
    // **** J-test **** //
    // --- NCPC --- //

  protected slots:
    void on_ncpc_set_pressed();

  protected:
    // --- TTM Peri. Buttons --- //

  protected slots:
    void on_ttmperi_l_pressed();
    void on_ttmperi_r_pressed();
    void on_ttmperi_u_pressed();
    void on_ttmperi_d_pressed();
    void on_ttmperi_scale_pressed();

  protected:
    // --- TTM Peri. --- //

    std::string m_ttmPeriFsmState;
    double      m_ttmperi_ch1{ 0 };
    double      m_ttmperi_ch2{ 0 };

    float m_ttmPeriStepSize{ 50 }; // This will be 25 after init

  protected slots:
    void on_ttmperi_rest_pressed();
    void on_ttmperi_set_pressed();

  protected:
    // **** PWFS Pupils **** //
    // --- Pupil Fitting --- //
    // --- Pupil Positions --- //
    // --- Camera Lens --- //
    std::string m_camlensxFsmState;
    std::string m_camlensyFsmState;
    float       m_camlensx_pos{ 0 };
    float       m_camlensy_pos{ 0 };

    float m_camlensStepSize{ 0.025 }; // this will set it to 0.01 after init
    // --- Camera Lens Buttons --- //
  protected slots:
    void on_camlens_u_pressed();
    void on_camlens_l_pressed();
    void on_camlens_d_pressed();
    void on_camlens_r_pressed();
    void on_camlens_scale_pressed();

  protected:
    // **** Pico Sci-x **** //

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  protected:
    // --- picoscix
    std::string m_picoState{ "UNKNOWN" };
    int         m_picoscixPos{ -1000000000 };

    int         m_picoscix_stepSize{ 50 };
    std::string m_picoscix_gotoSelection;

    // --- camwfs-fit
    std::string m_camwfsfitState;
    double      m_pwfs_median_1{ 0 };
    double      m_pwfs_median_2{ 0 };
    double      m_pwfs_median_3{ 0 };
    double      m_pwfs_median_4{ 0 };

    double m_x1{ 0 };
    double m_y1{ 0 };
    double m_D1{ 0 };

    double m_setx1{ 0 };
    double m_sety1{ 0 };
    double m_setD1{ 0 };

    double m_x2{ 0 };
    double m_y2{ 0 };
    double m_D2{ 0 };

    double m_setx2{ 0 };
    double m_sety2{ 0 };
    double m_setD2{ 0 };

    double m_x3{ 0 };
    double m_y3{ 0 };
    double m_D3{ 0 };

    double m_setx3{ 0 };
    double m_sety3{ 0 };
    double m_setD3{ 0 };

    double m_x4{ 0 };
    double m_y4{ 0 };
    double m_D4{ 0 };

    double m_setx4{ 0 };
    double m_sety4{ 0 };
    double m_setD4{ 0 };

    double m_threshold_current{ 0 };
    double m_threshold_target{ 0 };

    // -- camwfs-avg
    std::string m_camwfsavgState;
    unsigned    m_nAverage_current{ 0 };
    unsigned    m_nAverage_target{ 0 };

    // -- dmtweeter
    std::string m_dmtweeterState;
    bool        m_dmtweeterTestSet{ false };

    // -- dmncpc
    std::string m_dmncpcState;
    bool        m_dmncpcTestSet{ false };

    // -- ttmpupil
    std::string m_pupFsmState;
    double      m_ttmpupil_ch1{ 0 };
    double      m_ttmpupil_ch2{ 0 };

    float m_pupStepSize{ 0.5 };

    // -- Camera Lens

    // ****** Alignment ******** //

    // --- camwfs-align
    std::string m_camwfs_align_fsmState;
    bool        m_camwfsAlignLoopState{ false };

    // --- twAlign-camwfs-ctrl
    std::string m_twAlign_camwfs_ctrl_fsmState;
    bool        m_twAlignLoopState{ false };

    // --- twAlign-camwfs-wfs
    std::string m_twAlign_camwfs_wfs_fsmState;
    bool        m_twAlignSensorState{ false };

    void modGUISetEnable( bool enableModGUI, bool enableModArrows );

    void camwfsfitSetEnabled( bool enabled );

    /// Enable or disable the cameralens GUI
    /** IF whichcl is CAMLENS_BOTH, the action is applied to all components.
     * If it's CAMLENS_X or CAMLENS_Y, it is only applied to that access.  The common components are then enabled.
     */
    void camlensSetEnabled( bool enabled,               ///< true for enabled, false for disabled
                            int  whichcl = CAMLENS_BOTH ///< Which axis, or both.  CAMLENS_X, CAMLENS_Y, CAMLENS_BOTH
    );

    void camwfs_align_setEnabled( bool enabled, bool all );

    void twAlign_camwfs_ctrl_setEnabled( bool enabled, bool all );

    void twAlign_camwfs_wfs_setEnabled( bool enabled, bool all );

    void alignment_buttons_setEnabled( bool enabled, bool all );

  public slots:
    void updateGUI();

    //----------- modttm

    //------------- picoscix
    void move_picoscix( int delta );
    void on_picoscix_l_pressed();
    void on_picoscix_scale_pressed();
    void on_picoscix_r_pressed();
    void on_picoscix_go_pressed();

    //----------- ttmpupil

    //---------- TTM Peri

    void toggleExpFit( bool visible );
    void on_pwfs_coords_expand_pressed();

  private:
    Ui::pupilGuide ui;
};

pupilGuide::pupilGuide( QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f )
{
    char ss[64]; // for scale buttons

    ui.setupUi( this );

    // ===== Acquisition and Alignment ===== //
    // **** Acquistion **** //
    // --- Target --- //
    setXwFont( ui.target_label );

    ui.target_catobj->setup( "tcsi", "catalog", statusEntry::STRING, "Catalog", "" );
    ui.target_catobj->currEl( "object" );
    ui.target_catobj->targEl( "" );
    ui.target_catobj->readOnly( true );
    ui.target_catobj->setStretch( 0, 1, 2 );

    ui.target_name->setup( "observers", "target", statusEntry::STRING, "Observer", "" );
    ui.target_name->setStretch( 0, 1, 2 );

    // --- Telescope --- //

    setXwFont( ui.telacq_label );
    setXwFont( ui.telacq_acqfromguider );
    setXwFont( ui.telacq_acqfocus );
    setXwFont( ui.telacq_override_label );

    // --- camacq --- //

    setXwFont( ui.camacq_label );
    setXwFont( ui.camacq_acq_label );
    setXwFont( ui.camacq_see_label );
    setXwFont( ui.camacq_acq_restart );
    setXwFont( ui.camacq_see_measure );

    // **** PWFS Alignment **** //
    // --- modwfs --- //

    setXwFont( ui.modwfs_label );

    ui.modwfs_fsm->device( "modwfs" );
    ui.modwfs_fsm->NOTHOMED( "RIP" );
    ui.modwfs_fsm->READY( "SET" );
    ui.modwfs_fsm->OPERATING( "MODULATING" );

    setXwFont( ui.modwfs_freq_label );
    setXwFont( ui.modwfs_rad_label );

    ui.modwfs_freq->setup( "modwfs", "modFrequency", statusEntry::FLOAT, "", "" );
    ui.modwfs_freq->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.modwfs_freq->format( "%0.1f" );

    ui.modwfs_rad->setup( "modwfs", "modRadius", statusEntry::FLOAT, "", "" );
    ui.modwfs_rad->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.modwfs_rad->format( "%0.1f" );

    setXwFont( ui.modwfs_rest );
    setXwFont( ui.modwfs_set );
    setXwFont( ui.modwfs_modulate );

    ui.modwfs_ch1->setup( "fxngenmodwfs", "C1ofst", statusEntry::FLOAT, "Ch1", "V" );
    ui.modwfs_ch1->currEl( "value" );
    ui.modwfs_ch1->targEl( "value" );
    ui.modwfs_ch1->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.modwfs_ch1->format( "%0.2f" );

    ui.modwfs_ch2->setup( "fxngenmodwfs", "C2ofst", statusEntry::FLOAT, "Ch2", "V" );
    ui.modwfs_ch2->currEl( "value" );
    ui.modwfs_ch2->targEl( "value" );
    ui.modwfs_ch2->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.modwfs_ch2->format( "%0.2f" );

    // --- pwfsacq --- //

    setXwFont( ui.tip_alignment_label );

    setXwFont( ui.pwfsacq_what );

    ui.pwfsacq_scale->setProperty( "isScaleButton", true );
    on_pwfsacq_scale_pressed();

    ui.pwfsfoc_scale->setProperty( "isScaleButton", true );
    on_pwfsfoc_scale_pressed();

    // **** Pupil Alignment **** //
    // --- Act Align Loop ---- //

    setXwFont( ui.actalign_loop_label );

    ui.actalign_loop_deltaX->setup( "twAlign-camwfs-ctrl", "deltas", statusEntry::FLOAT, "", "" );
    ui.actalign_loop_deltaX->currEl( "delta0" );
    ui.actalign_loop_deltaX->highlightChanges( false );
    ui.actalign_loop_deltaX->readOnly( true );
    ui.actalign_loop_deltaX->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.actalign_loop_deltaX->format( "%0.03f" );

    ui.actalign_loop_deltaY->setup( "twAlign-camwfs-ctrl", "deltas", statusEntry::FLOAT, "", "" );
    ui.actalign_loop_deltaY->currEl( "delta1" );
    ui.actalign_loop_deltaY->highlightChanges( false );
    ui.actalign_loop_deltaY->readOnly( true );
    ui.actalign_loop_deltaY->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.actalign_loop_deltaY->format( "%0.03f" );

    ui.actalign_loop_slider->setup( "twAlign-camwfs-ctrl", "loop_state", "toggle", "" );
    ui.actalign_loop_slider->setStretch( 0, 0, 10, true, true );

    ui.actalign_loop_gain->setup( "twAlign-camwfs-ctrl", "loop_gain", statusEntry::FLOAT, "loop gain", "" );
    ui.actalign_loop_gain->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.actalign_loop_gain->format( "%0.2f" );

    // --- Act Align Sensor --- //

    setXwFont( ui.actalign_sensor_label );

    ui.actalign_sensor_slider->setup( "twAlign-camwfs-wfs", "continuous", "toggle", "" );
    ui.actalign_sensor_slider->setStretch( 0, 0, 10, true, true );

    ui.actalign_sensor_nAverage->setup( "twAlign-camwfs-wfs", "nPokeAverage", statusEntry::INT, "no. average", "" );
    ui.actalign_sensor_nAverage->setStretch( 1, 3, 6 );
    ui.actalign_sensor_nAverage->format( "%d" );

    ui.actalign_sensor_nImages->setup( "twAlign-camwfs-wfs", "nPokeImages", statusEntry::INT, "no. images", "" );
    ui.actalign_sensor_nImages->setStretch( 1, 3, 6 );
    ui.actalign_sensor_nImages->format( "%d" );

    ui.actalign_sensor_pokeAmp->setup( "twAlign-camwfs-wfs", "poke_amp", statusEntry::FLOAT, "poke amp.", "um" );
    ui.actalign_sensor_pokeAmp->setStretch( 1, 3, 6 );
    ui.actalign_sensor_pokeAmp->format( "%0.2f" );

    // --- Pupil Tracking Loop --- //

    setXwFont( ui.puptrack_loop_label );

    ui.puptrack_loop_deltaX->setup( "camwfs-align", "deltas", statusEntry::FLOAT, "", "" );
    ui.puptrack_loop_deltaX->currEl( "delta0" );
    ui.puptrack_loop_deltaX->highlightChanges( false );
    ui.puptrack_loop_deltaX->readOnly( true );
    ui.puptrack_loop_deltaX->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.puptrack_loop_deltaX->format( "%0.03f" );

    ui.puptrack_loop_deltaY->setup( "camwfs-align", "deltas", statusEntry::FLOAT, "", "" );
    ui.puptrack_loop_deltaY->currEl( "delta1" );
    ui.puptrack_loop_deltaY->highlightChanges( false );
    ui.puptrack_loop_deltaY->readOnly( true );
    ui.puptrack_loop_deltaY->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.puptrack_loop_deltaY->format( "%0.03f" );

    ui.puptrack_loop_slider->setup( "camwfs-align", "loop_state", "toggle", "" );
    ui.puptrack_loop_slider->setStretch( 0, 0, 10, true, true );

    ui.puptrack_loop_gain->setup( "camwfs-align", "loop_gain", statusEntry::FLOAT, "loop gain", "" );
    ui.puptrack_loop_gain->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.puptrack_loop_gain->format( "%0.2f" );

    // --- Auto Pupil Alignment --- //

    setXwFont( ui.autoalign_label );
    // Have to deal wit this -- it's not actually a toggle slider but it should be
    // ui.autoalign_slider->setup( "camwfs-align", "loop_state", "toggle", "" );
    // ui.autoalign_slider->setStretch( 0, 0, 10, true, true );

    // ===== Manual Alignment ===== //
    // **** F-test **** //
    // --- Tweeter --- //

    setXwFont( ui.tweeter_label );
    setXwFont( ui.tweeter_set );

    // --- TTM Pupil Buttons --- //

    ui.ttmpupil_scale->setProperty( "isScaleButton", true );
    on_ttmpupil_scale_pressed();

    // --- TTM Pupil --- //

    setXwFont( ui.ttmpupil_label );
    setXwFont( ui.ttmpupil_rest );
    setXwFont( ui.ttmpupil_set );
    ui.ttmpupil_fsm->device( "ttmpupil" );
    ui.ttmpupil_fsm->NOTHOMED( "RIP" );
    ui.ttmpupil_fsm->HOMING( "SETTING" );
    ui.ttmpupil_fsm->READY( "SET" );

    ui.ttmpupil_ch1->setup( "ttmpupil", "pos_1", statusEntry::FLOAT, "Ch 1", "V" );
    ui.ttmpupil_ch1->setStretch( 1, 2, 4 );
    ui.ttmpupil_ch1->highlightChanges( false );

    ui.ttmpupil_ch2->setup( "ttmpupil", "pos_2", statusEntry::FLOAT, "Ch 2", "V" );
    ui.ttmpupil_ch2->setStretch( 1, 2, 4 );
    ui.ttmpupil_ch2->highlightChanges( false );

    // **** J-test **** //
    // --- NCPC --- //

    setXwFont( ui.ncpc_label );
    setXwFont( ui.ncpc_set );

    // --- TTM Peri. Buttons --- //

    ui.ttmperi_scale->setProperty( "isScaleButton", true );
    on_ttmperi_scale_pressed();

    // --- TTM Peri. --- //

    setXwFont( ui.ttmperi_label );
    setXwFont( ui.ttmperi_rest );
    setXwFont( ui.ttmperi_set );
    ui.ttmperi_fsm->device( "ttmperi" );
    ui.ttmperi_fsm->READY( "RIP" );
    ui.ttmperi_fsm->OPERATING( "SET" );

    ui.ttmperi_ch1->setup( "ttmperi", "axis1_voltage", statusEntry::FLOAT, "Ch 1", "V" );
    ui.ttmperi_ch1->setStretch( 1, 2, 4 );
    ui.ttmperi_ch1->highlightChanges( false );

    ui.ttmperi_ch2->setup( "ttmperi", "axis2_voltage", statusEntry::FLOAT, "Ch 2", "V" );
    ui.ttmperi_ch2->highlightChanges( false );
    ui.ttmperi_ch2->setStretch( 1, 2, 4 );

    // **** PWFS Pupils **** //
    // --- Pupil Fitting --- //

    setXwFont( ui.pwfsfit_label );

    ui.pwfsfit_threshold->setup( "camwfs-fit", "threshold", statusEntry::FLOAT, "Thresh", "" );
    ui.pwfsfit_threshold->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.pwfsfit_threshold->format( "%0.3f" );

    ui.pwfsfit_avgtime->setup( "camwfs-avg", "avgTime", statusEntry::FLOAT, "Avg. T.", "s" );
    ui.pwfsfit_avgtime->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.pwfsfit_avgtime->format( "%0.3f" );

    setXwFont( ui.pwfs_medians_label );
    setXwFont( ui.pwfs_median_1 );
    setXwFont( ui.pwfs_median_2 );
    setXwFont( ui.pwfs_median_3 );
    setXwFont( ui.pwfs_median_4 );
    setXwFont( ui.pwfs_medians_delta );

    // --- Pupil Positions --- //

    setXwFont( ui.pwfs_coords_label );
    setXwFont( ui.pwfs_x_label );
    setXwFont( ui.pwfs_y_label );
    setXwFont( ui.pwfs_D_label );
    setXwFont( ui.pwfs_UR_label );
    setXwFont( ui.pwfs_UL_label );
    setXwFont( ui.pwfs_LR_label );
    setXwFont( ui.pwfs_LL_label );
    setXwFont( ui.pwfs_avg_label );
    setXwFont( ui.pwfs_UR_x );
    setXwFont( ui.pwfs_UR_y );
    setXwFont( ui.pwfs_UR_D );
    setXwFont( ui.pwfs_UL_x );
    setXwFont( ui.pwfs_UL_y );
    setXwFont( ui.pwfs_UL_D );
    setXwFont( ui.pwfs_LR_x );
    setXwFont( ui.pwfs_LR_y );
    setXwFont( ui.pwfs_LR_D );
    setXwFont( ui.pwfs_LL_x );
    setXwFont( ui.pwfs_LL_y );
    setXwFont( ui.pwfs_LL_D );
    setXwFont( ui.pwfs_avg_x );
    setXwFont( ui.pwfs_avg_y );
    setXwFont( ui.pwfs_avg_D );

    setXwFont( ui.pwfs_coords_delta );

    // --- Camera Lens --- //

    setXwFont( ui.camlens_label );
    setXwFont( ui.camlens_fsm_x_label );
    setXwFont( ui.camlens_fsm_y_label );
    ui.camlens_fsm_x->device( "stagecamlensx" );
    ui.camlens_fsm_y->device( "stagecamlensy" );

    ui.camlens_x->setup( "stagecamlensx", "position", statusEntry::FLOAT, "X", "mm" );
    ui.camlens_x->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.camlens_x->format( "%0.4f" );

    ui.camlens_y->setup( "stagecamlensy", "position", statusEntry::FLOAT, "Y", "mm" );
    ui.camlens_y->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.camlens_y->format( "%0.4f" );

    // --- Camera Lens Buttons --- //
    ui.camlens_scale->setProperty( "isScaleButton", true );
    on_camlens_scale_pressed();

    // **** Pico Sci-x **** //

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    setXwFont( ui.picoscix_label );

    ui.picoscix_pos->setup( "picomotors", "picoscix_pos", statusEntry::INT, "", "" );
    ui.picoscix_pos->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.picoscix_pos->format( "%d" );

    ui.picoscix_scale->setProperty( "isScaleButton", true );
    snprintf( ss, 5, "%0.2f", m_picoscix_stepSize / 1000. );
    ui.picoscix_scale->setText( ss );

    ui.picoscix_combo->addItem( "    " );
    ui.picoscix_combo->addItem( "65-35" );
    ui.picoscix_combo->addItem( "Ha-IR" );
    ui.picoscix_combo->setCurrentText( "    " );

    setXwFont( ui.picoscix_combo_label );

    /* pupil tracking loop */

    /* actuator alignment loop */

    /* actuator alignment sensor */

    /* alignment start/stop */

    /* Camera Lens */

    // Set the pupil fit boxes to invisible at startup
    toggleExpFit( false );

    onDisconnect();

    QTimer *timer = new QTimer( this );
    connect( timer, SIGNAL( timeout() ), this, SLOT( updateGUI() ) );
    timer->start( 250 );
}

pupilGuide::~pupilGuide()
{
}

void pupilGuide::subscribe()
{
    if( m_parent == nullptr )
    {
        return;
    }

    m_parent->addSubscriber( ui.target_catobj );
    m_parent->addSubscriber( ui.target_name );

    m_parent->addSubscriber( ui.modwfs_fsm );
    m_parent->addSubscriberProperty( this, "modwfs", "fsm" );
    m_parent->addSubscriberProperty( this, "modwfs", "modState" );

    m_parent->addSubscriber( ui.modwfs_freq );
    m_parent->addSubscriber( ui.modwfs_rad );
    m_parent->addSubscriber( ui.modwfs_ch1 );
    m_parent->addSubscriber( ui.modwfs_ch2 );

    m_parent->addSubscriberProperty( this, "camwfs", "fps" );

    m_parent->addSubscriberProperty( this, "tcsi", "fsm" );
    m_parent->addSubscriberProperty( this, "tcsi", "labMode" );

    m_parent->addSubscriberProperty( this, "dmwoofer", "fsm" );
    m_parent->addSubscriberProperty( this, "wooferModes", "fsm" );
    m_parent->addSubscriberProperty( this, "wooferModes", "current_amps" );

    m_parent->addSubscriber( ui.picoscix_pos );
    m_parent->addSubscriberProperty( this, "picomotors", "fsm" );
    m_parent->addSubscriberProperty( this, "picomotors", "picoscix_pos" );

    m_parent->addSubscriberProperty( this, "camwfs-fit", "fsm" );
    m_parent->addSubscriberProperty( this, "camwfs-fit", "quadrant1" );
    m_parent->addSubscriberProperty( this, "camwfs-fit", "quadrant2" );
    m_parent->addSubscriberProperty( this, "camwfs-fit", "quadrant3" );
    m_parent->addSubscriberProperty( this, "camwfs-fit", "quadrant4" );
    m_parent->addSubscriberProperty( this, "camwfs-fit", "threshold" );

    m_parent->addSubscriberProperty( this, "camwfs-avg", "fsm" );
    m_parent->addSubscriberProperty( this, "camwfs-avg", "nAverage" );

    m_parent->addSubscriber( ui.ttmpupil_fsm );
    m_parent->addSubscriber( ui.ttmpupil_ch1 );
    m_parent->addSubscriber( ui.ttmpupil_ch2 );
    m_parent->addSubscriberProperty( this, "ttmpupil", "fsm" );
    m_parent->addSubscriberProperty( this, "ttmpupil", "pos_1" );
    m_parent->addSubscriberProperty( this, "ttmpupil", "pos_2" );

    m_parent->addSubscriber( ui.ttmperi_fsm );
    m_parent->addSubscriber( ui.ttmperi_ch1 );
    m_parent->addSubscriber( ui.ttmperi_ch2 );
    m_parent->addSubscriberProperty( this, "ttmperi", "fsm" );
    m_parent->addSubscriberProperty( this, "ttmperi", "axis1_voltage" );
    m_parent->addSubscriberProperty( this, "ttmperi", "axis2_voltage" );

    m_parent->addSubscriberProperty( this, "dmtweeter", "fsm" );
    m_parent->addSubscriberProperty( this, "dmtweeter", "test_set" );
    m_parent->addSubscriberProperty( this, "dmtweeter", "test" );

    m_parent->addSubscriberProperty( this, "dmncpc", "fsm" );
    m_parent->addSubscriberProperty( this, "dmncpc", "test_set" );
    m_parent->addSubscriberProperty( this, "dmncpc", "test" );

    m_parent->addSubscriberProperty( this, "camwfs-align", "fsm" );
    m_parent->addSubscriberProperty( this, "camwfs-align", "loop_state" );

    m_parent->addSubscriber( ui.puptrack_loop_deltaX );
    m_parent->addSubscriber( ui.puptrack_loop_deltaY );

    m_parent->addSubscriber( ui.puptrack_loop_slider );
    m_parent->addSubscriber( ui.puptrack_loop_gain );

    m_parent->addSubscriberProperty( this, "twAlign-camwfs-ctrl", "fsm" );
    m_parent->addSubscriberProperty( this, "twAlign-camwfs-ctrl", "loop_state" );

    m_parent->addSubscriber( ui.actalign_loop_deltaX );
    m_parent->addSubscriber( ui.actalign_loop_deltaY );

    m_parent->addSubscriber( ui.actalign_loop_slider );

    m_parent->addSubscriber( ui.actalign_loop_gain );

    m_parent->addSubscriberProperty( this, "twAlign-camwfs-wfs", "fsm" );
    m_parent->addSubscriberProperty( this, "twAlign-camwfs-wfs", "loop_state" );

    m_parent->addSubscriber( ui.actalign_sensor_slider );

    m_parent->addSubscriber( ui.actalign_sensor_nAverage );
    m_parent->addSubscriber( ui.actalign_sensor_nImages );
    m_parent->addSubscriber( ui.actalign_sensor_pokeAmp );

    m_parent->addSubscriber( ui.pwfsfit_threshold );
    m_parent->addSubscriber( ui.pwfsfit_avgtime );

    /* Camera Lens */
    m_parent->addSubscriber( ui.camlens_fsm_x );
    m_parent->addSubscriber( ui.camlens_fsm_y );
    m_parent->addSubscriberProperty( this, "stagecamlensx", "fsm" );
    m_parent->addSubscriberProperty( this, "stagecamlensy", "fsm" );
    m_parent->addSubscriberProperty( this, "stagecamlensx", "position" ); // we need these too
    m_parent->addSubscriberProperty( this, "stagecamlensy", "position" );
    m_parent->addSubscriber( ui.camlens_x );
    m_parent->addSubscriber( ui.camlens_y );

    return;
}

void pupilGuide::onConnect()
{

    ui.target_catobj->onConnect();
    ui.target_name->onConnect();

    ui.modwfs_label->setEnabled( true );
    ui.pwfsfit_label->setEnabled( true );

    ui.modwfs_fsm->onConnect();
    ui.modwfs_freq->onConnect();
    ui.modwfs_rad->onConnect();
    ui.modwfs_ch1->onConnect();
    ui.modwfs_ch2->onConnect();

    ui.tip_alignment_label->setEnabled( true );
    ui.pwfsacq_what->setEnabled( true );

    ui.picoscix_label->setEnabled( true );
    ui.picoscix_pos->onConnect();
    ui.picoscix_l->setEnabled( true );
    ui.picoscix_scale->setEnabled( true );
    ui.picoscix_r->setEnabled( true );
    ui.picoscix_combo_label->setEnabled( true );
    ui.picoscix_combo->setEnabled( true );
    ui.picoscix_go->setEnabled( true );

    ui.tweeter_label->setEnabled( true );

    ui.ttmpupil_label->setEnabled( true );
    ui.ttmpupil_fsm->onConnect();
    ui.ttmpupil_ch1->onConnect();
    ui.ttmpupil_ch2->onConnect();

    ui.ncpc_label->setEnabled( false );

    ui.ttmperi_label->setEnabled( true );
    ui.ttmperi_fsm->onConnect();
    ui.ttmperi_ch1->onConnect();
    ui.ttmperi_ch2->onConnect();

    ui.pwfs_coords_label->setEnabled( true );

    /* Camera Lens */
    ui.camlens_label->setEnabled( true );
    ui.camlens_fsm_x_label->setEnabled( true );
    ui.camlens_fsm_y_label->setEnabled( true );

    ui.camlens_fsm_x->onConnect();
    ui.camlens_fsm_y->onConnect();
    ui.camlens_x->onConnect();
    ui.camlens_y->onConnect();

    ui.pwfsfit_threshold->onConnect();
    ui.pwfsfit_avgtime->onConnect();

    ui.puptrack_loop_deltaX->onConnect();
    ui.puptrack_loop_deltaY->onConnect();

    ui.puptrack_loop_slider->onConnect();
    ui.puptrack_loop_gain->onConnect();

    ui.actalign_loop_deltaX->onConnect();
    ui.actalign_loop_deltaY->onConnect();

    ui.actalign_loop_slider->onConnect();
    ui.actalign_loop_gain->onConnect();

    ui.actalign_sensor_slider->onConnect();
    ui.actalign_sensor_nAverage->onConnect();
    ui.actalign_sensor_nImages->onConnect();
    ui.actalign_sensor_pokeAmp->onConnect();

    camwfs_align_setEnabled( true, true );
    twAlign_camwfs_ctrl_setEnabled( true, true );
    twAlign_camwfs_wfs_setEnabled( true, true );
    alignment_buttons_setEnabled( true, true );

    setWindowTitle( "Alignment" );
}

void pupilGuide::onDisconnect()
{
    ui.target_catobj->onDisconnect();
    ui.target_name->onDisconnect();

    m_modwfsFsmState = "";

    ui.modwfs_label->setEnabled( false );
    ui.modwfs_fsm->onDisconnect();
    ui.modwfs_freq->onDisconnect();
    ui.modwfs_rad->onDisconnect();
    ui.modwfs_ch1->onDisconnect();
    ui.modwfs_ch2->onDisconnect();

    ui.tip_alignment_label->setEnabled( false );
    ui.pwfsacq_what->setEnabled( false );

    ui.picoscix_label->setEnabled( false );
    ui.picoscix_pos->onDisconnect();
    ui.picoscix_l->setEnabled( false );
    ui.picoscix_scale->setEnabled( false );
    ui.picoscix_r->setEnabled( false );
    ui.picoscix_combo_label->setEnabled( false );
    ui.picoscix_combo->setEnabled( false );
    ui.picoscix_go->setEnabled( false );

    ui.tweeter_label->setEnabled( false );

    m_pupFsmState = "";
    ui.ttmpupil_label->setEnabled( false );
    ui.ttmpupil_fsm->onDisconnect();
    ui.ttmpupil_ch1->onDisconnect();
    ui.ttmpupil_ch2->onDisconnect();

    ui.ncpc_label->setEnabled( false );

    ui.ttmperi_label->setEnabled( false );
    ui.ttmperi_fsm->onDisconnect();
    ui.ttmperi_ch1->onDisconnect();
    ui.ttmperi_ch2->onDisconnect();

    m_camlensxFsmState = "";
    m_camlensyFsmState = "";
    m_camwfsavgState   = "";
    m_camwfsfitState   = "";

    ui.pwfsfit_label->setEnabled( false );

    ui.pwfs_coords_label->setEnabled( false );

    /* Camera Lens */
    ui.camlens_label->setEnabled( false );
    ui.camlens_fsm_x_label->setEnabled( false );
    ui.camlens_fsm_y_label->setEnabled( false );

    ui.camlens_fsm_x->onDisconnect();
    ui.camlens_fsm_y->onDisconnect();
    ui.camlens_x->onDisconnect();
    ui.camlens_y->onDisconnect();
    camlensSetEnabled( false );

    ui.pwfsfit_threshold->onDisconnect();
    ui.pwfsfit_avgtime->onDisconnect();

    ui.puptrack_loop_deltaX->onDisconnect();
    ui.puptrack_loop_deltaY->onDisconnect();

    ui.puptrack_loop_slider->onDisconnect();
    ui.puptrack_loop_gain->onDisconnect();

    ui.actalign_loop_deltaX->onDisconnect();
    ui.actalign_loop_deltaY->onDisconnect();

    ui.actalign_loop_slider->onDisconnect();
    ui.actalign_loop_gain->onDisconnect();

    ui.actalign_sensor_slider->onDisconnect();
    ui.actalign_sensor_nAverage->onDisconnect();
    ui.actalign_sensor_nImages->onDisconnect();
    ui.actalign_sensor_pokeAmp->onDisconnect();

    camwfs_align_setEnabled( false, true );
    m_camwfs_align_fsmState = "";
    twAlign_camwfs_ctrl_setEnabled( false, true );
    m_twAlign_camwfs_ctrl_fsmState = "";
    twAlign_camwfs_wfs_setEnabled( false, true );
    m_twAlign_camwfs_wfs_fsmState = "";
    alignment_buttons_setEnabled( false, true );

    setWindowTitle( "Alignment (disconnected)" );
}

void pupilGuide::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    return handleSetProperty( ipRecv );
}

void pupilGuide::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
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
                m_modwfsFsmState = ipRecv["state"].get<std::string>();
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
                m_tcsiFsmState = ipRecv["state"].get<std::string>();
            }
        }
        else if( ipRecv.getName() == "labMode" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
                {
                    m_tcsiLabMode = true;
                }
                else
                {
                    m_tcsiLabMode = false;
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
                m_dmWooferFsmState = ipRecv["state"].get<std::string>();
            }
        }
    }
    else if( dev == "wooferModes" )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_wooferModesFsmState = ipRecv["state"].get<std::string>();
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
                m_focus = ipRecv["0002"].get<double>();
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
                m_pwfs_median_1 = ipRecv["med"].get<double>();
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
                m_pwfs_median_2 = ipRecv["med"].get<double>();
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
                m_pwfs_median_3 = ipRecv["med"].get<double>();
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
                m_pwfs_median_4 = ipRecv["med"].get<double>();
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
                m_ttmpupil_ch1 = ipRecv["current"].get<double>();
            }
        }
        else if( ipRecv.getName() == "pos_2" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_ttmpupil_ch2 = ipRecv["current"].get<double>();
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
                m_ttmperi_ch1 = ipRecv["current"].get<double>();
            }
        }
        else if( ipRecv.getName() == "axis2_voltage" )
        {
            if( ipRecv.find( "current" ) )
            {
                m_ttmperi_ch2 = ipRecv["current"].get<double>();
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
        else if( ipRecv.getName() == "loop_state" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
                {
                    m_camwfsAlignLoopState = true;
                }
                else
                {
                    m_camwfsAlignLoopState = false;
                }
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
        else if( ipRecv.getName() == "loop_state" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
                {
                    m_twAlignLoopState = true;
                }
                else
                {
                    m_twAlignLoopState = false;
                }
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
        else if( ipRecv.getName() == "loop_state" )
        {
            if( ipRecv.find( "toggle" ) )
            {
                if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
                {
                    m_twAlignSensorState = true;
                }
                else
                {
                    m_twAlignSensorState = false;
                }
            }
        }
    }
    return;
} // handleSetProperty

void pupilGuide::on_target_sync_pressed()
{
    std::cerr << "on_target_sync_pressed()\n";
}

void pupilGuide::on_telacq_acqfromguider_pressed()
{
    std::cerr << "on_telacq_acqfromguider_pressed()\n";
}

void pupilGuide::on_telacq_acqfocus_pressed()
{
    std::cerr << "on_telacq_acqfocus_pressed()\n";
}

void pupilGuide::on_camacq_acqstar_activated( int index )
{
    static_cast<void>( index );

    std::cerr << "on_camacq_acqstar_activated( int index )\n";
}
void pupilGuide::on_camacq_acq_go_pressed()
{
    std::cerr << "on_camacq_acq_go_pressed()\n";
}
void pupilGuide::on_camacq_seestar_activated( int index )
{
    static_cast<void>( index );

    std::cerr << "on_camacq_seestar_activated( int index )\n";
}
void pupilGuide::on_camacq_see_go_pressed()
{
    std::cerr << "on_camacq_see_go_pressed()\n";
}

void pupilGuide::on_camacq_acq_restart_pressed()
{
    std::cerr << "on_camacq_acq_restart_pressed()\n";
}

void pupilGuide::on_camacq_see_measure_pressed()
{
    std::cerr << "on_camacq_see_measure_pressed()\n";
}

void pupilGuide::on_modwfs_rest_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "modwfs" );
    ip.setName( "modState" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = 1;

    sendNewProperty( ip );
}

void pupilGuide::on_modwfs_set_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "modwfs" );
    ip.setName( "modState" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = 3;
    sendNewProperty( ip );
}

void pupilGuide::on_modwfs_modulate_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "modwfs" );
    ip.setName( "modState" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = 4;

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsacq_what_pressed()
{
    if( m_tipmovewhat == MOVE_TTM )
    {
        m_tipmovewhat = MOVE_WOOF;
        ui.pwfsacq_what->setText( "move woofer" );
    }
    else if( m_tipmovewhat == MOVE_WOOF && !m_tcsiLabMode )
    {
        m_tipmovewhat = MOVE_TEL;
        ui.pwfsacq_what->setText( "move telescope" );
    }
    else
    {
        m_tipmovewhat = MOVE_TTM;
        ui.pwfsacq_what->setText( "move ttm" );
    }
}

void pupilGuide::on_pwfsacq_u_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = m_pwfsacqScale;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = 0;
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, 0, m_pwfsacqScale );

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
        ip["y"] = m_pwfsacqScale * 5.;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = 0;
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsacq_ul_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = m_pwfsacqScale / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = -m_pwfsacqScale / sqrt( 2. );
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, m_pwfsacqScale / sqrt( 2. ), m_pwfsacqScale / sqrt( 2. ) );

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
        ip["y"] = m_pwfsacqScale * 5. / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_pwfsacqScale * 5. / sqrt( 2. );
    }

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsacq_l_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = 0;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = -m_pwfsacqScale;
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, m_pwfsacqScale, 0 );

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
        ip["x"] = -m_pwfsacqScale * 5.;
    }

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsacq_dl_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = -m_pwfsacqScale / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = -m_pwfsacqScale / sqrt( 2. );
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, m_pwfsacqScale / sqrt( 2. ), -m_pwfsacqScale / sqrt( 2. ) );

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
        ip["y"] = -m_pwfsacqScale * 5. / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = -m_pwfsacqScale * 5. / sqrt( 2. );
    }

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsacq_d_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = -m_pwfsacqScale;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = 0;
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, 0, -m_pwfsacqScale );

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
        ip["y"] = -m_pwfsacqScale * 5.;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = 0;
    }

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsacq_dr_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = -m_pwfsacqScale / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_pwfsacqScale / sqrt( 2. );
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, -m_pwfsacqScale / sqrt( 2. ), -m_pwfsacqScale / sqrt( 2. ) );

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
        ip["y"] = -m_pwfsacqScale * 5. / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_pwfsacqScale * 5. / sqrt( 2. );
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsacq_r_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = 0;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_pwfsacqScale;
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, -m_pwfsacqScale, 0 );

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
        ip["x"] = m_pwfsacqScale * 5.;
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsacq_ur_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_TTM )
    {
        ip.setDevice( "modwfs" );
        ip.setName( "offset" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = m_pwfsacqScale / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_pwfsacqScale / sqrt( 2. );
    }
    else if( m_tipmovewhat == MOVE_WOOF )
    {
        double tip, tilt;
        wooferTipTilt( tip, tilt, -m_pwfsacqScale / sqrt( 2. ), m_pwfsacqScale / sqrt( 2. ) );

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
        ip["y"] = m_pwfsacqScale * 5. / sqrt( 2. );
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = m_pwfsacqScale * 5. / sqrt( 2. );
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsacq_scale_pressed()
{
    if( ( (int)( 100 * m_pwfsacqScale ) ) == 100 )
    {
        m_pwfsacqScale = 0.5;
    }
    else if( ( (int)( 100 * m_pwfsacqScale ) ) == 50 )
    {
        m_pwfsacqScale = 0.1;
    }
    else if( ( (int)( 100 * m_pwfsacqScale ) ) == 10 )
    {
        m_pwfsacqScale = 0.05;
    }
    else if( ( (int)( 100 * m_pwfsacqScale ) ) == 5 )
    {
        m_pwfsacqScale = 0.01;
    }
    else if( ( (int)( 100 * m_pwfsacqScale ) ) == 1 )
    {
        m_pwfsacqScale = 1.0;
    }

    char ss[5];
    snprintf( ss, 5, "%0.2f", m_pwfsacqScale );
    ui.pwfsacq_scale->setText( ss );
}

void pupilGuide::on_pwfsfoc_p_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_WOOF )
    {

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0002" ) );
        ip["0002"] = m_focus + m_pwfsacqFocScale * 0.2;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "z" ) );
        ip["z"] = m_pwfsacqScale * 100.;
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsfoc_m_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_WOOF )
    {

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0002" ) );
        ip["0002"] = m_focus - m_pwfsacqFocScale * 0.2;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "z" ) );
        ip["z"] = -m_pwfsacqScale * 100.;
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_pwfsfoc_scale_pressed()
{
    if( ( (int)( 100 * m_pwfsacqFocScale ) ) == 100 )
    {
        m_pwfsacqFocScale = 0.5;
    }
    else if( ( (int)( 100 * m_pwfsacqFocScale ) ) == 50 )
    {
        m_pwfsacqFocScale = 0.1;
    }
    else if( ( (int)( 100 * m_pwfsacqFocScale ) ) == 10 )
    {
        m_pwfsacqFocScale = 0.05;
    }
    else if( ( (int)( 100 * m_pwfsacqFocScale ) ) == 5 )
    {
        m_pwfsacqFocScale = 0.01;
    }
    else if( ( (int)( 100 * m_pwfsacqFocScale ) ) == 1 )
    {
        m_pwfsacqFocScale = 1.0;
    }

    char ss[5];
    snprintf( ss, 5, "%0.2f", m_pwfsacqFocScale );
    ui.pwfsfoc_scale->setText( ss );
}

// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^//

void pupilGuide::modGUISetEnable( bool enableModGUI, bool enableModArrows )
{
    if( enableModGUI )
    {
        ui.modwfs_label->setEnabled( true );
        ui.modwfs_fsm->setEnabled( true );
        ui.modwfs_fsm->setEnabled( true );
        if( m_modState == 3 || m_modState == 4 )
        {
            ui.modwfs_freq_label->setEnabled( true );
            ui.modwfs_freq->setEnabled( true );
            ui.modwfs_rad_label->setEnabled( true );
            ui.modwfs_rad->setEnabled( true );

            ui.modwfs_ch1->setEnabled( true );
            ui.modwfs_ch2->setEnabled( true );
        }
        else
        {
            ui.modwfs_freq_label->setEnabled( false );
            ui.modwfs_freq->setEnabled( false );
            ui.modwfs_rad_label->setEnabled( false );
            ui.modwfs_rad->setEnabled( false );

            ui.modwfs_ch1->setEnabled( false );
            ui.modwfs_ch2->setEnabled( false );
        }

        if( enableModArrows )
        {
            ui.pwfsacq_ul->setEnabled( true );
            ui.pwfsacq_u->setEnabled( true );
            ui.pwfsacq_ur->setEnabled( true );
            ui.pwfsacq_l->setEnabled( true );
            ui.pwfsacq_scale->setEnabled( true );
            ui.pwfsacq_r->setEnabled( true );
            ui.pwfsacq_dl->setEnabled( true );
            ui.pwfsacq_d->setEnabled( true );
            ui.pwfsacq_dr->setEnabled( true );

            if( m_tipmovewhat == MOVE_TEL || m_tipmovewhat == MOVE_WOOF )
            {
                ui.pwfsfoc_p->setEnabled( true );
                ui.pwfsfoc_scale->setEnabled( true );
                ui.pwfsfoc_m->setEnabled( true );
            }
            else
            {
                ui.pwfsfoc_p->setEnabled( false );
                ui.pwfsfoc_scale->setEnabled( false );
                ui.pwfsfoc_m->setEnabled( false );
            }
        }
        else
        {
            ui.pwfsacq_ul->setEnabled( false );
            ui.pwfsacq_u->setEnabled( false );
            ui.pwfsacq_ur->setEnabled( false );
            ui.pwfsacq_l->setEnabled( false );
            ui.pwfsacq_scale->setEnabled( false );
            ui.pwfsacq_r->setEnabled( false );
            ui.pwfsacq_dl->setEnabled( false );
            ui.pwfsacq_d->setEnabled( false );
            ui.pwfsacq_dr->setEnabled( false );

            ui.pwfsfoc_p->setEnabled( false );
            ui.pwfsfoc_scale->setEnabled( false );
            ui.pwfsfoc_m->setEnabled( false );
        }
    }
    else
    {
        if( m_modwfsFsmState != "POWEROFF" && m_modwfsFsmState != "CONFIGURING" )
        {
            ui.modwfs_label->setEnabled( false );
        }
        else
        {
            ui.modwfs_label->setEnabled( true );
        }
        ui.modwfs_fsm->setEnabled( false );
        ui.modwfs_freq_label->setEnabled( false );
        ui.modwfs_freq->setEnabled( false );
        ui.modwfs_rad_label->setEnabled( false );
        ui.modwfs_rad->setEnabled( false );
        ui.modwfs_rest->setEnabled( false );
        ui.modwfs_set->setEnabled( false );
        ui.modwfs_modulate->setEnabled( false );
        ui.modwfs_ch1->setEnabled( false );
        ui.modwfs_ch2->setEnabled( false );

        if( !enableModArrows )
        {
            ui.pwfsacq_ul->setEnabled( false );
            ui.pwfsacq_u->setEnabled( false );
            ui.pwfsacq_ur->setEnabled( false );
            ui.pwfsacq_l->setEnabled( false );
            ui.pwfsacq_scale->setEnabled( false );
            ui.pwfsacq_r->setEnabled( false );
            ui.pwfsacq_dl->setEnabled( false );
            ui.pwfsacq_d->setEnabled( false );
            ui.pwfsacq_dr->setEnabled( false );

            ui.pwfsfoc_p->setEnabled( false );
            ui.pwfsfoc_scale->setEnabled( false );
            ui.pwfsfoc_m->setEnabled( false );
        }
        else
        {
            ui.pwfsacq_ul->setEnabled( true );
            ui.pwfsacq_u->setEnabled( true );
            ui.pwfsacq_ur->setEnabled( true );
            ui.pwfsacq_l->setEnabled( true );
            ui.pwfsacq_scale->setEnabled( true );
            ui.pwfsacq_r->setEnabled( true );
            ui.pwfsacq_dl->setEnabled( true );
            ui.pwfsacq_d->setEnabled( true );
            ui.pwfsacq_dr->setEnabled( true );

            if( m_tipmovewhat != MOVE_TTM )
            {
                ui.pwfsfoc_p->setEnabled( true );
                ui.pwfsfoc_scale->setEnabled( true );
                ui.pwfsfoc_m->setEnabled( true );
            }
        }
    }
}

void pupilGuide::camwfsfitSetEnabled( bool enabled )
{
    ui.pwfs_medians_label->setEnabled( enabled );
    ui.pwfs_median_1->setEnabled( enabled );
    ui.pwfs_median_2->setEnabled( enabled );
    ui.pwfs_median_3->setEnabled( enabled );
    ui.pwfs_median_4->setEnabled( enabled );
    ui.pwfs_medians_delta->setEnabled( enabled );
    ui.pwfsfit_threshold->setEnabled( enabled );

    if( enabled == false )
    {
        ui.pwfs_median_1->setText( "" );
        ui.pwfs_median_2->setText( "" );
        ui.pwfs_median_3->setText( "" );
        ui.pwfs_median_4->setText( "" );
    }

    ui.pwfs_LL_D->setEnabled( enabled );
    ui.pwfs_LR_D->setEnabled( enabled );
    ui.pwfs_UL_D->setEnabled( enabled );
    ui.pwfs_UR_D->setEnabled( enabled );
    ui.pwfs_LL_x->setEnabled( enabled );
    ui.pwfs_LR_x->setEnabled( enabled );
    ui.pwfs_UL_x->setEnabled( enabled );
    ui.pwfs_UR_x->setEnabled( enabled );
    ui.pwfs_LL_y->setEnabled( enabled );
    ui.pwfs_LR_y->setEnabled( enabled );
    ui.pwfs_UL_y->setEnabled( enabled );
    ui.pwfs_UR_y->setEnabled( enabled );
    ui.pwfs_avg_D->setEnabled( enabled );
    ui.pwfs_avg_x->setEnabled( enabled );
    ui.pwfs_avg_y->setEnabled( enabled );
    ui.pwfs_coords_delta->setEnabled( enabled );
    ui.pwfs_x_label->setEnabled( enabled );
    ui.pwfs_y_label->setEnabled( enabled );
    ui.pwfs_D_label->setEnabled( enabled );
    ui.pwfs_UR_label->setEnabled( enabled );
    ui.pwfs_UL_label->setEnabled( enabled );
    ui.pwfs_LR_label->setEnabled( enabled );
    ui.pwfs_LL_label->setEnabled( enabled );
    ui.pwfs_avg_label->setEnabled( enabled );
}

void pupilGuide::camlensSetEnabled( bool enabled, int whichcl )
{
    if( whichcl == CAMLENS_BOTH )
    {
        ui.camlens_scale->setEnabled( enabled );
    }
    else
    {
        ui.camlens_scale->setEnabled( true );
    }

    if( whichcl == CAMLENS_X || whichcl == CAMLENS_BOTH )
    {
        ui.camlens_fsm_x->setEnabled( enabled );
        ui.camlens_x->setEnabled( enabled );
        ui.camlens_l->setEnabled( enabled );
        ui.camlens_r->setEnabled( enabled );
    }

    if( whichcl == CAMLENS_Y || whichcl == CAMLENS_BOTH )
    {
        ui.camlens_fsm_y->setEnabled( enabled );
        ui.camlens_y->setEnabled( enabled );
        ui.camlens_u->setEnabled( enabled );
        ui.camlens_d->setEnabled( enabled );
    }
}

void pupilGuide::camwfs_align_setEnabled( bool enabled, bool all )
{
    if( all )
    {
        ui.puptrack_loop_label->setEnabled( enabled );
    }
    ui.puptrack_loop_deltaX->setEnabled( enabled );
    ui.puptrack_loop_deltaY->setEnabled( enabled );
    ui.puptrack_loop_slider->setEnabled( enabled );
    ui.puptrack_loop_gain->setEnabled( enabled );
}

void pupilGuide::twAlign_camwfs_ctrl_setEnabled( bool enabled, bool all )
{
    if( all )
    {
        ui.actalign_loop_label->setEnabled( enabled );
    }
    ui.actalign_loop_deltaX->setEnabled( enabled );
    ui.actalign_loop_deltaY->setEnabled( enabled );
    ui.actalign_loop_slider->setEnabled( enabled );
    ui.actalign_loop_gain->setEnabled( enabled );
}

void pupilGuide::twAlign_camwfs_wfs_setEnabled( bool enabled, bool all )
{
    if( all )
    {
        ui.actalign_sensor_label->setEnabled( enabled );
    }
    ui.actalign_sensor_slider->setEnabled( enabled );
    ui.actalign_sensor_nAverage->setEnabled( enabled );
    ui.actalign_sensor_nImages->setEnabled( enabled );
    ui.actalign_sensor_pokeAmp->setEnabled( enabled );
}

void pupilGuide::alignment_buttons_setEnabled( bool enabled, bool all )
{
    if( all )
    {
        ui.autoalign_label->setEnabled( enabled );
    }
    ui.autoalign_slider->setEnabled( enabled );
    //    ui.button_startAlignment->setEnabled( enabled );
    //    ui.button_stopAlignment->setEnabled( enabled );
}

void pupilGuide::updateGUI()
{

    //--------- Modulation

    bool enableModGUI    = true;
    bool enableModArrows = true;

    char str[16];
    if( m_modwfsFsmState == "NOTHOMED" )
    {
        if( m_tipmovewhat == MOVE_TTM )
        {
            enableModArrows = false;
        }
    }
    else if( ( m_modwfsFsmState != "READY" ) && ( m_modwfsFsmState != "OPERATING" ) )
    {
        enableModGUI = false;
        if( m_tipmovewhat == MOVE_TTM )
        {
            enableModArrows = false;
        }
    }

    // If moving woofer and either woofer or wooferModes aren't ready we disable the arrows
    if( m_tipmovewhat == MOVE_WOOF && ( m_dmWooferFsmState != "OPERATING" || m_wooferModesFsmState != "READY" ) )
    {
        enableModArrows = false;
    }

    // If moving telescope and tcsi isn't connected we disable the arrows
    if( m_tipmovewhat == MOVE_TEL && ( m_tcsiFsmState != "CONNECTED" ) )
    {
        enableModArrows = false;
    }

    modGUISetEnable( enableModGUI, enableModArrows );

    if( m_modState == 3 && enableModGUI )
    {
        ui.modwfs_rest->setEnabled( true );
        ui.modwfs_set->setEnabled( false );
        ui.modwfs_modulate->setEnabled( true );
    }
    else if( m_modState == 4 && enableModGUI )
    {
        ui.modwfs_rest->setEnabled( true );
        ui.modwfs_set->setEnabled( true );
        ui.modwfs_modulate->setEnabled( true );
    }
    else
    {
        if( enableModGUI )
        {
            ui.modwfs_rest->setEnabled( true );
            ui.modwfs_set->setEnabled( true );
            ui.modwfs_modulate->setEnabled( false );
        }
    }

    ui.modwfs_fsm->updateGUI();
    ui.modwfs_freq->updateGUI();
    ui.modwfs_rad->updateGUI();
    ui.modwfs_ch1->updateGUI();
    ui.modwfs_ch2->updateGUI();

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

        if( ui.pwfs_medians_delta->checkState() == Qt::Checked )
        {
            double ave = 0.25 * ( m_pwfs_median_1 + m_pwfs_median_2 + m_pwfs_median_3 + m_pwfs_median_4 );
            m1         = m_pwfs_median_1 - ave;
            m2         = m_pwfs_median_2 - ave;
            m3         = m_pwfs_median_3 - ave;
            m4         = m_pwfs_median_4 - ave;
        }
        else
        {
            m1 = m_pwfs_median_1;
            m2 = m_pwfs_median_2;
            m3 = m_pwfs_median_3;
            m4 = m_pwfs_median_4;
        }

        snprintf( str, 16, "%0.1f", m1 );
        ui.pwfs_median_1->setText( str );

        snprintf( str, 16, "%0.1f", m2 );
        ui.pwfs_median_2->setText( str );

        snprintf( str, 16, "%0.1f", m3 );
        ui.pwfs_median_3->setText( str );

        snprintf( str, 16, "%0.1f", m4 );
        ui.pwfs_median_4->setText( str );

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

        if( ui.pwfs_coords_delta->checkState() == Qt::Checked )
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
        ui.pwfs_LL_D->setText( str );

        snprintf( str, 16, "%0.2f", D2 );
        ui.pwfs_LR_D->setText( str );

        snprintf( str, 16, "%0.2f", D3 );
        ui.pwfs_UL_D->setText( str );

        snprintf( str, 16, "%0.2f", D4 );
        ui.pwfs_UR_D->setText( str );

        snprintf( str, 16, "%0.2f", x1 );
        ui.pwfs_LL_x->setText( str );

        snprintf( str, 16, "%0.2f", x2 );
        ui.pwfs_LR_x->setText( str );

        snprintf( str, 16, "%0.2f", x3 );
        ui.pwfs_UL_x->setText( str );

        snprintf( str, 16, "%0.2f", x4 );
        ui.pwfs_UR_x->setText( str );

        snprintf( str, 16, "%0.2f", y1 );
        ui.pwfs_LL_y->setText( str );

        snprintf( str, 16, "%0.2f", y2 );
        ui.pwfs_LR_y->setText( str );

        snprintf( str, 16, "%0.2f", y3 );
        ui.pwfs_UL_y->setText( str );

        snprintf( str, 16, "%0.2f", y4 );
        ui.pwfs_UR_y->setText( str );

        snprintf( str, 16, "%0.2f", 0.25 * ( D1 + D2 + D3 + D4 ) );
        ui.pwfs_avg_D->setText( str );

        snprintf( str, 16, "%0.2f", 0.25 * ( x1 + x2 + x3 + x4 ) );
        ui.pwfs_avg_x->setText( str );

        snprintf( str, 16, "%0.2f", 0.25 * ( y1 + y2 + y3 + y4 ) );
        ui.pwfs_avg_y->setText( str );
    }

    // ------ camwfs averaging
    if( m_camwfsavgState == "READY" || m_camwfsavgState == "OPERATING" )
    {
        ui.pwfsfit_avgtime->setEnabled( true );
    }
    else
    {
        ui.pwfsfit_avgtime->setEnabled( false );
    }

    // ------ dmtweeter

    if( m_dmtweeterState == "READY" || m_dmtweeterState == "OPERATING" )
    {
        ui.tweeter_set->setEnabled( true );
        if( m_dmtweeterTestSet )
        {
            ui.tweeter_set->setText( "zero test" );
        }
        else
        {
            ui.tweeter_set->setText( "set test" );
        }
    }
    else
    {
        ui.tweeter_set->setEnabled( false );
        ui.tweeter_set->setText( "set test" );
    }

    // ------ dmncpc

    if( m_dmncpcState == "READY" || m_dmncpcState == "OPERATING" )
    {
        ui.ncpc_set->setEnabled( true );

        if( m_dmncpcTestSet )
        {
            ui.ncpc_set->setText( "zero test" );
        }
        else
        {
            ui.ncpc_set->setText( "set test" );
        }
    }
    else
    {
        ui.ncpc_set->setEnabled( false );
        ui.ncpc_set->setText( "set test" );
    }

    // ------ Pupil Steering
    bool enablePupFSM       = true;
    bool enablePupFSMArrows = true;

    if( m_pupFsmState == "READY" )
    {
        ui.ttmpupil_fsm->setEnabled( true );
        ui.ttmpupil_ch1->setEnabled( true );
        ui.ttmpupil_ch2->setEnabled( true );
        ui.ttmpupil_set->setEnabled( false );
        ui.ttmpupil_rest->setEnabled( true );
    }
    else if( m_pupFsmState == "NOTHOMED" )
    {
        ui.ttmpupil_fsm->setEnabled( true );
        ui.ttmpupil_ch1->setEnabled( false );
        ui.ttmpupil_ch2->setEnabled( false );
        ui.ttmpupil_set->setEnabled( true );
        ui.ttmpupil_rest->setEnabled( false );
        enablePupFSMArrows = false;
    }
    else if( m_pupFsmState == "HOMING" )
    {
        ui.ttmpupil_fsm->setEnabled( true );
        ui.ttmpupil_ch1->setEnabled( false );
        ui.ttmpupil_ch2->setEnabled( false );
        ui.ttmpupil_set->setEnabled( false );
        ui.ttmpupil_rest->setEnabled( true );
        enablePupFSMArrows = false;
    }
    else
    {
        enablePupFSM = false;
        if( m_pupFsmState == "" )
        {
            ui.ttmpupil_fsm->setEnabled( false );
        }
        else
        {
            ui.ttmpupil_fsm->setEnabled( true );
        }
    }

    if( enablePupFSM )
    {
        if( enablePupFSMArrows )
        {
            ui.ttmpupil_ul->setEnabled( true );
            ui.ttmpupil_ur->setEnabled( true );
            ui.ttmpupil_scale->setEnabled( true );
            ui.ttmpupil_dl->setEnabled( true );
            ui.ttmpupil_dr->setEnabled( true );
        }
        else
        {
            ui.ttmpupil_ul->setEnabled( false );
            ui.ttmpupil_ur->setEnabled( false );
            ui.ttmpupil_scale->setEnabled( false );
            ui.ttmpupil_dl->setEnabled( false );
            ui.ttmpupil_dr->setEnabled( false );
        }
    }
    else
    {

        ui.ttmpupil_set->setEnabled( false );
        ui.ttmpupil_rest->setEnabled( false );
        ui.ttmpupil_ch1->setEnabled( false );
        ui.ttmpupil_ch2->setEnabled( false );

        ui.ttmpupil_ul->setEnabled( false );
        ui.ttmpupil_ur->setEnabled( false );
        ui.ttmpupil_scale->setEnabled( false );
        ui.ttmpupil_dl->setEnabled( false );
        ui.ttmpupil_dr->setEnabled( false );
    }

    // ------ TTM Peri
    bool enableTTMPeriFSM       = true;
    bool enableTTMPeriFSMArrows = true;

    if( m_ttmPeriFsmState == "READY" )
    {
        ui.ttmperi_fsm->setEnabled( true );
        ui.ttmperi_ch1->setEnabled( false );
        ui.ttmperi_ch2->setEnabled( false );
        ui.ttmperi_set->setEnabled( true );
        ui.ttmperi_rest->setEnabled( false );

        enableTTMPeriFSMArrows = false;
    }
    else if( m_ttmPeriFsmState == "OPERATING" )
    {
        ui.ttmperi_fsm->setEnabled( true );
        ui.ttmperi_ch1->setEnabled( true );
        ui.ttmperi_ch2->setEnabled( true );
        ui.ttmperi_set->setEnabled( false );
        ui.ttmperi_rest->setEnabled( true );
        enableTTMPeriFSMArrows = true;
    }
    else
    {
        enableTTMPeriFSM = false;

        if( m_ttmPeriFsmState == "" )
        {
            ui.ttmperi_fsm->setEnabled( false );
        }
        else
        {
            ui.ttmperi_fsm->setEnabled( true );
        }

        ui.ttmperi_ch1->setEnabled( false );
        ui.ttmperi_ch2->setEnabled( false );
        ui.ttmperi_set->setEnabled( false );
        ui.ttmperi_rest->setEnabled( false );
    }

    if( enableTTMPeriFSM )
    {
        if( enableTTMPeriFSMArrows )
        {
            ui.ttmperi_l->setEnabled( true );
            ui.ttmperi_r->setEnabled( true );
            ui.ttmperi_scale->setEnabled( true );
            ui.ttmperi_u->setEnabled( true );
            ui.ttmperi_d->setEnabled( true );
        }
        else
        {
            ui.ttmperi_l->setEnabled( false );
            ui.ttmperi_r->setEnabled( false );
            ui.ttmperi_scale->setEnabled( false );
            ui.ttmperi_u->setEnabled( false );
            ui.ttmperi_d->setEnabled( false );
        }
    }
    else
    {
        ui.ttmperi_l->setEnabled( false );
        ui.ttmperi_r->setEnabled( false );
        ui.ttmperi_scale->setEnabled( false );
        ui.ttmperi_u->setEnabled( false );
        ui.ttmperi_d->setEnabled( false );
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
        ui.camlens_y->onDisconnect();
    }
    else if( !( m_camlensxFsmState == "READY" || m_camlensxFsmState == "OPERATING" ) &&
             ( m_camlensyFsmState == "READY" || m_camlensyFsmState == "OPERATING" ) )
    {
        camlensSetEnabled( false, CAMLENS_X );
        ui.camlens_x->onDisconnect();

        camlensSetEnabled( true, CAMLENS_Y );
    }
    else
    {
        camlensSetEnabled( false );
        ui.camlens_x->onDisconnect();
        ui.camlens_y->onDisconnect();
    }

    if( m_camlensxFsmState == "SHUTDOWN" )
    {
        ui.camlens_x->onDisconnect();
    }

    if( m_camlensyFsmState == "SHUTDOWN" )
    {
        ui.camlens_y->onDisconnect();
    }

    ui.camlens_fsm_x->updateGUI();
    ui.camlens_fsm_y->updateGUI();
    ui.camlens_x->updateGUI();
    ui.camlens_y->updateGUI();

    ui.pwfsfit_threshold->updateGUI();
    ui.pwfsfit_avgtime->updateGUI();

    if( m_camwfs_align_fsmState != "READY" && m_camwfs_align_fsmState != "OPERATING" )
    {
        camwfs_align_setEnabled( false, false );
    }
    else
    {
        camwfs_align_setEnabled( true, true );
    }

    ui.puptrack_loop_deltaX->updateGUI();
    ui.puptrack_loop_deltaY->updateGUI();
    ui.puptrack_loop_slider->updateGUI();
    ui.puptrack_loop_gain->updateGUI();

    if( m_twAlign_camwfs_ctrl_fsmState != "READY" && m_twAlign_camwfs_ctrl_fsmState != "OPERATING" )
    {
        twAlign_camwfs_ctrl_setEnabled( false, false );
    }
    else
    {
        twAlign_camwfs_ctrl_setEnabled( true, true );
    }

    ui.actalign_loop_deltaX->updateGUI();
    ui.actalign_loop_deltaY->updateGUI();
    ui.actalign_loop_slider->updateGUI();
    ui.actalign_loop_gain->updateGUI();

    if( m_twAlign_camwfs_wfs_fsmState != "READY" && m_twAlign_camwfs_wfs_fsmState != "OPERATING" )
    {
        twAlign_camwfs_wfs_setEnabled( false, false );
    }
    else
    {
        twAlign_camwfs_wfs_setEnabled( true, true );
    }

    ui.actalign_sensor_slider->updateGUI();
    ui.actalign_sensor_nAverage->updateGUI();
    ui.actalign_sensor_nImages->updateGUI();
    ui.actalign_sensor_pokeAmp->updateGUI();

} // updateGUI()

// ------------- modttm

//----------- picoscix

void pupilGuide::move_picoscix( int delta )
{
    if( m_picoState != "READY" || m_picoscixPos < -1000000 )
    {
        return;
    }

    int newpos = m_picoscixPos + delta;

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

void pupilGuide::on_tweeter_set_pressed()
{
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

void pupilGuide::on_ncpc_set_pressed()
{
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

void pupilGuide::on_ttmpupil_rest_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmpupil" );
    ip.setName( "releaseDM" );
    ip.add( pcf::IndiElement( "request" ) );
    ip["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ip );
}

void pupilGuide::on_ttmpupil_set_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmpupil" );
    ip.setName( "initDM" );
    ip.add( pcf::IndiElement( "request" ) );
    ip["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ip );
}

void pupilGuide::on_ttmpupil_ul_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmpupil" );
    ip.setName( "pos_1" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = m_ttmpupil_ch1 + m_pupStepSize / sqrt( 2 );

    sendNewProperty( ip );

    pcf::IndiProperty ip2( pcf::IndiProperty::Number );

    ip2.setDevice( "ttmpupil" );
    ip2.setName( "pos_2" );
    ip2.add( pcf::IndiElement( "target" ) );

    ip2["target"] = m_ttmpupil_ch2 + m_pupStepSize / sqrt( 2 );

    sendNewProperty( ip2 );
}

void pupilGuide::on_ttmpupil_dl_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmpupil" );
    ip.setName( "pos_1" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = m_ttmpupil_ch1 - m_pupStepSize / sqrt( 2 );

    sendNewProperty( ip );

    pcf::IndiProperty ip2( pcf::IndiProperty::Number );

    ip2.setDevice( "ttmpupil" );
    ip2.setName( "pos_2" );
    ip2.add( pcf::IndiElement( "target" ) );

    ip2["target"] = m_ttmpupil_ch2 + m_pupStepSize / sqrt( 2 );

    sendNewProperty( ip2 );
}

void pupilGuide::on_ttmpupil_dr_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmpupil" );
    ip.setName( "pos_1" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = m_ttmpupil_ch1 - m_pupStepSize / sqrt( 2 );

    sendNewProperty( ip );

    pcf::IndiProperty ip2( pcf::IndiProperty::Number );

    ip2.setDevice( "ttmpupil" );
    ip2.setName( "pos_2" );
    ip2.add( pcf::IndiElement( "target" ) );

    ip2["target"] = m_ttmpupil_ch2 - m_pupStepSize / sqrt( 2 );

    sendNewProperty( ip2 );
}

void pupilGuide::on_ttmpupil_ur_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmpupil" );
    ip.setName( "pos_1" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = m_ttmpupil_ch1 + m_pupStepSize / sqrt( 2 );

    sendNewProperty( ip );

    pcf::IndiProperty ip2( pcf::IndiProperty::Number );

    ip2.setDevice( "ttmpupil" );
    ip2.setName( "pos_2" );
    ip2.add( pcf::IndiElement( "target" ) );

    ip2["target"] = m_ttmpupil_ch2 - m_pupStepSize / sqrt( 2 );

    sendNewProperty( ip2 );
}

void pupilGuide::on_ttmpupil_scale_pressed()
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
    ui.ttmpupil_scale->setText( ss );
}

void pupilGuide::on_ttmperi_rest_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmperi" );
    ip.setName( "set" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"].setSwitchState( pcf::IndiElement::Off );

    sendNewProperty( ip );
}

void pupilGuide::on_ttmperi_set_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmperi" );
    ip.setName( "set" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ip );
}

void pupilGuide::on_ttmperi_l_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmperi" );
    ip.setName( "axis1_voltage" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = m_ttmperi_ch1 + m_ttmPeriStepSize;

    sendNewProperty( ip );
}

void pupilGuide::on_ttmperi_r_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmperi" );
    ip.setName( "axis1_voltage" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = m_ttmperi_ch1 - m_ttmPeriStepSize;

    sendNewProperty( ip );
}

void pupilGuide::on_ttmperi_u_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmperi" );
    ip.setName( "axis2_voltage" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = m_ttmperi_ch2 + m_ttmPeriStepSize;

    sendNewProperty( ip );
}

void pupilGuide::on_ttmperi_d_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "ttmperi" );
    ip.setName( "axis2_voltage" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = m_ttmperi_ch2 - m_ttmPeriStepSize;

    sendNewProperty( ip );
}

void pupilGuide::on_ttmperi_scale_pressed()
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
    ui.ttmperi_scale->setText( ss );
}

void pupilGuide::toggleExpFit( bool st )
{

    ui.pwfs_D_label->setVisible( st );

    ui.pwfs_UR_label->setVisible( st );
    ui.pwfs_UR_x->setVisible( st );
    ui.pwfs_UR_y->setVisible( st );
    ui.pwfs_UR_D->setVisible( st );

    ui.pwfs_UL_label->setVisible( st );
    ui.pwfs_UL_x->setVisible( st );
    ui.pwfs_UL_y->setVisible( st );
    ui.pwfs_UL_D->setVisible( st );

    ui.pwfs_LR_label->setVisible( st );
    ui.pwfs_LR_x->setVisible( st );
    ui.pwfs_LR_y->setVisible( st );
    ui.pwfs_LR_D->setVisible( st );

    ui.pwfs_LL_label->setVisible( st );
    ui.pwfs_LL_x->setVisible( st );
    ui.pwfs_LL_y->setVisible( st );
    ui.pwfs_LL_D->setVisible( st );

    ui.pwfs_avg_D->setVisible( st );

    if( st )
    {
        ui.pwfs_coords_expand->setIcon( QIcon( ":/icons/keyboard_double_arrow_up.png" ) );
    }
    else
    {
        ui.pwfs_coords_expand->setIcon( QIcon( ":/icons/keyboard_double_arrow_down.png" ) );
    }
}

void pupilGuide::on_pwfs_coords_expand_pressed()
{
    bool st = !ui.pwfs_D_label->isVisible();
    toggleExpFit( st );
}

void pupilGuide::on_camlens_u_pressed()
{
    if( m_camlensyFsmState != "READY" )
        return;

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "stagecamlensy" );
    ip.setName( "position" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = m_camlensy_pos - m_camlensStepSize;

    sendNewProperty( ip );
}

void pupilGuide::on_camlens_l_pressed()
{
    if( m_camlensxFsmState != "READY" )
        return;

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "stagecamlensx" );
    ip.setName( "position" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = m_camlensx_pos - m_camlensStepSize;

    sendNewProperty( ip );
}

void pupilGuide::on_camlens_d_pressed()
{
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

void pupilGuide::on_camlens_r_pressed()
{
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

void pupilGuide::on_camlens_scale_pressed()
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
    ui.camlens_scale->setText( ss );
}
/*
void pupilGuide::on_button_startAlignment_pressed()
{
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
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "twAlign-camwfs-wfs" );
    ip.setName( "continuous" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"] = pcf::IndiElement::Off;

    sendNewProperty( ip );

    ip.setDevice( "twAlign-camwfs-ctrl" );
    ip.setName( "loop_state" );
    ip["toggle"] = pcf::IndiElement::Off;

    sendNewProperty( ip );

    if( m_tcsiLabMode )
    {
        ip.setDevice( "camwfs-align" );
        ip.setName( "loop_state" );
        ip["toggle"] = pcf::IndiElement::Off;

        sendNewProperty( ip );
    }
}*/

} // namespace xqt

#include "moc_pupilGuide.cpp"

#endif
