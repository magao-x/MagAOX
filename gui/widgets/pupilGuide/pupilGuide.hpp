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
    double rot = ( 180. + 29.0 ) * 3.14159 / 180.;
    double scale = -1.0;

    tip = scale * ( x * cos( rot ) - y * sin( rot ) );
    tilt = scale * ( x * sin( rot ) + y * cos( rot ) );
}

class pupilGuide : public xWidget
{
    Q_OBJECT

    enum camera
    {
        FLOWFS,
        LLOWFS,
        CAMSCIS
    };

  protected:
    std::string m_appState;

    QMutex m_mutex;

    // --- modttm
    std::string m_modFsmState;
    int m_modState{ 0 };

    double m_modCh1{ 0 };
    double m_modCh2{ 0 };

    double m_camwfsFreq{ 0 };

    double m_modRad{ 0 };
    double m_modRad_tgt{ 0 };

    float m_stepSize{ 0.1 };

    int m_tipmovewhat{ MOVE_TTM };



    // --- TCS

    std::string m_tcsiState;

    bool m_labMode {true};

    // --- woofer

    std::string m_dmWooferState;
    std::string m_wooferModesState;

    double m_tilt{ 0 };  ///< current value of tilt mode from wooferModes
    double m_tip{ 0 };   ///< current value of tip mode from wooferModes
    double m_focus{ 0 }; ///< current value of focus mode from wooferModes

    float m_focusStepSize{ 0.1 };


    // --- picoscix
    std::string m_picoState {"UNKNOWN"};
    int m_picoscixPos {-1000000000};

    int m_picoscix_stepSize {50};
    std::string m_picoscix_gotoSelection;



    // --- camwfs-fit
    std::string m_camwfsfitState;
    double m_med1{ 0 };
    double m_med2{ 0 };
    double m_med3{ 0 };
    double m_med4{ 0 };

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
    unsigned m_nAverage_current{ 0 };
    unsigned m_nAverage_target{ 0 };

    // -- dmtweeter
    std::string m_dmtweeterState;
    bool m_dmtweeterTestSet{ false };

    // -- dmncpc
    std::string m_dmncpcState;
    bool m_dmncpcTestSet{ false };

    // -- ttmpupil
    std::string m_pupFsmState;
    double m_pupCh1{ 0 };
    double m_pupCh2{ 0 };

    float m_pupStepSize{ 0.5 };

    int m_pupCam{ CAMSCIS };

    // -- ttmperi
    std::string m_ttmPeriFsmState;
    double m_ttmPeriCh1{ 0 };
    double m_ttmPeriCh2{ 0 };

    float m_ttmPeriStepSize{ 25 };

    // -- Camera Lens
    std::string m_camlensxFsmState;
    std::string m_camlensyFsmState;
    float m_camlensx_pos{ 0 };
    float m_camlensy_pos{ 0 };

    float m_camlensStepSize{ 0.01 };

    // ****** Alignment ******** //

    // --- camwfs-align
    std::string m_camwfs_align_fsmState;
    bool m_camwfsAlignLoopState{ false };

    // --- twAlign-camwfs-ctrl
    std::string m_twAlign_camwfs_ctrl_fsmState;
    bool m_twAlignLoopState {false};

    // --- twAlign-camwfs-wfs
    std::string m_twAlign_camwfs_wfs_fsmState;
    bool m_twAlignSensorState {false};

  public:
    pupilGuide( QWidget *Parent = 0, Qt::WindowFlags f = Qt::WindowFlags() );

    ~pupilGuide();

    void subscribe();

    virtual void onConnect();
    virtual void onDisconnect();

    void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );
    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    void modGUISetEnable( bool enableModGUI, bool enableModArrows );

    void camwfsfitSetEnabled( bool enabled );

    /// Enable or disable the cameralens GUI
    /** IF whichcl is CAMLENS_BOTH, the action is applied to all components.
     * If it's CAMLENS_X or CAMLENS_Y, it is only applied to that access.  The common components are then enabled.
     */
    void camlensSetEnabled( bool enabled,            ///< true for enabled, false for disabled
                            int whichcl = CAMLENS_BOTH ///< Which axis, or both.  CAMLENS_X, CAMLENS_Y, CAMLENS_BOTH
    );

    void camwfs_align_setEnabled( bool enabled, bool all );

    void twAlign_camwfs_ctrl_setEnabled( bool enabled, bool all );

    void twAlign_camwfs_wfs_setEnabled( bool enabled, bool all );

    void alignment_buttons_setEnabled( bool enabled, bool all );

  public slots:
    void updateGUI();

    //----------- modttm

    void on_buttonMod_mod_pressed();
    void on_buttonMod_set_pressed();
    void on_buttonMod_rest_pressed();

    void on_button_ttmtel_pressed();

    void on_button_tip_u_pressed();
    void on_button_tip_ul_pressed();
    void on_button_tip_l_pressed();
    void on_button_tip_dl_pressed();
    void on_button_tip_d_pressed();
    void on_button_tip_dr_pressed();
    void on_button_tip_r_pressed();
    void on_button_tip_ur_pressed();
    void on_button_tip_scale_pressed();

    //------------- focus
    void on_button_focus_p_pressed();
    void on_button_focus_m_pressed();
    void on_button_focus_scale_pressed();

    //------------- picoscix
    void move_picoscix(int delta);
    void on_picoscix_l_pressed();
    void on_picoscix_scale_pressed();
    void on_picoscix_r_pressed();
    void on_picoscix_go_pressed();

    //----------- dmtweeter
    void on_buttonTweeterTest_set_pressed();

    //----------- dmncpc
    void on_buttonNCPCTest_set_pressed();

    //----------- ttmpupil
    void on_buttonPup_rest_pressed();
    void on_buttonPup_set_pressed();

    void on_button_camera_pressed();

    void on_button_pup_ul_pressed();
    void on_button_pup_dl_pressed();
    void on_button_pup_dr_pressed();
    void on_button_pup_ur_pressed();
    void on_button_pup_scale_pressed();

    //---------- TTM Peri
    void on_button_ttmPeri_rest_pressed();
    void on_button_ttmPeri_set_pressed();

    void on_button_ttmPeri_l_pressed();
    void on_button_ttmPeri_r_pressed();
    void on_button_ttmPeri_u_pressed();
    void on_button_ttmPeri_d_pressed();
    void on_button_ttmPeri_scale_pressed();

    void toggleExpFit( bool visible );
    void on_buttonExpFit_pressed();

    void on_button_camlens_u_pressed();
    void on_button_camlens_l_pressed();
    void on_button_camlens_d_pressed();
    void on_button_camlens_r_pressed();
    void on_button_camlens_scale_pressed();

    // ******** alignment *********//

    void on_button_startAlignment_pressed();
    void on_button_stopAlignment_pressed();

  private:
    Ui::pupilGuide ui;
};

pupilGuide::pupilGuide( QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f )
{
    char ss[64]; //for scale buttons

    ui.setupUi( this );

    ui.button_focus_scale->setProperty( "isScaleButton", true );
    ui.button_pup_scale->setProperty( "isScaleButton", true );
    ui.button_ttmPeri_scale->setProperty( "isScaleButton", true );

    //-----------modwfs controls ------------

    setXwFont( ui.label_modulation );

    ui.modwfs_fsm->device( "modwfs" );
    ui.modwfs_fsm->NOTHOMED( "RIP" );
    ui.modwfs_fsm->READY("SET");
    ui.modwfs_fsm->OPERATING("MODULATING");

    setXwFont( ui.label_modFreq );
    setXwFont( ui.label_modRad );

    setXwFont( ui.buttonMod_rest );
    setXwFont( ui.buttonMod_set );
    setXwFont( ui.buttonMod_mod );

    ui.modFreq_current->setup( "modwfs", "modFrequency", statusEntry::FLOAT, "", "" );
    ui.modFreq_current->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.modFreq_current->format( "%0.1f" );
    //ui.modFreq_current->onDisconnect();

    ui.modRad_current->setup( "modwfs", "modRadius", statusEntry::FLOAT, "", "" );
    ui.modRad_current->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.modRad_current->format( "%0.1f" );
    //ui.modRad_current->onDisconnect();

    ui.modCh1->setup( "fxngenmodwfs", "C1ofst", statusEntry::FLOAT, "Ch1", "V" );
    ui.modCh1->currEl( "value" );
    ui.modCh1->targEl( "value" );
    ui.modCh1->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.modCh1->format( "%0.2f" );
    //ui.modCh1->onDisconnect();

    ui.modCh2->setup( "fxngenmodwfs", "C2ofst", statusEntry::FLOAT, "Ch2", "V" );
    ui.modCh2->currEl( "value" );
    ui.modCh2->targEl( "value" );
    ui.modCh2->setStretch( 0, 1, 6 ); // removes spacer and maximizes text field
    ui.modCh2->format( "%0.2f" );
    //ui.modCh2->onDisconnect();

    //-----------tip alignment controls ------------

    setXwFont( ui.picoscix_label );

    ui.picoscix_pos->setup( "picomotors", "picoscix_pos", statusEntry::INT, "", "" );
    ui.picoscix_pos->setStretch( 0, 0, 6 ); // removes spacer and maximizes text field
    ui.picoscix_pos->format( "%d" );

    ui.picoscix_scale->setProperty( "isScaleButton", true );
    snprintf( ss, 5, "%0.2f", m_picoscix_stepSize/1000. );
    ui.button_tip_scale->setText( ss );

    ui.picoscix_combo->addItem("    ");
    ui.picoscix_combo->addItem("65-35");
    ui.picoscix_combo->addItem("Ha-IR");
    ui.picoscix_combo->setCurrentText("    ");

    setXwFont( ui.picoscix_combo_label );


    //-----------orphans ------------
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
    ui.pupState->READY("SET");

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
    ui.ttmPeriState->READY("RIP");
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

    m_parent->addSubscriber(ui.modwfs_fsm);
    m_parent->addSubscriberProperty( this, "modwfs", "fsm" );
    m_parent->addSubscriberProperty( this, "modwfs", "modState" );

    m_parent->addSubscriber( ui.modFreq_current );
    m_parent->addSubscriber( ui.modRad_current );
    m_parent->addSubscriber( ui.modCh1 );
    m_parent->addSubscriber( ui.modCh2 );

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

    m_parent->addSubscriber( ui.pupState );
    m_parent->addSubscriber( ui.pupCh1 );
    m_parent->addSubscriber( ui.pupCh2 );
    m_parent->addSubscriberProperty( this, "ttmpupil", "fsm" );
    m_parent->addSubscriberProperty( this, "ttmpupil", "pos_1" );
    m_parent->addSubscriberProperty( this, "ttmpupil", "pos_2" );

    m_parent->addSubscriber( ui.ttmPeriState );
    m_parent->addSubscriber( ui.ttmPeriCh1 );
    m_parent->addSubscriber( ui.ttmPeriCh2 );
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

    m_parent->addSubscriber( ui.pupTrackLoop_deltaX );
    m_parent->addSubscriber( ui.pupTrackLoop_deltaY );

    m_parent->addSubscriber( ui.pupTrackLoop_slider );
    m_parent->addSubscriber( ui.pupTrackLoop_gain );

    m_parent->addSubscriberProperty( this, "twAlign-camwfs-ctrl", "fsm" );
    m_parent->addSubscriberProperty( this, "twAlign-camwfs-ctrl", "loop_state" );

    m_parent->addSubscriber( ui.actAlignLoop_deltaX );
    m_parent->addSubscriber( ui.actAlignLoop_deltaY );

    m_parent->addSubscriber( ui.actAlignLoop_slider );

    m_parent->addSubscriber( ui.actAlignLoop_gain );

    m_parent->addSubscriberProperty( this, "twAlign-camwfs-wfs", "fsm" );
    m_parent->addSubscriberProperty( this, "twAlign-camwfs-wfs", "loop_state" );

    m_parent->addSubscriber( ui.actAlignSensor_slider );

    m_parent->addSubscriber( ui.actAlignSensor_nAverage );
    m_parent->addSubscriber( ui.actAlignSensor_nImages );
    m_parent->addSubscriber( ui.actAlignSensor_pokeAmp );

    m_parent->addSubscriber( ui.fitThreshold );
    m_parent->addSubscriber( ui.fitAvgTime );

    /* Camera Lens */
    m_parent->addSubscriber( ui.camlensX_fsm );
    m_parent->addSubscriber( ui.camlensY_fsm );
    m_parent->addSubscriberProperty( this, "stagecamlensx", "fsm" );
    m_parent->addSubscriberProperty( this, "stagecamlensy", "fsm" );
    m_parent->addSubscriberProperty( this, "stagecamlensx", "position" ); // we need these too
    m_parent->addSubscriberProperty( this, "stagecamlensy", "position" );
    m_parent->addSubscriber( ui.camlensX_pos );
    m_parent->addSubscriber( ui.camlensY_pos );

    return;
}

void pupilGuide::onConnect()
{
    ui.label_modulation->setEnabled( true );
    ui.labelPupilFitting->setEnabled( true );

    ui.modwfs_fsm->onConnect();
    ui.modFreq_current->onConnect();
    ui.modRad_current->onConnect();
    ui.modCh1->onConnect();
    ui.modCh2->onConnect();

    ui.label_tipAlignment->setEnabled(true);
    ui.button_ttmtel->setEnabled(true);

    ui.picoscix_label->setEnabled(true);
    ui.picoscix_pos->onConnect();
    ui.picoscix_l->setEnabled(true);
    ui.picoscix_scale->setEnabled(true);
    ui.picoscix_r->setEnabled(true);
    ui.picoscix_combo_label->setEnabled(true);
    ui.picoscix_combo->setEnabled(true);
    ui.picoscix_go->setEnabled(true);

    ui.button_camera->setEnabled(true);

    ui.label_tweeter->setEnabled(true);

    ui.labelPupilSteering->setEnabled( true );
    ui.pupState->onConnect();
    ui.pupCh1->onConnect();
    ui.pupCh2->onConnect();

    ui.label_ncpc->setEnabled(false);

    ui.labelTTMPeri->setEnabled( true );
    ui.ttmPeriState->onConnect();
    ui.ttmPeriCh1->onConnect();
    ui.ttmPeriCh2->onConnect();

    ui.label_pupilPositions->setEnabled(true);

    /* Camera Lens */
    ui.label_camlens->setEnabled( true );
    ui.label_camlensX_fsm->setEnabled( true );
    ui.label_camlensY_fsm->setEnabled( true );

    ui.camlensX_fsm->onConnect();
    ui.camlensY_fsm->onConnect();
    ui.camlensX_pos->onConnect();
    ui.camlensY_pos->onConnect();

    ui.fitThreshold->onConnect();
    ui.fitAvgTime->onConnect();


    ui.pupTrackLoop_deltaX->onConnect();
    ui.pupTrackLoop_deltaY->onConnect();

    ui.pupTrackLoop_slider->onConnect();
    ui.pupTrackLoop_gain->onConnect();

    ui.actAlignLoop_deltaX->onConnect();
    ui.actAlignLoop_deltaY->onConnect();

    ui.actAlignLoop_slider->onConnect();
    ui.actAlignLoop_gain->onConnect();

    ui.actAlignSensor_slider->onConnect();
    ui.actAlignSensor_nAverage->onConnect();
    ui.actAlignSensor_nImages->onConnect();
    ui.actAlignSensor_pokeAmp->onConnect();


    camwfs_align_setEnabled(true, true);
    twAlign_camwfs_ctrl_setEnabled(true, true);
    twAlign_camwfs_wfs_setEnabled(true, true);
    alignment_buttons_setEnabled(true, true);

    setWindowTitle( "Alignment" );
}

void pupilGuide::onDisconnect()
{
    m_modFsmState = "";

    ui.label_modulation->setEnabled( false );
    ui.modwfs_fsm->onDisconnect();
    ui.modFreq_current->onDisconnect();
    ui.modRad_current->onDisconnect();
    ui.modCh1->onDisconnect();
    ui.modCh2->onDisconnect();

    ui.label_tipAlignment->setEnabled(false);
    ui.button_ttmtel->setEnabled(false);

    ui.picoscix_label->setEnabled(false);
    ui.picoscix_pos->onDisconnect();
    ui.picoscix_l->setEnabled(false);
    ui.picoscix_scale->setEnabled(false);
    ui.picoscix_r->setEnabled(false);
    ui.picoscix_combo_label->setEnabled(false);
    ui.picoscix_combo->setEnabled(false);
    ui.picoscix_go->setEnabled(false);

    ui.button_camera->setEnabled(false);

    ui.label_tweeter->setEnabled(false);

    m_pupFsmState = "";
    ui.labelPupilSteering->setEnabled( false );
    ui.pupState->onDisconnect();
    ui.pupCh1->onDisconnect();
    ui.pupCh2->onDisconnect();

    ui.label_ncpc->setEnabled(false);

    ui.labelTTMPeri->setEnabled( false );
    ui.ttmPeriState->onDisconnect();
    ui.ttmPeriCh1->onDisconnect();
    ui.ttmPeriCh2->onDisconnect();

    m_camlensxFsmState = "";
    m_camlensyFsmState = "";
    m_camwfsavgState = "";
    m_camwfsfitState = "";


    ui.labelPupilFitting->setEnabled( false );


    ui.label_pupilPositions->setEnabled(false);

    /* Camera Lens */
    ui.label_camlens->setEnabled( false );
    ui.label_camlensX_fsm->setEnabled( false );
    ui.label_camlensY_fsm->setEnabled( false );

    ui.camlensX_fsm->onDisconnect();
    ui.camlensY_fsm->onDisconnect();
    ui.camlensX_pos->onDisconnect();
    ui.camlensY_pos->onDisconnect();
    camlensSetEnabled(false);

    ui.fitThreshold->onDisconnect();
    ui.fitAvgTime->onDisconnect();

    ui.pupTrackLoop_deltaX->onDisconnect();
    ui.pupTrackLoop_deltaY->onDisconnect();

    ui.pupTrackLoop_slider->onDisconnect();
    ui.pupTrackLoop_gain->onDisconnect();

    ui.actAlignLoop_deltaX->onDisconnect();
    ui.actAlignLoop_deltaY->onDisconnect();

    ui.actAlignLoop_slider->onDisconnect();
    ui.actAlignLoop_gain->onDisconnect();

    ui.actAlignSensor_slider->onDisconnect();
    ui.actAlignSensor_nAverage->onDisconnect();
    ui.actAlignSensor_nImages->onDisconnect();
    ui.actAlignSensor_pokeAmp->onDisconnect();

    camwfs_align_setEnabled(false, true);
    m_camwfs_align_fsmState = "";
    twAlign_camwfs_ctrl_setEnabled(false, true);
    m_twAlign_camwfs_ctrl_fsmState = "";
    twAlign_camwfs_wfs_setEnabled(false, true);
    m_twAlign_camwfs_wfs_fsmState = "";
    alignment_buttons_setEnabled(false, true);

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
                if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
                {
                    m_labMode = true;
                }
                else
                {
                    m_labMode = false;
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
}

void pupilGuide::modGUISetEnable( bool enableModGUI, bool enableModArrows )
{
    if( enableModGUI )
    {
        ui.label_modulation->setEnabled( true );
        ui.modwfs_fsm->setEnabled( true );
        ui.modwfs_fsm->setEnabled( true );
        if(m_modState == 3 || m_modState == 4)
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
        if(m_modFsmState != "POWEROFF" && m_modFsmState != "CONFIGURING")
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

        if(!enableModArrows)
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

void pupilGuide::camlensSetEnabled( bool enabled,
                                    int whichcl
)
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
    if(all)
    {
        ui.label_pupTrackLoop->setEnabled(enabled);
    }
    ui.pupTrackLoop_deltaX->setEnabled(enabled);
    ui.pupTrackLoop_deltaY->setEnabled(enabled);
    ui.pupTrackLoop_slider->setEnabled(enabled);
    ui.pupTrackLoop_gain->setEnabled(enabled);
}

void pupilGuide::twAlign_camwfs_ctrl_setEnabled( bool enabled, bool all )
{
    if(all)
    {
        ui.label_actAlignLoop->setEnabled(enabled);
    }
    ui.actAlignLoop_deltaX->setEnabled(enabled);
    ui.actAlignLoop_deltaY->setEnabled(enabled);
    ui.actAlignLoop_slider->setEnabled(enabled);
    ui.actAlignLoop_gain->setEnabled(enabled);
}

void pupilGuide::twAlign_camwfs_wfs_setEnabled( bool enabled, bool all )
{
    if(all)
    {
        ui.label_actAlignSensor->setEnabled(enabled);
    }
    ui.actAlignSensor_slider->setEnabled(enabled);
    ui.actAlignSensor_nAverage->setEnabled(enabled);
    ui.actAlignSensor_nImages->setEnabled(enabled);
    ui.actAlignSensor_pokeAmp->setEnabled(enabled);
}

void pupilGuide::alignment_buttons_setEnabled( bool enabled, bool all )
{
    if(all)
    {
        ui.label_alignment->setEnabled(enabled);
    }
    ui.button_startAlignment->setEnabled(enabled);
    ui.button_stopAlignment->setEnabled(enabled);

}

void pupilGuide::updateGUI()
{

    //--------- Modulation

    bool enableModGUI = true;
    bool enableModArrows = true;

    char str[16];
    if( m_modFsmState == "NOTHOMED" )
    {
        if( m_tipmovewhat == MOVE_TTM )
        {
           enableModArrows = false;
        }
    }
    else if( (m_modFsmState != "READY") && (m_modFsmState != "OPERATING") )
    {
        enableModGUI = false;
        if( m_tipmovewhat == MOVE_TTM )
        {
           enableModArrows = false;
        }
    }

    //If moving woofer and either woofer or wooferModes aren't ready we disable the arrows
    if( m_tipmovewhat == MOVE_WOOF && (m_dmWooferState != "OPERATING" || m_wooferModesState != "READY"))
    {
        enableModArrows = false;
    }

    //If moving telescope and tcsi isn't connected we disable the arrows
    if( m_tipmovewhat == MOVE_TEL && (m_tcsiState != "CONNECTED"))
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

    ui.modwfs_fsm->updateGUI();
    ui.modFreq_current->updateGUI();
    ui.modRad_current->updateGUI();
    ui.modCh1->updateGUI();
    ui.modCh2->updateGUI();

    // ------picoscis
    if(m_picoState != "READY")
    {
        ui.picoscix_pos->setEnabled(false);
        ui.picoscix_l->setEnabled(false);
        ui.picoscix_scale->setEnabled(false);
        ui.picoscix_r->setEnabled(false);
        ui.picoscix_combo->setEnabled(false);
        ui.picoscix_go->setEnabled(false);
    }
    else
    {
        ui.picoscix_pos->setEnabled(true);
        ui.picoscix_l->setEnabled(true);
        ui.picoscix_scale->setEnabled(true);
        ui.picoscix_r->setEnabled(true);
        ui.picoscix_combo->setEnabled(true);
        ui.picoscix_go->setEnabled(true);
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
            m1 = m_med1 - ave;
            m2 = m_med2 - ave;
            m3 = m_med3 - ave;
            m4 = m_med4 - ave;
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
    bool enablePupFSM = true;
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
    bool enableTTMPeriFSM = true;
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
        ui.camlensY_pos->onDisconnect();
    }
    else if( !( m_camlensxFsmState == "READY" || m_camlensxFsmState == "OPERATING" ) &&
             ( m_camlensyFsmState == "READY" || m_camlensyFsmState == "OPERATING" ) )
    {
        camlensSetEnabled( false, CAMLENS_X );
        ui.camlensX_pos->onDisconnect();

        camlensSetEnabled( true, CAMLENS_Y );
    }
    else
    {
        camlensSetEnabled( false );
        ui.camlensX_pos->onDisconnect();
        ui.camlensY_pos->onDisconnect();
    }

    if(m_camlensxFsmState == "SHUTDOWN")
    {
        ui.camlensX_pos->onDisconnect();
    }

    if(m_camlensyFsmState == "SHUTDOWN")
    {
        ui.camlensY_pos->onDisconnect();
    }

    ui.camlensX_fsm->updateGUI();
    ui.camlensY_fsm->updateGUI();
    ui.camlensX_pos->updateGUI();
    ui.camlensY_pos->updateGUI();





    ui.fitThreshold->updateGUI();
    ui.fitAvgTime->updateGUI();

    if(m_camwfs_align_fsmState != "READY" && m_camwfs_align_fsmState != "OPERATING")
    {
        camwfs_align_setEnabled(false, false);
    }
    else
    {
        camwfs_align_setEnabled(true, true);
    }

    ui.pupTrackLoop_deltaX->updateGUI();
    ui.pupTrackLoop_deltaY->updateGUI();
    ui.pupTrackLoop_slider->updateGUI();
    ui.pupTrackLoop_gain->updateGUI();

    if(m_twAlign_camwfs_ctrl_fsmState != "READY" && m_twAlign_camwfs_ctrl_fsmState != "OPERATING")
    {
        twAlign_camwfs_ctrl_setEnabled(false, false);
    }
    else
    {
        twAlign_camwfs_ctrl_setEnabled(true, true);
    }

    ui.actAlignLoop_deltaX->updateGUI();
    ui.actAlignLoop_deltaY->updateGUI();
    ui.actAlignLoop_slider->updateGUI();
    ui.actAlignLoop_gain->updateGUI();

    if(m_twAlign_camwfs_wfs_fsmState != "READY" && m_twAlign_camwfs_wfs_fsmState != "OPERATING")
    {
        twAlign_camwfs_wfs_setEnabled(false, false);
    }
    else
    {
        twAlign_camwfs_wfs_setEnabled(true, true);
    }

    ui.actAlignSensor_slider->updateGUI();
    ui.actAlignSensor_nAverage->updateGUI();
    ui.actAlignSensor_nImages->updateGUI();
    ui.actAlignSensor_pokeAmp->updateGUI();

} // updateGUI()

// ------------- modttm

void pupilGuide::on_buttonMod_mod_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "modwfs" );
    ip.setName( "modState" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = 4;

    sendNewProperty( ip );
}

void pupilGuide::on_buttonMod_set_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( "modwfs" );
    ip.setName( "modState" );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = 3;
    sendNewProperty( ip );
}

void pupilGuide::on_buttonMod_rest_pressed()
{
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
        ip["x"] = m_stepSize * 5. / sqrt( 2. );
    }

    sendNewProperty( ip );
}

void pupilGuide::on_button_tip_l_pressed()
{
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
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_WOOF )
    {

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0002" ) );
        ip["0002"] = m_focus + m_focusStepSize * 0.2;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "z" ) );
        ip["z"] = m_stepSize * 100.;
    }
    else
        return;

    sendNewProperty( ip );
}

void pupilGuide::on_button_focus_m_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    if( m_tipmovewhat == MOVE_WOOF )
    {

        ip.setDevice( "wooferModes" );
        ip.setName( "target_amps" );
        ip.add( pcf::IndiElement( "0002" ) );
        ip["0002"] = m_focus - m_focusStepSize * 0.2;
    }
    else if( m_tipmovewhat == MOVE_TEL )
    {
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "z" ) );
        ip["z"] = -m_stepSize * 100.;
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

void pupilGuide::move_picoscix(int delta)
{
    if(m_picoState != "READY" || m_picoscixPos < -1000000)
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
    move_picoscix(+m_picoscix_stepSize);
}

void pupilGuide::on_picoscix_scale_pressed()
{
    if( m_picoscix_stepSize  == 1000 )
    {
        m_picoscix_stepSize = 500;
    }
    else if( m_picoscix_stepSize  == 500 )
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
    snprintf( ss, 5, "%0.2f", m_picoscix_stepSize/1000. );
    ui.picoscix_scale->setText( ss );
}

void pupilGuide::on_picoscix_r_pressed()
{
    move_picoscix(-m_picoscix_stepSize);
}

void pupilGuide::on_picoscix_go_pressed()
{
    QString select = ui.picoscix_combo->currentText();

    if(select == "65-35")
    {
        move_picoscix(-7000);
    }

    if(select == "Ha-IR")
    {
        move_picoscix(7000);
    }

    ui.picoscix_combo->setCurrentText("    ");
}



//----------- dmtweeter

void pupilGuide::on_buttonTweeterTest_set_pressed()
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

void pupilGuide::on_buttonNCPCTest_set_pressed()
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

void pupilGuide::on_buttonPup_rest_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmpupil" );
    ip.setName( "releaseDM" );
    ip.add( pcf::IndiElement( "request" ) );
    ip["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ip );
}

void pupilGuide::on_buttonPup_set_pressed()
{
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
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmperi" );
    ip.setName( "set" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"].setSwitchState( pcf::IndiElement::Off );

    sendNewProperty( ip );
}

void pupilGuide::on_button_ttmPeri_set_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( "ttmperi" );
    ip.setName( "set" );
    ip.add( pcf::IndiElement( "toggle" ) );
    ip["toggle"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ip );
}

void pupilGuide::on_button_ttmPeri_l_pressed()
{
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
        ui.buttonExpFit->setIcon( QIcon(":/icons/keyboard_double_arrow_up.png") );
    }
    else
    {
        ui.buttonExpFit->setIcon( QIcon(":/icons/keyboard_double_arrow_down.png") );
    }
}

void pupilGuide::on_buttonExpFit_pressed()
{
    bool st = !ui.labelD->isVisible();
    toggleExpFit( st );
}

void pupilGuide::on_button_camlens_u_pressed()
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

void pupilGuide::on_button_camlens_l_pressed()
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

void pupilGuide::on_button_camlens_d_pressed()
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

void pupilGuide::on_button_camlens_r_pressed()
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


    if(m_labMode)
    {
        ip.setDevice( "camwfs-align" );
        ip.setName( "loop_state" );
        ip["toggle"] = pcf::IndiElement::Off;

        sendNewProperty( ip );
    }
}

} // namespace xqt

#include "moc_pupilGuide.cpp"

#endif
