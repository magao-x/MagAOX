/** \file picam_advanced.h
 * \brief Minimal PICam SDK declarations for offline picamCtrl tests.
 */
#ifndef tests_picam_advanced_h
#define tests_picam_advanced_h

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
using piint                                  = int;
using piflt                                  = double;
using pibln                                  = int;
using pichar                                 = char;
using pibyte                                 = unsigned char;
using pi64s                                  = long;
using PicamHandle                            = void *;
using PicamModel                             = int;
using PicamComputerInterface                 = int;
using PicamConstraintScope                   = int;
using PicamConstraintSeverity                = int;
constexpr piint PicamStringSize_SensorName   = 64;
constexpr piint PicamStringSize_SerialNumber = 64;

enum PicamError
{
    PicamError_None                  = 0,
    PicamError_InvalidParameterValue = 2,
    PicamError_TimeOutOccurred       = 32,
};

enum PicamEnumeratedType
{
    PicamEnumeratedType_Error         = 1,
    PicamEnumeratedType_Model         = 2,
    PicamEnumeratedType_AdcAnalogGain = 7,
    PicamEnumeratedType_AdcQuality    = 8,
};

enum PicamParameter
{
    PicamParameter_ExposureTime              = 33685527,
    PicamParameter_AdcSpeed                  = 50462753,
    PicamParameter_AdcAnalogGain             = 50593827,
    PicamParameter_AdcQuality                = 50593828,
    PicamParameter_AdcEMGain                 = 33620021,
    PicamParameter_TriggerResponse           = 50593822,
    PicamParameter_TriggerDetermination      = 50593823,
    PicamParameter_ReadoutControlMode        = 50593818,
    PicamParameter_ReadoutTimeCalculation    = 16908315,
    PicamParameter_VerticalShiftRate         = 50462733,
    PicamParameter_Rois                      = 67436581,
    PicamParameter_ReadoutCount              = 33947688,
    PicamParameter_FrameSize                 = 16842794,
    PicamParameter_FrameStride               = 16842795,
    PicamParameter_FramesPerReadout          = 16842796,
    PicamParameter_ReadoutStride             = 16842797,
    PicamParameter_PixelBitDepth             = 16842800,
    PicamParameter_FrameRateCalculation      = 16908339,
    PicamParameter_TimeStamps                = 50593860,
    PicamParameter_TimeStampResolution       = 50724933,
    PicamParameter_SensorTemperatureSetPoint = 33685518,
    PicamParameter_SensorTemperatureReading  = 16908303,
    PicamParameter_SensorTemperatureStatus   = 17039376,
    PicamParameter_DisableCoolingFan         = 50528285,
    PicamParameter_CoolingFanStatus          = 17039522,
};

enum PicamAdcQuality
{
    PicamAdcQuality_LowNoise           = 1,
    PicamAdcQuality_ElectronMultiplied = 3,
};

enum PicamCoolingFanStatus
{
    PicamCoolingFanStatus_Off      = 1,
    PicamCoolingFanStatus_On       = 2,
    PicamCoolingFanStatus_ForcedOn = 3,
};

enum PicamReadoutControlMode
{
    PicamReadoutControlMode_FrameTransfer = 2,
};

enum PicamSensorTemperatureStatus
{
    PicamSensorTemperatureStatus_Unlocked = 1,
    PicamSensorTemperatureStatus_Locked   = 2,
    PicamSensorTemperatureStatus_Faulted  = 3,
};

enum PicamTimeStampsMask
{
    PicamTimeStampsMask_ExposureStarted = 1,
};

enum PicamTriggerDetermination
{
    PicamTriggerDetermination_RisingEdge = 3,
};

enum PicamTriggerResponse
{
    PicamTriggerResponse_NoResponse        = 1,
    PicamTriggerResponse_ReadoutPerTrigger = 2,
};

enum PicamConstraintCategory
{
    PicamConstraintCategory_Required = 2,
};

enum PicamAcquisitionErrorsMask
{
    PicamAcquisitionErrorsMask_None              = 0,
    PicamAcquisitionErrorsMask_CameraFaulted     = 16,
    PicamAcquisitionErrorsMask_ConnectionLost    = 2,
    PicamAcquisitionErrorsMask_ShutterOverheated = 8,
    PicamAcquisitionErrorsMask_DataLost          = 1,
    PicamAcquisitionErrorsMask_DataNotArriving   = 4,
};

/// SDK layout used by the offline double.
struct PicamCameraID
{
    /// Camera model identifier.
    PicamModel model;

    /// Camera communication interface.
    PicamComputerInterface computer_interface;

    /// Sensor name returned by discovery.
    pichar sensor_name[PicamStringSize_SensorName];

    /// Camera serial number returned by discovery.
    pichar serial_number[PicamStringSize_SerialNumber];
};

/// SDK layout used by the offline double.
struct PicamRoi
{
    /// First sensor column in the ROI.
    piint x;

    /// ROI width before binning.
    piint width;

    /// Horizontal binning factor.
    piint x_binning;

    /// First sensor row in the ROI.
    piint y;

    /// ROI height before binning.
    piint height;

    /// Vertical binning factor.
    piint y_binning;
};

/// SDK layout used by the offline double.
struct PicamRois
{
    /// SDK-owned array of regions.
    PicamRoi *roi_array;

    /// Number of regions in the array.
    piint roi_count;
};

/// SDK layout used by the offline double.
struct PicamRangeConstraint
{
    /// Whether the constraint depends on other parameters.
    PicamConstraintScope scope;

    /// Severity of violating the constraint.
    PicamConstraintSeverity severity;

    /// Whether no candidate values are permitted.
    pibln empty_set;

    /// Smallest value in the linear range.
    piflt minimum;

    /// Largest value in the linear range.
    piflt maximum;

    /// Spacing between permitted values.
    piflt increment;

    /// SDK-owned values excluded from the linear range.
    const piflt *excluded_values_array;

    /// Number of excluded values.
    piint excluded_values_count;

    /// SDK-owned permitted values outside the linear range.
    const piflt *outlying_values_array;

    /// Number of permitted outlying values.
    piint outlying_values_count;
};

/// SDK layout used by the offline double.
struct PicamAvailableData
{
    /// Borrowed pointer to available acquisition data.
    void *initial_readout;

    /// Number of available readouts.
    pi64s readout_count;
};

/// SDK layout used by the offline double.
struct PicamAcquisitionStatus
{
    /// Whether acquisition remains active.
    pibln running;

    /// Acquisition error flags.
    PicamAcquisitionErrorsMask errors;

    /// Reported readout rate.
    piflt readout_rate;
};

/// SDK layout used by the offline double.
struct PicamAcquisitionBuffer
{
    /// App-owned acquisition storage.
    void *memory;

    /// Allocated acquisition bytes.
    pi64s memory_size;
};

extern "C"
{
    /// Offline SDK entry point for PicamAdvanced_GetCameraModel.
    PicamError PicamAdvanced_GetCameraModel( PicamHandle  camera /**< [in] Camera handle. */,
                                             PicamHandle *model /**< [out] Camera model handle. */ );

    /// Offline SDK entry point for PicamAdvanced_GetParameterRangeConstraints.
    PicamError PicamAdvanced_GetParameterRangeConstraints(
        PicamHandle                  camera_or_accessory /**< [in] Camera or accessory handle. */,
        PicamParameter               parameter /**< [in] Parameter to access. */,
        const PicamRangeConstraint **constraint_array /**< [out] SDK range constraints. */,
        piint                       *constraint_count /**< [out] Number of range constraints. */ );

    /// Offline SDK entry point for PicamAdvanced_OpenCameraDevice.
    PicamError PicamAdvanced_OpenCameraDevice( const PicamCameraID *id /**< [in] Camera identifier. */,
                                               PicamHandle         *device /**< [out] Camera device handle. */ );

    /// Offline SDK entry point for PicamAdvanced_SetAcquisitionBuffer.
    PicamError PicamAdvanced_SetAcquisitionBuffer(
        PicamHandle                   device /**< [in] Camera device handle. */,
        const PicamAcquisitionBuffer *buffer /**< [in] Acquisition storage descriptor. */ );

    /// Offline SDK entry point for Picam_CanReadParameter.
    PicamError Picam_CanReadParameter( PicamHandle    camera_or_accessory /**< [in] Camera or accessory handle. */,
                                       PicamParameter parameter /**< [in] Parameter to access. */,
                                       pibln         *readable /**< [out] Whether direct reading is supported. */ );

    /// Offline SDK entry point for Picam_CanSetParameterFloatingPointValue.
    PicamError
    Picam_CanSetParameterFloatingPointValue( PicamHandle camera_or_accessory /**< [in] Camera or accessory handle. */,
                                             PicamParameter parameter /**< [in] Parameter to access. */,
                                             piflt          value /**< [in] Parameter value. */,
                                             pibln *settable /**< [out] Whether the candidate value is valid. */ );

    /// Offline SDK entry point for Picam_CanSetParameterOnline.
    PicamError Picam_CanSetParameterOnline( PicamHandle    camera_or_accessory /**< [in] Camera or accessory handle. */,
                                            PicamParameter parameter /**< [in] Parameter to access. */,
                                            pibln *onlineable /**< [out] Whether online changes are supported. */ );

    /// Offline SDK entry point for Picam_CloseCamera.
    PicamError Picam_CloseCamera( PicamHandle camera /**< [in] Camera handle. */ );

    /// Offline SDK entry point for Picam_CommitParameters.
    PicamError Picam_CommitParameters(
        PicamHandle            camera_or_accessory /**< [in] Camera or accessory handle. */,
        const PicamParameter **failed_parameter_array /**< [out] SDK array of parameters that failed to commit. */,
        piint                 *failed_parameter_count /**< [out] Number of failed parameters. */ );

    /// Offline SDK entry point for Picam_DestroyCameraIDs.
    PicamError Picam_DestroyCameraIDs( const PicamCameraID *id_array /**< [in] SDK camera identifiers. */ );

    /// Offline SDK entry point for Picam_DestroyParameters.
    PicamError Picam_DestroyParameters(
        const PicamParameter *parameter_array /**< [in] SDK-owned parameter identifiers to release. */ );

    /// Offline SDK entry point for Picam_DestroyRangeConstraints.
    PicamError
    Picam_DestroyRangeConstraints( const PicamRangeConstraint *constraint_array /**< [in] SDK range constraints. */ );

    /// Offline SDK entry point for Picam_DestroyRois.
    PicamError Picam_DestroyRois( const PicamRois *rois /**< [in] ROI descriptors. */ );

    /// Offline SDK entry point for Picam_DestroyString.
    PicamError Picam_DestroyString( const pichar *s /**< [in] SDK enumeration description. */ );

    /// Offline SDK entry point for Picam_DoesParameterExist.
    PicamError Picam_DoesParameterExist( PicamHandle    camera_or_accessory /**< [in] Camera or accessory handle. */,
                                         PicamParameter parameter /**< [in] Parameter to access. */,
                                         pibln         *exists /**< [out] Whether the parameter is available. */ );

    /// Offline SDK entry point for Picam_GetAvailableCameraIDs.
    PicamError Picam_GetAvailableCameraIDs( const PicamCameraID **id_array /**< [out] SDK camera identifiers. */,
                                            piint *id_count /**< [out] Number of camera identifiers. */ );

    /// Offline SDK entry point for Picam_GetEnumerationString.
    PicamError Picam_GetEnumerationString( PicamEnumeratedType type /**< [in] Enumeration type. */,
                                           piint               value /**< [in] Parameter value. */,
                                           const pichar      **s /**< [out] SDK enumeration description. */ );

    /// Offline SDK entry point for Picam_GetParameterFloatingPointValue.
    PicamError
    Picam_GetParameterFloatingPointValue( PicamHandle    camera_or_accessory /**< [in] Camera or accessory handle. */,
                                          PicamParameter parameter /**< [in] Parameter to access. */,
                                          piflt         *value /**< [out] Parameter value. */ );

    /// Offline SDK entry point for Picam_GetParameterIntegerValue.
    PicamError Picam_GetParameterIntegerValue( PicamHandle camera_or_accessory /**< [in] Camera or accessory handle. */,
                                               PicamParameter parameter /**< [in] Parameter to access. */,
                                               piint         *value /**< [out] Parameter value. */ );

    /// Offline SDK entry point for Picam_GetParameterLargeIntegerValue.
    PicamError Picam_GetParameterLargeIntegerValue( PicamHandle    camera /**< [in] Camera handle. */,
                                                    PicamParameter parameter /**< [in] Parameter to access. */,
                                                    pi64s         *value /**< [out] Parameter value. */ );

    /// Offline SDK entry point for Picam_GetParameterRangeConstraint.
    PicamError Picam_GetParameterRangeConstraint(
        PicamHandle                  camera_or_accessory /**< [in] Camera or accessory handle. */,
        PicamParameter               parameter /**< [in] Parameter to access. */,
        PicamConstraintCategory      category /**< [in] Constraint category to query. */,
        const PicamRangeConstraint **constraint /**< [out] Allocated required range. */ );

    /// Offline SDK entry point for Picam_GetParameterRoisValue.
    PicamError Picam_GetParameterRoisValue( PicamHandle       camera /**< [in] Camera handle. */,
                                            PicamParameter    parameter /**< [in] Parameter to access. */,
                                            const PicamRois **value /**< [out] Parameter value. */ );

    /// Offline SDK entry point for Picam_InitializeLibrary.
    PicamError Picam_InitializeLibrary();

    /// Offline SDK entry point for Picam_IsAcquisitionRunning.
    PicamError Picam_IsAcquisitionRunning( PicamHandle camera /**< [in] Camera handle. */,
                                           pibln      *running /**< [out] Whether acquisition is running. */ );

    /// Offline SDK entry point for Picam_SetParameterFloatingPointValue.
    PicamError
    Picam_SetParameterFloatingPointValue( PicamHandle    camera_or_accessory /**< [in] Camera or accessory handle. */,
                                          PicamParameter parameter /**< [in] Parameter to access. */,
                                          piflt          value /**< [in] Parameter value. */ );

    /// Offline SDK entry point for Picam_SetParameterFloatingPointValueOnline.
    PicamError Picam_SetParameterFloatingPointValueOnline( PicamHandle    camera /**< [in] Camera handle. */,
                                                           PicamParameter parameter /**< [in] Parameter to access. */,
                                                           piflt          value /**< [in] Parameter value. */ );

    /// Offline SDK entry point for Picam_SetParameterIntegerValue.
    PicamError Picam_SetParameterIntegerValue( PicamHandle camera_or_accessory /**< [in] Camera or accessory handle. */,
                                               PicamParameter parameter /**< [in] Parameter to access. */,
                                               piint          value /**< [in] Parameter value. */ );

    /// Offline SDK entry point for Picam_SetParameterIntegerValueOnline.
    PicamError Picam_SetParameterIntegerValueOnline( PicamHandle    camera /**< [in] Camera handle. */,
                                                     PicamParameter parameter /**< [in] Parameter to access. */,
                                                     piint          value /**< [in] Parameter value. */ );

    /// Offline SDK entry point for Picam_SetParameterLargeIntegerValue.
    PicamError Picam_SetParameterLargeIntegerValue( PicamHandle    camera /**< [in] Camera handle. */,
                                                    PicamParameter parameter /**< [in] Parameter to access. */,
                                                    pi64s          value /**< [in] Parameter value. */ );

    /// Offline SDK entry point for Picam_SetParameterRoisValue.
    PicamError Picam_SetParameterRoisValue( PicamHandle      camera /**< [in] Camera handle. */,
                                            PicamParameter   parameter /**< [in] Parameter to access. */,
                                            const PicamRois *value /**< [in] Parameter value. */ );

    /// Offline SDK entry point for Picam_StartAcquisition.
    PicamError Picam_StartAcquisition( PicamHandle camera /**< [in] Camera handle. */ );

    /// Offline SDK entry point for Picam_StopAcquisition.
    PicamError Picam_StopAcquisition( PicamHandle camera /**< [in] Camera handle. */ );

    /// Offline SDK entry point for Picam_UninitializeLibrary.
    PicamError Picam_UninitializeLibrary();

    /// Offline SDK entry point for Picam_WaitForAcquisitionUpdate.
    PicamError Picam_WaitForAcquisitionUpdate( PicamHandle         camera /**< [in] Camera handle. */,
                                               piint               readout_time_out /**< [in] Acquisition timeout. */,
                                               PicamAvailableData *available /**< [out] Available readout data. */,
                                               PicamAcquisitionStatus *status /**< [out] Acquisition status. */ );
}
/// \endcond
#endif
