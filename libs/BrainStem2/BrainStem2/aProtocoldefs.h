/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: aProtocoldefs.h                                           //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: Defines used for Brainstem communications.         //
//                                                                 //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// Copyright (c) 2018 Acroname Inc. - All Rights Reserved          //
//                                                                 //
// This file is part of the BrainStem release. See the license.txt //
// file included with this package or go to                        //
// https://acroname.com/software/brainstem-development-kit         //
// for full license details.                                       //
/////////////////////////////////////////////////////////////////////

#ifndef _aProtocolDefs_H_
#define _aProtocolDefs_H_

#include "aError.h"

/////////////////////////////////////////////////////////////////////
/// BrainStem Protocol Definitions

/** \defgroup aProtocoldefs Protocol Defines
 * \ref aProtocoldefs "aProtocoldefs.h" Provides protocol and BrainStem
 * specific defines for entities, communication, and protocol specifics.
 */

/////////////////////////////////////////////////////////////////////
/// BrainStem model codes
#ifndef __aMODULE_DEF__
#define __aMODULE_DEF__

#define aMODULE_TYPE_USBStem_1                                      4
#define aMODULE_TYPE_EtherStem_1                                    5
#define aMODULE_TYPE_MTMIOSerial_1                                 13
#define aMODULE_TYPE_MTM_PM_1                                      14
#define aMODULE_TYPE_MTM_EtherStem                                 15
#define aMODULE_TYPE_MTM_USBStem                                   16
#define aMODULE_TYPE_USBHub2x4                                     17
#define aMODULE_TYPE_MTM_Relay                                     18
#define aMODULE_TYPE_USBHub3p                                      19
#define aMODULE_TYPE_MTM_DAQ_1                                     20
#define aMODULE_TYPE_USBC_Switch                                   21
#define aMODULE_TYPE_MTM_DAQ_2                                     22

#endif // __aMODULE_DEF__



/////////////////////////////////////////////////////////////////////
/// **8 Bytes** - Packet protocol payload maximum.
#define aBRAINSTEM_MAXPACKETBYTES                                  28

/////////////////////////////////////////////////////////////////////
/// UEI and Command support for C/C++ and Reflex languages.

/** \defgroup UEI_Defines (UEI Specific Defines)
 *
 * @{
 */

/// **0x1F** - Mask bits for Index on index byte.
#define ueiSPECIFIER_INDEX_MASK                                  0x1F
/// **0xE0** -  Mask bits for Return value on index byte.
#define ueiSPECIFIER_RETURN_MASK                                 0xE0
/// **1 << 5** - Specifier Bit for UEI response to host.
#define ueiSPECIFIER_RETURN_HOST                             (1 << 5)
/// **2 << 5** - Specifier Bit for UEI response to Module over I2C.
#define ueiSPECIFIER_RETURN_I2C                              (2 << 5)
/// **3 << 5** - Specifier Bit for UEI response to VM on module.
#define ueiSPECIFIER_RETURN_VM                               (3 << 5)
/// **1 << 7** - Error flag on response in index byte.
#define ueiREPLY_ERROR                                       (1 << 7)

/// **0x40** - Option byte code for UEI Get request.
#define ueiOPTION_GET                                            0x40
/// **0x00** - Option byte code for UEI Val response.
#define ueiOPTION_VAL                                            0x00
/// **0x80** - Option byte code for UEI Set request.
#define ueiOPTION_SET                                            0x80
/// **0xC0** - Option byte code for UEI Ack response.
#define ueiOPTION_ACK                                            0xC0
/// **0x3F** - Mask for getting command option from option byte.
#define ueiOPTION_MASK                                           0x3F
/// **0xC0** - Mask for getting Operation Get/Set/Val/Ack
#define ueiOPTION_OP_MASK                                        0xC0
/** @} */

/////////////////////////////////////////////////////////////////////
// Command codes

/////////////////////////////////////////////////////////////////////
// Internal commands, not exposed at C/C++ API
#define cmdHB                                                       0
#define   val_HB_S2H_UP                                             0
#define   val_HB_S2H_DOWN                                           1
#define   val_HB_H2S_UP                                             2
#define   val_HB_H2S_DOWN                                           3
#define   val_HB_M2R_UP                                             4
#define   val_HB_M2R_DOWN                                           5
#define cmdROUTE                                                    1
#define cmdI2C_XMIT                                                 2
#define cmdMAGIC                                                 0xAD
#define cmdFORCEROUTE                                            0xAF

/////////////////////////////////////////////////////////////////////
// API Commands

/** \defgroup cmdSYSTEM_Defines System Command Defines
 * System entity defines
 *
 * @{
 */

/// **3** - System entity command code.
#define cmdSYSTEM                                                   3

/** \defgroup cmdSYSTEM_Command_Options System Command Options
 * \ingroup cmdSYSTEM_Defines
 *
 * @{
 */

/// **1** - Module address option code.
#define    systemModule                                             1
/// **2** - Router address option code.
#define    systemRouter                                             2
/// **3** - Heartbeat interval option code.
#define    systemHBInterval                                         3
/// **4** - User LED option code.
#define    systemLED                                                4
/// **5** - Sleep option code.
#define    systemSleep                                              5
/// **6** - Boot Slot option code.
#define    systemBootSlot                                           6
/// **255** - Disable boot slot value for Boot Slot option.
#define       aSystemBootSlotNone                                 255
/// **7** - Firmware Version option code.
#define    systemVersion                                            7
/// **8** - Model option code.
#define    systemModel                                              8
/// **9** - Serial Number option code.
#define    systemSerialNumber                                       9
/// **10** - System save option code.
#define    systemSave                                              10
/// **11** - System reset option code.
#define    systemReset                                             11
/// **12** - Input voltage option code.
#define    systemInputVoltage                                      12
/// **13** - Module Offset option code.
#define    systemModuleHardwareOffset                              13
/// **14** - Module Base address option code.
#define    systemModuleBaseAddress                                 14
/// **15** - Module Software offset option code.
#define    systemModuleSoftwareOffset                              15
/// **16** - Router address setting option code.
#define    systemRouterAddressSetting                              16
/// **17** - IP configuration setting option code
#define    systemIPConfiguration                                   17
#define        systemIPModeDHCP                                     0
#define        systemIPModeStatic                                   1
#define        systemIPModeDefault                                  0
/// **18** - IP address setting option code
#define    systemIPAddress                                         18
/// **19** - Static IP address setting option code
#define    systemIPStaticAddressSetting                            19
/// **20** - Route to me setting option code
#define    systemRouteToMe                                         20
/// **21** - Input current option code.
#define    systemInputCurrent                                      21
/// **22** - System uptime option code.
#define    systemUptime                                            22
/// **23** - System max temperature option code.
#define    systemMaxTemperature                                    23
/// **24** - System log events option code.
#define    systemLogEvents                                         24
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////


/** \defgroup cmdSLOT_Defines Slot Command Defines
 *  System entity defines
 *
 * @{
 */

/// **4** - Slot Command Code.
#define cmdSLOT                                                     4

/** \defgroup cmdSLOT_Command_Options Slot Command Options
 * \ingroup cmdSLOT_Defines
 *
 * @{
 */

/// **1** - Slot Capacity option code.
#define	   slotCapacity                                             1
/// **2** - Slot size option code
#define	   slotSize                                                 2
/// **3** - Slot Open Read option code.
#define	   slotOpenRead                                             3
/// **4** - Slot Open Write option code.
#define	   slotOpenWrite                                            4
/// **5** - Slot Seek option code.
#define	   slotSeek                                                 5
/// **6** - Slot Read option code.
#define	   slotRead                                                 6
/// **7** - Slot Write option code.
#define	   slotWrite                                                7
/// **8** - Slot Close option code.
#define    slotClose                                                8
/// **0x80** - Bit Slot error code.
#define    bitSlotError                                          0x80

/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////
// Allows users to create custom behavior via reflex code.

/** \defgroup cmdAPP_Defines App Command Defines
 * App Entity defines
 *
 * @{
 */

/// **5** - App command code.
#define cmdAPP                                                      5

/** \defgroup cmdAPP_Command_Options App Command Options
 * \ingroup cmdAPP_Defines
 *
 * @{
 */
/// **1** - Execute option code.
#define    appExecute                                               1
/// **2** - Return option code.
#define    appReturn                                                2
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////

/** \defgroup cmdMUX_Defines Mux Command Defines
 * Mux Entity defines
 *
 * @{
 */

/// **6** - Mux command code.
#define cmdMUX                                                      6

/** \defgroup cmdMUX_Command_Options Mux Command Options
 * \ingroup cmdMUX_Defines
 *
 * @{
 */

/// **1** - Channel enable option code.
#define    muxEnable                                                1
/// **2** - Select the active channel on the mux.
#define    muxChannel                                               2
/// **3** - Get voltage measurement for the channel.
#define    muxVoltage                                               3
/// **4** - Get voltage measurement for the channel.
#define    muxConfig                                                4
#define       muxConfig_default                                     0
#define       muxConfig_splitMode                                   1
#define       muxConfig_crossbar                                    2
/// **5** - Get voltage measurement for the channel.
#define    muxSplit                                                 5
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////

/** \defgroup cmdPOINTER_Defines Pointer command defines
 * Pointer entity defines.
 *
 * @{
 */

/// **7** -  Pointer command code.
#define cmdPOINTER                                                  7

/** \defgroup cmdPOINTER_Command_Options Pointer command options
 * \ingroup cmdPOINTER_Defines
 *
 * @{
 */

/// **1** - Pointer offset option code.
#define    pointerOffset                                            1
/// **2** - Pointer mode option code.
#define    pointerMode                                              2
/// **0** - Static pointer mode for pointer mode option code.
#define       pointerModeStatic                                     0
/// **1** - Increment pointer mode for pointer mode option code.
#define       pointerModeIncrement                                  1
/// **pointerModeStatic** - Default pointer mode for pointer mode option code.
#define       DefaultPointerMode                    pointerModeStatic
/// **3** - Set Transfer store option code.
#define    pointerTransferStore                                     3
/// **4** - Char pointer option code.
#define    pointerChar                                              4
/// **5** - Short pointer option code.
#define    pointerShort                                             5
/// **6** - Int pointer option code.
#define    pointerInt                                               6
/// **7** - Transfer to Store option code.
#define    pointerTransferToStore                                   7
/// **8** - Transfer From store option code.
#define    pointerTransferFromStore                                 8

/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////

/// Write and read devices on I2C bus.
#define cmdI2C                                                      8
/// Set pullup enable and disable on stems which support software control
#define    i2cSetPullup                                             1
/// Route-to address (Reserved for internal use)
#define cmdRTA                                                      9

/// Possible speed settings for I2C bus.
#define     i2cDefaultSpeed                                         0
#define     i2cSpeed_100Khz                                         1
#define     i2cSpeed_400Khz                                         2
#define     i2cSpeed_1000Khz                                        3



/////////////////////////////////////////////////////////////////////
/** \defgroup cmdSERVO_Defines RCServo command defines
 * RCServo entity defines.
 *
 * @{
 */

/// **13** - RC Servo command code.
#define cmdSERVO                                                   13

/** \defgroup cmdSERVO_Command_Options RCServo command options
 * \ingroup cmdSERVO_Defines
 *
 * @{
 */
/// **1** - RCServo enable/disable option code.
#define    servoEnable                                              1
/// **2** - RCServo position option code.
#define    servoPosition                                            2
/// **3** - RCServo reverse option code.
#define    servoReverse                                             3

/** @} */
/** @} */
/////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////
/** \defgroup cmdSIGNAL Signal command defines
 * Signal entity defines.
 *
 * @{
 */

/// **14** - cmdSIGNAL command code.
#define cmdSIGNAL                                                  14

/** \defgroup cmdSIGNAL_Command_Options Digital signal command options
 * \ingroup cmdSIGNAL_Defines
 *
 * @{
 */
/// **1** - Signal enable/disable option code.
#define    signalEnable                                              1
/// **2** - Signal get/set inversion of duty cycle
#define    signalInvert                                              2
/// **3** - Signal get/set period in nanoseconds.
#define    signalT3Time                                              3
/// **4** - Signal get/set active time in nanoseconds (See reference).
#define    signalT2Time                                              4

/** @} */
/** @} */
/////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////
/** \defgroup cmdEQUALIZER Equalizer command defines
 * Equalizer entity defines.
 *
 * @{
 */

/// **14** - cmdSIGNAL command code.
#define cmdEQUALIZER                                             15

/** \defgroup cmdEQUALIZER_Command_Options Equalizer signal command options
 * \ingroup cmdEQUALIZER_Defines
 *
 * @{
 */
/// **1** - Equalizer receiver config
#define    equalizerReceiverConfig                                1
/// **2** - Equalizer driver config
#define    equalizerTransmitterConfig                             2
/// **3** - Equalizer manual configuration
#define    equalizerManualConfig                                  3


/** @} */
/** @} */
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/// **0** - Equalizer for USB 2.0
#define    equalizer2p0                                           0
/// **1** - Equalizer for USB 3.0
#define    equalizer3p0                                           1

/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////

/// Debug command
#define cmdDEBUG                                                   23

/// app NOTIFY currently resverve for future use.
#define cmdNOTIFY                                                  24

/////////////////////////////////////////////////////////////////////

/** \defgroup cmdANALOG_Defines Analog Command defines
 * Analog Entity defines.
 *
 * @{
 */

/// **30** - Analog command code.
#define cmdANALOG                                                  30

/** \defgroup cmdANALOG_Command_Options Analog Command options
 * \ingroup cmdANALOG_Defines
 *
 * @{
 */

/// **1** - Analog configuration option code.
#define    analogConfiguration                                      1
/// **0** - Input configuration for configuration option code.
#define      analogConfigurationInput                               0
/// **1** - Output configuration for configuration option code.
#define      analogConfigurationOutput                              1
/// **2** - HiZ configuration for configuration option code.
#define      analogConfigurationHiZ                                 2
/// **2** - Analog Value option code.
#define    analogValue                                              2
/// **3** - Analog Voltage option code.
#define    analogVoltage                                            3
/// **4** - Analog Bulk Capture Sample Rate option code.
#define    analogBulkCaptureSampleRate                              4
/// **7000** - minimum hertz sample rate for Bulk capture Sample Rate option code.
#define    analog_Hz_Minimum                                     7000
/// **200000** - maximum hertz sample rate for Bulk capture Sample Rate option code.
#define    analog_Hz_Maximum                                   200000
/// **5** - Bulk Capture number of samples option code.
#define    analogBulkCaptureNumberOfSamples                         5
/// **6** - Bulk Capture option code.
#define    analogBulkCapture                                        6
/// **7** - Bulk Capture State option code.
#define    analogBulkCaptureState                                   7
/// **0** - Idle state for Bulk Capture state option code.
#define      bulkCaptureIdle                                        0
/// **1** - Pending state for Bulk Capture state option code.
#define      bulkCapturePending                                     1
/// **2** - Finished state for Bulk Capture state option code.
#define      bulkCaptureFinished                                    2
/// **3** - Error state for Bulk Capture state option code.
#define      bulkCaptureError                                       3
/// **8** - Analog Range option code.
#define    analogRange                                              8
/// **0** - +/- 64mV range for Analog Range option code.
#define      analogRange_P0V064N0V064                               0
/// **1** - +/- 640mV range for Analog Range option code.
#define      analogRange_P0V64N0V64                                 1
/// **2** - +/- 128mV range for Analog Range option code.
#define      analogRange_P0V128N0V128                               2
/// **3** - +/- 1.28V range for Analog Range option code.
#define      analogRange_P1V28N1V28                                 3
/// **4** - 0-1.28V range for Analog Range option code.
#define      analogRange_P1V28N0V0                                  4
/// **5** - +/- 256mV range for Analog Range option code.
#define      analogRange_P0V256N0V256                               5
/// **6** - +/- 2.56V range for Analog Range option code.
#define      analogRange_P2V56N2V56                                 6
/// **7** - 0-2.56V range for Analog Range option code.
#define      analogRange_P2V56N0V0                                  7
/// **8** - +/- 512mV range for Analog Range option code.
#define      analogRange_P0V512N0V512                               8
/// **9** - +/- 5.12V range for Analog Range option code.
#define      analogRange_P5V12N5V12                                 9
/// **10** - 0-5.12V range for Analog Range option code.
#define      analogRange_P5V12N0V0                                 10
/// **11** - +/- 1.024V range for Analog Range option code.
#define      analogRange_P1V024N1V024                              11
/// **12** - +/- 10.24V range for Analog Range option code.
#define      analogRange_P10V24N10V24                              12
/// **13** - 0-10.24V range for Analog Range option code.
#define      analogRange_P10V24N0V0                                13
/// **14** - 0-2.048V range for Analog Range option code.
#define      analogRange_P2V048N0V0                                14
/// **15** - 0-4.096V range for Analog Range option code.
#define      analogRange_P4V096N0V0                                15
/// **9** - Analog Enable option code.
#define    analogEnable                                             9
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////

/** \defgroup cmdDIGITAL_Defines Digital command defines
 * Digital entity defines.
 *
 * @{
 */

/// **31** - Digital command code.
#define cmdDIGITAL                                                 31

/** \defgroup cmdDIGITAL_Command_Options Digital command options
 * \ingroup cmdDIGITAL_Defines
 *
 * @{
 */
/// **1** - Digital configuration option code.
#define    digitalConfiguration                                     1
/// **0** - Input Digital configuration for configuration option code.
#define        digitalConfigurationInput                         0x00
/// **1** - Output Digital configuration for configuration option code.
#define        digitalConfigurationOutput                        0x01
/// **2** - RC Servo Input Digital configuration for configuration option code.
#define        digitalConfigurationRCServoInput                  0x02
/// **3** - RC Servo Output Digital configuration for configuration option code.
#define        digitalConfigurationRCServoOutput                 0x03
/// **4** - Hi Z the digital pin.
#define        digitalConfigurationHiZ                           0x04
/// **0** - Input digital configuration with pull-up.
#define        digitalConfigurationInputPullUp                   0x00
/// **4** - Input digital configuration with no pull-up/pull-down.
#define        digitalConfigurationInputNoPull                   0x04
/// **5** - Input digital configuration with pull-down.
#define        digitalConfigurationInputPullDown                 0x05
/// **6** - Signal output configuration
#define        digitalConfigurationSignalOutput                  0x06
/// **7** - Signal input configuration
#define        digitalConfigurationSignalInput                   0x07
/// **8** - Signal input conter configuration
#define        digitalConfigurationSignalCounterInput            0x08
/// **9** - State option code.
#define    digitalState                                             2
#define    digitalStateAll                                          3
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////


/** \defgroup cmdRAIL_Defines Rail command defines
 * Rail entity defines.
 *
 * @{
 */

/// **32** - Rail command code.
#define cmdRAIL                                                    32

/** \defgroup cmdRAIL_Command_Options Rail command options
 * \ingroup cmdRAIL_Defines
 *
 * @{
 */

/// **1** - Rail Voltage option code.
#define    railVoltage                                              1
/// **11** - Rail Setpoint Voltage option code
#define    railVoltageSetpoint                                     11
/// **2** - Rail Current option code.
#define    railCurrent                                              2
/// **3** - Rail Current limit option code.
#define    railCurrentLimit                                         3
/// **4** - Rail Temperature option code.
#define    railTemperature                                          4
/// **5** - Rail Enable option code.
#define    railEnable                                               5
/// **6** - Rail Value option code.
#define    railValue                                                6
/// **7** - Rail Kelvin sensing Mode option code.
#define    railKelvinSensingEnable                                  7
/// **0** - Kelvin Sensing off mode for Kelvin Sensing mode option code.
#define       kelvinSensingOff                                      0
/// **1** - Kelvin Sensing on mode for Kelvin Sensing mode option code.
#define       kelvinSensingOn                                       1
/// **8** - Kelving Sensing state option code.
#define    railKelvinSensingState                                   8
/// **9** - Operational mode option code.
#define    railOperationalMode                                      9
/// **0** - Auto operational mode for operational mode option code.
#define       operationalModeAuto                                   0
/// **1** - Linear mode for operational mode option code.
#define       operationalModeLinear                                 1
/// **2** - Switcher mode for operational mode option code.
#define       operationalModeSwitcher                               2
/// **3** - Switcher Linerar mode for operational mode option code.
#define       operationalModeSwitcherLinear                         3
/// **operationalModeAuto** - Default operational mode for operational mode option code.
#define       DefaultOperationalRailMode          operationalModeAuto
/// **10** - Operational state option code.
#define    railOperationalState                                    10
/// **0** - Initializing state for operational state option code.
#define       operationalState_Initializing                         0
/// **1** - PowerGood state for operational state option code.
#define       operationalState_PowerGood                            1
/// **2** - PowerFault state for operational state option code.
#define       operationalState_PowerFault                           2
/// **3** - LDO over temp state for operational state option code.
#define       operationalState_LDOOverTemp                          3
/// **4** - Linear state for operational state option mode.
#define       operationalStateLinear                                4
/// **5** - Switcher state for operational state option mode.
#define       operationalStateSwitcher                              5
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////
/** \defgroup cmdTEMPERATURE_Defines Temperature command defines
 * Temperature entity defines.
 *
 * @{
 */

/// **33** - Temperature command code.
#define cmdTEMPERATURE                                             33

/** \defgroup cmdTEMPERATURE_Command_Options Temperature command options
 * \ingroup cmdTEMPERATURE_Defines
 *
 * @{
 */

/// **1** - Temperature option code.
#define    temperatureMicroCelsius                                  1
/** @} */
/** @} */
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/** \defgroup cmdRELAY_Defines Relay command defines
 * Relay entity defines.
 *
 * @{
 */

/// **34** - Relay command code.
#define cmdRELAY                                                    34

/** \defgroup cmdRELAY_Command_Options Relay command options
 * \ingroup cmdRELAY_Defines
 *
 * @{
 */
/// **1** - Relay State option code.
#define    relayEnable                                               1
/// **2** - Relay Voltage option code.
#define    relayVoltage                                              2
/** @} */
/** @} */
/////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////
/** \defgroup cmdUART_Defines UART command defines
 * UART entity defines.
 *
 * @{
 */
/// **35** - UART command code.
#define cmdUART                                                     35

/** \defgroup cmdUART_Command_Options UART command options
 * \ingroup cmdUART_Defines
 *
 * @{
 */

/// **1** - UART Channel Enable code
#define     uartEnable                                               1
/** @} */
/** @} */
/////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////

/** \defgroup cmdCAPACITY_Defines Capacity command defines
 * Capacity command.
 *
 * @{
 */
/// **73** - Capacity command code.
#define cmdCAPACITY                                                73

/** \defgroup cmdCAPACITY_Command_Options Capacity command options
 * \ingroup cmdCAPACITY_Defines
 *
 * @{
 */
/// **1** - UEI command option.
#define    capacityUEI                                              1
/// **3** - SubClass size command option.
#define    capacitySubClassSize                                     3
/// **4** - Class Quantity command option.
#define    capacityClassQuantity                                    4
/// **5** - SubClass Quantity command option.
#define    capacitySubClassQuantity                                 5
/// **6** - Entity Group command option.
#define    capacityEntityGroup                                      6
/// **7** - Build command option.
#define    capacityBuild                                          255
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////

/** \defgroup cmdSTORE_Defines Store command defines
 * Store entity defines.
 *
 * @{
 */
/// **77** - Store command code.
#define cmdSTORE                                                  77

/** \defgroup cmdSTORE_Command_Options Store command options
 * \ingroup cmdSTORE_Defines
 *
 * @{
 */
/// **1** - Slot Enable option code.
#define	   storeSlotEnable                                         1
/// **2** - Slot Disable option code.
#define	   storeSlotDisable                                        2
/// **3** - Slot State option code.
#define	   storeSlotState                                          3
/// **4** - Write Slot option code.
#define	   storeWriteSlot                                          4
/// **5** - Read Slot option code.
#define	   storeReadSlot                                           5
/// **6** - Close Slot option code.
#define	   storeCloseSlot                                          6
/** @} */

/** \defgroup cmdSTORE_Types Store command options
 * \ingroup cmdSTORE_Defines
 *
 * @{
 */

/////////////////////////////////////////////////////////////////////
/// **0** - Internal store type.
#define    storeInternalStore		                                0
/// **1** - RAM store type.
#define    storeRAMStore                                            1
/// **2** - SD Store type.
#define    storeSDStore	                                            2
/// **2** - Max type index.
#define    storeMaxStoreIndex                                       2
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////

/** \defgroup cmdTIMER_Defines Timer command options
 * Timer Entity Defines.
 *
 * @{
 */
/// **79** - Timer command code.
#define cmdTIMER                                                    79

/** \defgroup cmdTIMER_Command_Options Timer command options
 * \ingroup cmdTIMER_Defines
 *
 * @{
 */
/// **1** - Timer expiration option code.
#define    timerExpiration                                          1
/// **2** - Timer Mode option code.
#define    timerMode                                                2
/// **0** - Single mode for timer mode option code.
#define       timerModeSingle                                       0
/// **1** - Repeat mode for timer mode option code.
#define       timerModeRepeat                                       1
/// **timerModeSingle** - Default mode for timer mode option code.
#define       DefaultTimerMode                        timerModeSingle
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////

/** \defgroup cmdCLOCK_Defines Clock command defines
 * Clock entity defines.
 *
 * @{
 */

/// **83** - Clock command code.
#define cmdCLOCK                                                   83

/** \defgroup cmdCLOCK_Command_Options Clock command options
 * \ingroup cmdCLOCK_Defines
 *
 * @{
 */
/// **1** - Year option code.
#define    clockYear                                                1
/// **2** - Month option code.
#define    clockMonth                                               2
/// **3** - Day option code.
#define    clockDay                                                 3
/// **4** - Hour option code.
#define    clockHour                                                4
/// **5** - Minute option code.
#define    clockMinute                                              5
/// **6** - Second option code.
#define    clockSecond                                              6
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////

/** \defgroup cmdUSB_Defines USB c ommand defines
 * USB entity defines.
 *
 * @{
 */
/// **18** - USB command code.
#define cmdUSB                                                     18

/** \defgroup cmdUSB_Command_Options USB command options
 * \ingroup cmdUSB_Defines
 *
 * @{
 */
/// **1** - Port Enable option code.
#define    usbPortEnable                                            1
/// **2** - Port Disable option code.
#define    usbPortDisable                                           2
/// **3** - Data Enable option code.
#define    usbDataEnable                                            3
/// **4** - Data Disable option code.
#define    usbDataDisable                                           4
/// **5** - Power Enable option code.
#define    usbPowerEnable                                           5
/// **6** - Power Disable option code.
#define    usbPowerDisable                                          6
/// **7** - Port Current option code.
#define    usbPortCurrent                                           7
/// **8** - Port Voltage option code.
#define    usbPortVoltage                                           8
/// **9** - Hub Mode option code.
#define    usbHubMode                                               9
// Option codes 10 and 11 are reserved.
/// **12** - Hub Clear Error Status option code.
#define    usbPortClearErrorStatus                                 12
/// **13** - SystemTemperature option code.
// Option code 13 is reserved.
#define    usbUpstreamMode                                         14
/// **2** - UpstreamMode Auto for upstream mode option code.
#define         usbUpstreamModeAuto                                 2
/// **0** - UpstreamMode Port 0 for upstream mode option code.
#define         usbUpstreamModePort0                                0
/// **1** - UpstreamMode Port 1 for upstream mode option code.
#define         usbUpstreamModePort1                                1
/// **255** - UpstreamMode None to turn off all upstream connections.
#define         usbUpstreamModeNone                               255
/// **1** - UpstreamMode default for upstream mode option code.
#define         usbUpstreamModeDefault             usbUpstreamModeAuto
/// **15** - UpstreamState option code.
#define    usbUpstreamState                                         15
/// **2** - UpstreamMode Auto for upstream mode option code.
#define         usbUpstreamStateNone                                 2
/// **0** - UpstreamMode Port 0 for upstream mode option code.
#define         usbUpstreamStatePort0                                0
/// **1** - UpstreamMode Port 1 for upstream mode option code.
#define         usbUpstreamStatePort1                                1
/// **16** - Downstream ports enumeration delay option code.
#define    usbHubEnumerationDelay                                   16
/// **17** - Set or get the port current limit option code.
#define    usbPortCurrentLimit                                      17
/// **18** - Set/Get upstream boost mode.
#define    usbUpstreamBoostMode                                     18
/// **19** - Set/Get downstream boost mode.
#define    usbDownstreamBoostMode                                   19
/// **0** - Boost mode off, no boost
#define        usbBoostMode_0                                       0
/// **1** - Boost mode 4%
#define        usbBoostMode_4                                       1
/// **2** - Boost mode 8%
#define        usbBoostMode_8                                       2
/// **3** - Boost mode 12%
#define        usbBoostMode_12                                      3
/// **20** - Set/Get Port mode (bit-packed)
///  The portMode bits follow and numbered according to their bit position.
///  if they are set i.e. a 1 in the bit position the corresponding setting
///  is enabled.
#define    usbPortMode                                              20
/// **0** - Standard Downstream port (0.5 Amp)
#define        usbPortMode_sdp                                      0
/// **1** - Charging Downstream port (2.1 Amp)
#define        usbPortMode_cdp                                      1
/// **2** - Trickle changing functionality
#define        usbPortMode_charging                                 2
/// **3** - Electrical pasthrough of VBUS
#define        usbPortMode_passive                                  3
/// **4** - USB2 dataline A side enabled
#define        usbPortMode_USB2AEnable                              4
/// **4** - USB2 dataline B side enabled
#define        usbPortMode_USB2BEnable                              5
/// **5** - USB VBUS enabled
#define        usbPortMode_VBusEnable                               6
/// **6** - USB SS Speed dataline side A enabled
#define        usbPortMode_SuperSpeed1Enable                        7
/// **7** - USB SS Speed dataline side B enabled
#define        usbPortMode_SuperSpeed2Enable                        8
/// **8** - USB2 Boost Mode Enabled
#define        usbPortMode_USB2BoostEnable                          9
/// **9** - USB3 Boost Mode Enabled
#define        usbPortMode_USB3BoostEnable                          10
/// **10** - Auto-connect Mode Enabled
#define        usbPortMode_AutoConnectEnable                        11
/// **11** - CC1 Enabled
#define        usbPortMode_CC1Enable                                12
/// **12** - CC2 Enabled
#define        usbPortMode_CC2Enable                                13
/// **13** - SBU1 Enabled
#define        usbPortMode_SBUEnable                                14
/// **15** - Flip CC1 and CC2
#define        usbPortMode_CCFlipEnable                             15
/// **16** - Flip Super speed data lines
#define        usbPortMode_SSFlipEnable                             16
/// **17** - Flip Side Band Unit lines.
#define        usbPortMode_SBUFlipEnable                            17
/// **18** - Flip Side Band Unit lines.
#define        usbPortMode_USB2FlipEnable                           18
/// **19** - Internal Use
#define        usbPortMode_CC1InjectEnable                          19
/// **20** - Internal Use
#define        usbPortMode_CC2InjectEnable                          20
/// **21** - Hi-Speed Data Enable option code.
#define    usbHiSpeedDataEnable                                     21
/// **22** - Hi-Speed Data Disable option code.
#define    usbHiSpeedDataDisable                                    22
/// **23** - SuperSpeed Data Enable option code.
#define    usbSuperSpeedDataEnable                                  23
/// **24** -SuperSpeed Data Disable option code.
#define    usbSuperSpeedDataDisable                                 24
/// **25** - Get downstream port speed option code.
#define    usbDownstreamDataSpeed                                   25
/// **0** - Unknown
#define        usbDownstreamDataSpeed_na                            0
/// **1** - Hi-Speed (2.0)
#define        usbDownstreamDataSpeed_hs                            1
/// **2** - SuperSpeed (3.0)
#define        usbDownstreamDataSpeed_ss                            2
/// **3** - TODO
#define        usbDownstreamDataSpeed_ls                            3
/// **26** USB connect mode option code
#define    usbConnectMode                                           26
/// **0** - Auto connect disabled
#define        usbManualConnect                                     0
/// **1** - Auto connect enabled
#define        usbAutoConnect                                       1
/// **27** - CC1 Enable option code (USB Type C).
#define    usbCC1Enable                                             27
/// **28** - CC2 Disable option code (USB Type C).
#define    usbCC2Enable                                             28
/// **29** - SBU1/2 enable option code (USB Type C).
#define    usbSBUEnable                                             29
/// **30** - CC1 get current option code (USB Type C).
#define    usbCC1Current                                            30
/// **31** - CC2 get current option code (USB Type C).
#define    usbCC2Current                                            31
/// **32** - CC1 get voltage option code (USB Type C).
#define    usbCC1Voltage                                            32
/// **33** - CC2 get voltage option code (USB Type C).
#define    usbCC2Voltage                                            33
/// **34** - TODO
#define    usbPortState                                             34
/// **35** - TODO
#define    usbPortError                                             35
/// **36** - TODO
#define    usbCableFlip                                             36
/// **37** - USB Alt Mode configuration.
#define    usbAltMode                                               37
/// **0** - Disabled mode
#define        usbAltMode_disabled                                  0
/// **1** - Normal mode (USB 3.1)
#define        usbAltMode_normal                                    1
/// **2** - Alt Mode - 4 lanes of display port "Common" side connected to host
#define        usbAltMode_4LaneDP_ComToHost                         2
/// **3** - Alt Mode - 4 lanes of display port "Mux" side connected to host
#define        usbAltMode_4LaneDP_MuxToHost                         3
/// **4** - Alt Mode - 2 lanes of display port "Common" side connected to host with USB3.1
#define        usbAltMode_2LaneDP_ComToHost_wUSB3                   4
/// **5** - Alt Mode - 2 lanes of display port "Mux" side connected to host with USB3.1
#define        usbAltMode_2LaneDP_MuxToHost_wUSB3                   5
/** @} */
/** @} */

/////////////////////////////////////////////////////////////////////

/// Upgrade command.
#define cmdUPGRADE                                                 95


/////////////////////////////////////////////////////////////////////

/// Last command.
#define cmdLAST                                                    95

#endif //_aProtocolDefs_H_
