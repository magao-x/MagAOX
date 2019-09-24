/////////////////////////////////////////////////////////////////////
//                                                                 //
// file: BrainStem-entity.h                                        //
//                                                                 //
/////////////////////////////////////////////////////////////////////
//                                                                 //
// description: BrainStem API's and support.                       //
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

#ifndef __BrainStem_entity_H__
#define __BrainStem_entity_H__
#include "BrainStem-core.h"

#ifdef __GNUC__
#define DEPRECATED __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED __declspec(deprecated)
#else
#define DEPRECATED
#pragma message("DEPRECATED is not defined for this compiler")
#endif

namespace Acroname {
    namespace BrainStem {


        // MARK:  - App Class
        /////////////////////////////////////////////////////////////////////
        /// AppClass. Used to send a cmAPP packet to the BrainStem network.
        /// These commands are used for either host-to-stem or stem-to-stem interactions.
        /// BrainStem modules can implement a reflex origin to complete an action when
        /// a cmdAPP packet is addressed to the module.
        class aLIBEXPORT AppClass : public EntityClass
        {
        public:
            /// Constructor.
            AppClass(void);

            /// Destructor.
            ~AppClass(void);

            /// Initialize the class.
            /// \param pModule The module.
            /// \param index The cmdAPP reflex index to be addressed.
            void init(Module* pModule,
                      const uint8_t index);

            /// Execute the app reflex on the module. Don't wait for a return
            /// value from the execute call; this call returns immediately upon execution
            /// of the module's reflex.
            /// \param appParam The app parameter handed to the reflex.
            /// \return ::aErrNone - success.
            /// \return ::aErrTimeout - The request timed out waiting to start execution.
            /// \return ::aErrConnection - No active link connection.
            /// \return ::aErrNotFound - the app reflex was not found or not enabled on
            ///                        the module.
            aErr execute(const uint32_t appParam);

            /// Execute the app reflex on the module. Wait for a return from the
            /// reflex execution for msTimoue milliseconds. This method will block for
            /// up to msTimeout.
            /// \param appParam The app parameter handed to the reflex.
            /// \param returnVal The return value filled in from the result of
            ///                  executing the reflex routine.
            /// \param msTimeout The amount of time to wait for the return value
            ///                  from the reflex routine. The default value is 1000
            ///                  milliseconds if not specified.
            /// \return ::aErrNone - success.
            /// \return ::aErrTimeout - The request timed out waiting for a response.
            /// \return ::aErrConnection - No active link connection.
            /// \return ::aErrNotFound - the app reflex was not found or not enabled on
            ///                        the module.
            aErr execute(const uint32_t appParam,
                         uint32_t* returnVal,
                         const uint32_t msTimeout = 1000);

        };


        // MARK:  - Analog Class
        /////////////////////////////////////////////////////////////////////
        /// AnalogClass. Interface to analog entities on BrainStem modules.
        /// Analog entities may be configured as a input or output depending
        /// on hardware capabilities. Some modules are capable of providing actual
        /// voltage readings, while other simply return the raw analog-to-digital converter (ADC)
        /// output value. The resolution of the voltage or number of useful bits is also
        /// hardware dependent.
        class aLIBEXPORT AnalogClass : public EntityClass
        {
        public:
            /// Constructor.
            AnalogClass(void);

            /// Destructor.
            ~AnalogClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the analog entity being initialized.
            void init(Module* pModule,
                      const uint8_t index);

            /// Get the raw ADC output value in bits.
            /// \param value 16 bit analog reading with 0 corresponding to the negative
            ///  analog voltage reference and
            /// 0xFFFF corresponding to the positive analog voltage reference.
            /// \note Not all modules are provide 16 useful bits; this value's least
            /// significant bits are zero-padded to 16 bits. Refer to the module's
            /// datasheet to determine analog bit depth and reference voltage.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getValue(uint16_t* value);

            /// Get the scaled micro volt value with refrence to ground.
            /// \param microVolts 32 bit signed integer (in microVolts) based on the board's
            /// ground and reference voltages.
            /// \note Not all modules provide 32 bits of accuracy; Refer to the module's
            /// datasheet to determine the analog bit depth and reference voltage.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getVoltage(int32_t* microVolts);

            /// Get the analog input range.
            /// \param range 8 bit value corresponding to a discrete range option
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getRange(uint8_t* range);

            /// Get the analog output enable status.
            /// \param enable 0 if disabled 1 if enabled.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getEnable(uint8_t* enable);

            /// Set the value of an analog output (DAC) in bits.
            /// \param value 16 bit analog set point with 0 corresponding to the negative
            /// analog voltage reference  and
            /// 0xFFFF corresponding to the positive analog voltage reference.
            /// \note Not all modules are provide 16 useful bits; the least significant bits
            /// are discarded. E.g. for a 10 bit DAC, 0xFFC0 to 0x0040 is the useful range.
            /// Refer to the module's
            /// datasheet to determine analog bit depth and reference voltage.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setValue(const uint16_t value);

            /// Set the voltage level of an analog output (DAC) in microVolts.
            /// \param microVolts 32 bit signed integer (in microVolts) based
            /// on the board's ground and reference voltages.
            /// \note Voltage range is dependent on the specific DAC channel
            /// range.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setVoltage(const int32_t microVolts);

            /// Set the analog input range.
            /// \param range 8 bit value corresponding to a discrete range option
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setRange(const uint8_t range);

            /// Set the analog output enable state.
            /// \param enable set 1 to enable or 0 to disable.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setEnable(const uint8_t enable);

            /// Set the analog configuration.
            /// \param configuration - bitAnalogConfigurationOutput configures the
            ///                        analog entity as an output.
            /// \returns EntityReturnValues "common entity" return values
            /// \retval aErrConfiguration - Entity does not support this configuration.
            aErr setConfiguration(const uint8_t configuration);

            /// Get the analog configuration.
            /// \param configuration - Current configuration of the analog entity.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getConfiguration(uint8_t* configuration);

            /// Set the sample rate for this analog when bulk capturing.
            /// \param value sample rate in samples per second (Hertz).
            /// Minimum rate: 7,000 Hz
            /// Maximum rate: 200,000 Hz
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setBulkCaptureSampleRate(const uint32_t value);

            /// Get the current sample rate setting for this analog when bulk capturing.
            /// \param value upon success filled with current sample rate in samples per second (Hertz).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getBulkCaptureSampleRate(uint32_t* value);

            /// Set the number of samples to capture for this analog when bulk capturing.
            /// \param value number of samples.
            /// Minimum # of Samples: 0
            /// Maximum # of Samples: (BRAINSTEM_RAM_SLOT_SIZE / 2) = (3FFF / 2) = 1FFF = 8191
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setBulkCaptureNumberOfSamples(const uint32_t value);

            /// get the current number of samples setting for this analog when bulk capturing.
            /// \param value number of samples.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getBulkCaptureNumberOfSamples(uint32_t* value);

            /// Initiate a BulkCapture on this analog. Captured measurements are stored in the
            /// module's RAM store (RAM_STORE) slot 0. Data is stored in a contiguous byte array
            /// with each sample stored in two consecutive bytes, LSB first.
            /// \return Returns \ref EntityReturnValues "common entity" return values. When the
            /// bulk capture is complete getBulkCaptureState() will return either
            /// bulkCaptureFinished or bulkCaptureError.
            aErr initiateBulkCapture(void);

            /// get the current bulk capture state for this analog.
            /// \param state the state of bulk capture.
            /// - Idle: bulkCaptureIdle = 0
            /// - Pending: bulkCapturePending = 1
            /// - Finished: bulkCaptureFinished = 2
            /// - Error: bulkCaptureError = 3
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getBulkCaptureState(uint8_t* state);

        };


        // MARK:  - Clock Class
        /////////////////////////////////////////////////////////////////////
        /// ClockClass.
        /// Provides an interface to a real-time clock entity on a BrainStem module.
        /// The clock entity may be used to get and set the real time of the system.
        /// The clock entity has a one second resolution.
        /// @note Clock time must be reset if power to the BrainStem module is lost.
        class aLIBEXPORT ClockClass : public EntityClass
        {
        public:
            /// Constructor.
            ClockClass(void);

            /// Destructor.
            virtual ~ClockClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the clock entity being initialized.
            void init(Module* pModule,
                      const uint8_t index);

            /// Get the four digit year value (0-4095).
            /// \param year Get the year portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getYear(uint16_t* year);

            /// Set the four digit year value (0-4095).
            /// \param year Set the year portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setYear(const uint16_t year);

            /// Get the two digit month value (1-12).
            /// \param month The two digit month portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getMonth(uint8_t* month);

            /// Set the two digit month value (1-12).
            /// \param month The two digit month portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setMonth(const uint8_t month);

            /// Get the two digit day of month value (1-28, 29, 30 or 31 depending
            /// on the month).
            /// \param day The two digit day portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getDay(uint8_t* day);

            /// Get the two digit day of month value (1-28, 29, 30 or 31 depending
            /// on the month).
            /// \param day The two digit day portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setDay(const uint8_t day);

            /// Get the two digit hour value (0-23).
            /// \param hour The two digit hour portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getHour(uint8_t* hour);

            /// Set the two digit hour value (0-23).
            /// \param hour The two digit hour portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setHour(const uint8_t hour);

            /// Get the two digit minute value (0-59).
            /// \param min The two digit minute portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getMinute(uint8_t* min);

            /// Set the two digit minute value (0-59).
            /// \param min The two digit minute portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setMinute(const uint8_t min);

            /// Get the two digit second value (0-59).
            /// \param sec The two digit second portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getSecond(uint8_t* sec);

            /// Set the two digit second value (0-59).
            /// \param sec The two digit second portion of the real-time clock value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setSecond(const uint8_t sec);
        };

        // MARK:  - Digital Class
        /////////////////////////////////////////////////////////////////////
        /// DigitalClass. Interface to digital entities on BrainStem modules.
        /// Digital entities have the following 5 possabilities: Digital Input,
        /// Digital Output, RCServo Input, RCServo Output, and HighZ.
        /// Other capabilities may be available and not all pins support all
        /// configurations. Please see the product datasheet.
        class aLIBEXPORT DigitalClass : public EntityClass
        {
        public:
            /// Constructor.
            DigitalClass(void);

            /// Destructor.
            virtual ~DigitalClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the digital entity being initialized.
            void init(Module* pModule,
                      const uint8_t index);

            /// Set the digital configuration to one of the avaliable 5 states.
            /// Note: Some configurations are only supported on specific pins.
            /// \param configuration
            /// - Digital Input: digitalConfigurationInput = 0
            /// - Digital Output: digitalConfigurationOutput = 1
            /// - RCServo Input: digitalConfigurationRCServoInput = 2
            /// - RCServo Output: digitalConfigurationRCServoOutput = 3
            /// - High Z State: digitalConfigurationHiZ = 4
            /// - Digital Input: digitalConfigurationInputPullUp = 0
            /// - Digital Input: digitalConfigurationInputNoPull = 4
            /// - Digital Input: digitalConfigurationInputPullDown = 5
            /// \return Returns \ref EntityReturnValues "common entity" return values
            /// \return ::aErrConfiguration - Entity does not support this configuration.
            aErr setConfiguration(const uint8_t configuration);

            /// Get the digital configuration.
            /// \param configuration - Current configuration of the digital entity.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getConfiguration(uint8_t* configuration);

            /// Set the logical state.
            /// \param state The state to be set. 0 is logic low, 1 is logic high.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setState(const uint8_t state);
            
            /// Get the state.
            /// \param state The current state of the digital entity. 0 is logic low,
            /// 1 is logic high. Note: If in high Z state an error will be returned.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getState(uint8_t* state);
            
            /// Sets the logical state of all available digitals based on the bit mapping.
            /// Number of digitals varies across BrainStem modules.  Refer
            /// to the datasheet for the capabilities  of your module.
            /// \param state The state to be set for all digitals in a bit mapped representation.
            /// 0 is logic low, 1 is logic high. Where bit 0 = digital 0, bit 1 = digital 1 etc.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setStateAll(const uint32_t state);
            
            /// Gets the logical state of all available digitals in a bit mapped representation.
            /// Number of digitals varies across BrainStem modules.  Refer
            /// to the datasheet for the capabilities  of your module.
            /// \param state The state of all digitals where bit 0 = digital 0,
            /// bit 1 = digital 1 etc. 0 is logic low, 1 is logic high.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getStateAll(uint32_t* state);

        };


        // MARK:  - RCServo Class
        /////////////////////////////////////////////////////////////////////
        /// RCServoClass. Interface to servo entities on BrainStem modules.
        /// Servo entities are built upon the digital input/output pins and therefore
        /// can also be inputs or outputs. Please see the product datasheet on the
        /// configuration limitations.
        class aLIBEXPORT RCServoClass : public EntityClass
        {
        public:
            /// Constructor.
            RCServoClass(void);

            /// Destructor.
            virtual ~RCServoClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the servo entity being initialized.
            void init(Module* pModule,
                      const uint8_t index);

            /// Enable the servo channel
            /// \param enable The state to be set. 0 is disabled, 1 is enabled.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setEnable(const uint8_t enable);

            /// Get the enable status of the servo channel.
            /// \param enable The current enable status of the servo entity. 0 is disabled,
            /// 1 is enabled.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getEnable(uint8_t* enable);

            /// Set the position of the servo channel
            /// \param position The position to be set. Default 64 = a 1ms pulse and
            /// 192 = a 2ms pulse.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setPosition(const uint8_t position);

            /// Get the position of the servo channel
            /// \param position The current position of the servo channel. Default
            /// 64 = a 1ms pulse and 192 = a 2ms pulse.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getPosition(uint8_t* position);

//            /// Get the state.
//            /// \param state The current state of the digital entity. 0 is logic low,
//            /// 1 is logic high.
//            /// \return Returns \ref EntityReturnValues "common entity" return values
//            aErr setRange(const uint8_t min, const uint8_t max, const uint8_t trim);
//
//            /// Get the state.
//            /// \param state The current state of the digital entity. 0 is logic low,
//            /// 1 is logic high.
//            /// \return Returns \ref EntityReturnValues "common entity" return values
//            aErr getRange(uint8_t* min, uint8_t* max, uint8_t* trim);

            /// Set the output to be reversed on the servo channel
            /// \param reverse Reverses the value set by "setPosition".  ie. if the
            /// position is set to 64 (1ms pulse) the output will now be 192 (2ms pulse);
            /// however, "getPostion" will return the set value of 64. 0 = not reversed,
            /// 1 = reversed.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setReverse(const uint8_t reverse);

            /// Get the reverse status of the servo channel
            /// \param reverse The current reverse status of the servo entity. 0 = not
            /// reversed, 1 = reversed.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getReverse(uint8_t* reverse);

        };


        // MARK:  - I2C Class
        /////////////////////////////////////////////////////////////////////
        /// I2CClass. Interface the I2C busses on BrainStem modules.
        /// The class provides a way to send read and write commands to I2C devices
        /// on the entitie's bus.
        class aLIBEXPORT I2CClass : public EntityClass
        {
        public:
            /// Constructor.
            I2CClass(void);

            /// Destructor.
            virtual ~I2CClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the digital entity being initialized.
            void init(Module* pModule,
                      const uint8_t index);

            /// Read from a device on this I2C bus.
            /// \param address - The I2C address (7bit <XXXX-XXX0>) of the device to read.
            /// \param length - The length of the data to read in bytes.
            /// \param result - The array of bytes that will be filled with the result, upon success.
            ///                 This array should be larger or equivalent to aBRAINSTEM_MAXPACKETBYTES - 5

            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr read(const uint8_t address, const uint8_t length, uint8_t* result);

            /// Write to a device on this I2C bus.
            /// \param address - The I2C address (7bit <XXXX-XXX0>) of the device to read.
            /// \param length - The length of the data to read in bytes.
            /// \param data - The data to send to the device, This array should be no larger than
            ///                aBRAINSTEM_MAXPACKETBYTES - 5
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr write(const uint8_t address, const uint8_t length, const uint8_t* data);

            /// Set bus pullup state.
            /// This call only works with stems that have software controlled pullups.
            /// Check the datasheet for more information. This parameter is saved when
            /// system.save is called.
            ///
            /// \param bEnable - true enables pullups false disables them.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setPullup(const bool bEnable);


            /// Set I2C bus speed.
            ///
            /// This call sets the communication speed for I2C transactions through
            /// this API. Speed is an enumeration value which can take the following
            /// values.
            ///     1 - 100Khz
            ///     2 - 400Khz
            ///     3 - 1MHz
            ///
            /// \param speed - The speed setting value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setSpeed(const uint8_t speed);


            /// Get I2C bus speed.
            ///
            /// This call gets the communication speed for I2C transactions through
            /// this API. Speed is an enumeration value which can take the following
            /// values.
            ///     1 - 100Khz
            ///     2 - 400Khz
            ///     3 - 1MHz
            ///
            /// \param speed - The speed setting value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getSpeed(uint8_t* speed);


        private:

            uint8_t _busSpeed;

        };


        // MARK:  - Mux Class
        /////////////////////////////////////////////////////////////////////
        /// MuxClass.
        /// A MUX is a multiplexer that takes one or more similar inputs
        /// (bus, connection, or signal) and allows switching to one or more outputs.
        /// An analogy would be the switchboard of a telephone operator.  Calls (inputs)
        /// come in and by re-connecting the input to an output, the operator
        /// (multiplexor) can direct that input to on or more outputs.
        ///
        /// One possible output is to not connect the input to anything which
        /// essentially disables that input's connection to anything.
        ///
        /// Not every MUX has multiple inputs.  Some may simply be a single input that
        /// can be enabled (connected to a single output) or disabled
        /// (not connected to anything).
        class aLIBEXPORT MuxClass : public EntityClass
        {
        public:
            /// Constructor.
            MuxClass(void);

            /// Destructor.
            ~MuxClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the entity, i.e. aMUX_UART or aMUX_USB.
            void init(Module* pModule,
                      const uint8_t index);

            /// Get the mux enable/disable status
            /// \param bEnabled true: mux is enabled, false: the mux is disabled.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getEnable(uint8_t* bEnabled);

            /// Enable the mux.
            /// \param bEnable true: enables the mux for the selected channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setEnable(const uint8_t bEnable);

            /// Get the current selected mux channel.
            /// \param channel Indicates which chanel is selected.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getChannel(uint8_t* channel);

            /// Set the current mux channel.
            /// \param channel mux channel to select.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setChannel(const uint8_t channel);

            /// Get the voltage of the indicated mux channel.
            /// \param channel The channel in which voltage was requested.
            /// \param microVolts 32 bit signed integer (in microVolts) based on the board's
            /// ground and reference voltages.
            /// \note Not all modules provide 32 bits of accuracy; Refer to the module's
            /// datasheet to determine the analog bit depth and reference voltage.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getChannelVoltage(const uint8_t channel, int32_t* microVolts);
            
            /// Get the configuration of the mux.
            /// \param config integer representing the mux configuration either default, or split-mode.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getConfiguration(int32_t* config);
            
            /// Set the configuration of the mux.
            /// \param config integer representing the mux configuration either muxConfig_default, or muxConfig_splitMode.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setConfiguration(const int32_t config);
            
            /// Get the current mux mode.
            /// \param splitMode integer representing the channel selection for each sub-channel within the mux. See the data-sheet for the device for specific information.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getSplitMode(int32_t* splitMode);
            
            /// Set the current mux mode.
            /// \param splitMode integer representing the channel selection for each sub-channel within the mux. See the data-sheet for the device for specific information.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setSplitMode(const int32_t splitMode);
            
            

        };


        // MARK:  - UART Class
        /////////////////////////////////////////////////////////////////////
        /// UARTClass.
        /// A UART is a "Universal Asynchronous Receiver/Transmitter.  Many times
        /// referred to as a COM (communication), Serial, or TTY (teletypewriter) port.
        ///
        /// The UART Class allows the enabling and disabling of the UART data lines.
        class aLIBEXPORT UARTClass : public EntityClass
        {
        public:
            /// Constructor.
            UARTClass(void);

            /// Destructor.
            ~UARTClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the entity, i.e. aMUX_UART or aMUX_USB.
            void init(Module* pModule,
                      const uint8_t index);

            /// Enable the UART channel.
            /// \param bEnable False or 0 = Disabled, True or 1 = Enabled
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setEnable(const uint8_t bEnable);

            /// Get the UART channel state.
            /// \param bEnabled true: enabled, false: disabled.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getEnable(uint8_t* bEnabled);

        };



        // MARK:  - Pointer Class
        /////////////////////////////////////////////////////////////////////
        /// PointerClass. Access the reflex scratchpad from a host computer.
        ///
        /// The Pointers access the pad which is a shared memory area on a
        /// BrainStem module. The interface allows the use of the brainstem
        /// scratchpad from the host, and provides a mechanism for allowing the
        /// host application and brainstem relexes to communicate.
        ///
        /// The Pointer allows access to the pad in a similar manner as a file
        /// pointer accesses the underlying file. The cursor position can be
        /// set via setOffset. A read of a character short or int can be made
        /// from that cursor position. In addition the mode of the pointer can
        /// be set so that the cursor position automatically increments or set
        /// so that it does not this allows for multiple reads of the same pad
        /// value, or reads of multi-record values, via and incrementing pointer.

        class aLIBEXPORT PointerClass : public EntityClass
        {
        public:
            /// Constructor.
            PointerClass(void);

            /// Destructor.
            ~PointerClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index the pointer element index.
            void init(Module* pModule,
                      const uint8_t index);

            /// Get the offset of the pointer
            /// \param offset The value of the offset.
            /// \return All possible standard UEI return values.
            aErr getOffset(uint8_t* offset);

            /// Set the offset of the pointer
            /// \param offset The value of the offset.
            /// \return All possible standard UEI return values.
            aErr setOffset(uint8_t offset);

            /// Get the mode of the pointer
            /// \param mode The mode: aPOINTER_MODE_STATIC or aPOINTER_MODE_AUTO_INCREMENT.
            /// \return All possible standard UEI return values.
            aErr getMode(uint8_t* mode);

            /// Set the mode of the pointer
            /// \param mode The mode: aPOINTER_MODE_STATIC or aPOINTER_MODE_AUTO_INCREMENT.
            /// \return All possible standard UEI return values.
            aErr setMode(uint8_t mode);

            /// Get the handle to the store.
            /// \param handle The handle of the store.
            /// \return All possible standard UEI return handles.
            aErr getTransferStore(uint8_t* handle);

            /// Set the handle to the store.
            /// \param handle The handle of the store.
            /// \return All possible standard UEI return handles.
            aErr setTransferStore(uint8_t handle);

            /// Transfer data to the store.
            /// \param length The length of the data transfer.
            /// \return All possible standard UEI return values.
            aErr initiateTransferToStore(uint8_t length);

            /// Transfer data from the store.
            /// \param length The length of the data transfer.
            /// \return All possible standard UEI return values.
            aErr initiateTransferFromStore(uint8_t length);

            /// Get a char (1 byte) value from the pointer at this object's index,
            /// where elements are 1 byte long.
            /// \param value The value of a single character (1 byte) stored in the pointer.
            /// \return All possible standard UEI return values.
            aErr getChar(uint8_t* value);

            /// Set a char (1 byte) value to the pointer at this object's element index,
            /// where elements are 1 byte long.
            /// \param value The single char (1 byte) value to be stored in the pointer.
            /// \return All possible standard UEI return values.
            aErr setChar(const uint8_t value);

            /// Get a short (2 byte) value from the pointer at this objects index,
            /// where elements are 2 bytes long
            /// \param value The value of a single short (2 byte) stored in the pointer.
            /// \return All possible standard UEI return values.
            aErr getShort(uint16_t* value);

            /// Set a short (2 bytes) value to the pointer at this object's element index,
            /// where elements are 2 bytes long.
            /// \param value The single short (2 byte) value to be set in the pointer.
            /// \return All possible standard UEI return values.
            aErr setShort(const uint16_t value);

            /// Get an int (4 bytes) value from the pointer at this objects index,
            /// where elements are 4 bytes long
            /// \param value The value of a single int (4 byte) stored in the pointer.
            /// \return All possible standard UEI return values.
            aErr getInt(uint32_t* value);

            /// Set an int (4 bytes) value from the pointer at this objects index,
            /// where elements are 4 bytes long
            /// \param value The single int (4 byte) value to be stored in the pointer.
            /// \return All possible standard UEI return values.
            aErr setInt(const uint32_t value);

        };

        // MARK:  - Rail Class
        /////////////////////////////////////////////////////////////////////
        /// RailClass. Provides power rail functionality on certain modules.
        /// This entity is only available on certain modules. The RailClass can
        /// be used to control power to downstream devices, I has the ability to
        /// take current and voltage measurements, and depending on hardware, may
        /// have additional modes and capabilities.
        class aLIBEXPORT RailClass : public EntityClass
        {
        public:
            /// Constructor.
            RailClass(void);

            /// Destructor.
            ~RailClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the entity. Each rail index refers to a specific hardware
            /// voltage plane or "rail". Refer to the module datasheet for definition of the hardware
            /// voltage planes and specific capabilities.
            void init(Module* pModule,
                      const uint8_t index);

            /// Get the rail current.
            /// \param microamps The current in micro-amps (1 == 1e-6A).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getCurrent(int32_t* microamps);

            /// Set the rail current limit setting. (Check product datasheet to see if this feature is available)
            /// \param microamps The current in micro-amps (1 == 1e-6A).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setCurrentLimit(const int32_t microamps);

            /// Get the rail current limit setting. (Check product datasheet to see if this feature is available)
            /// \param microamps The current in micro-amps (1 == 1e-6A).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getCurrentLimit(int32_t* microamps);

            /// Get the rail temperature.
            /// \param microcelsius The measured temperature associated with the rail in
            /// micro-Celsius (1 == 1e-6˚C). The temperature may be associated with the module's
            /// internal rail circuitry or an externally connected temperature sensors. Refer to
            /// the module datasheet for definition of the temperature measurement location and
            /// specific capabilities.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getTemperature(int32_t* microcelsius);

            /// Get the state of the external rail switch. Not all rails can be switched
            /// on and off. Refer to the module datasheet for capability specification of the rails.
            /// \param bEnable true: enabled: connected to the supply rail voltage;
            /// false: disabled: disconnected from the supply rail voltage
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getEnable(uint8_t* bEnable);

            /// Set the state of the external rail switch. Not all rails can be switched
            /// on and off. Refer to the module datasheet for capability specification of the rails.
            /// \param bEnable true: enable and connect to the supply rail voltage;
            /// false: disable and disconnect from the supply rail voltage
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setEnable(const uint8_t bEnable);

            /// Set the rail supply voltage. Rail voltage control capabilities vary between modules.
            /// Refer to the module datasheet for definition of the rail voltage capabilities.
            /// \param microvolts The voltage in micro-volts (1 == 1e-6V) to be supply by the rail.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setVoltage(const int32_t microvolts);

            /// Get the rail supply voltage. Rail voltage control capabilities vary between modules.
            /// Refer to the module datasheet for definition of the rail voltage capabilities.
            /// \param microvolts The voltage in micro-volts (1 == 1e-6V) currently supplied by
            /// the rail. On some modules this is a measured value so may not exactly match what was
            /// previously set via the setVoltage interface. Refer to the module datasheet to
            /// to determine if this is a measured or stored value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getVoltage(int32_t* microvolts);

            /// Get the rail setpoint voltage. Rail voltage control capabilities vary between modules.
            /// Refer to the module datasheet for definition of the rail voltage capabilities.
            /// \param microvolts The voltage in micro-volts (1 == 1e-6V) the rail is trying to
            /// achieve. On some modules this is a measured value so may not exactly match what was
            /// previously set via the setVoltage interface. Refer to the module datasheet to
            /// to determine if this is a measured or stored value.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getVoltageSetpoint(int32_t* microvolts);

            /// Enable or Disable kelvin sensing on the module.
            /// Refer to the module datasheet for definition of the rail kelvin sensing capabilities.
            /// \param bEnable enable or disable kelvin sensing.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setKelvinSensingEnable(const uint8_t bEnable);

            /// Determine whether kelvin sensing is enabled or disabled.
            /// Refer to the module datasheet for definition of the rail kelvin sensing capabilities.
            /// \param bEnable Kelvin sensing is enabled or disabled.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getKelvinSensingEnable(uint8_t* bEnable);

            /// Determine whether kelvin sensing has been disabled by the system.
            /// Refer to the module datasheet for definition of the rail kelvin sensing capabilities.
            /// \param state Kelvin sensing is enabled or disabled.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getKelvinSensingState(uint8_t* state);

            /// Set the operational mode of the rail.
            /// Refer to the module datasheet for definition of the rail operational capabilities.
            /// \param mode The operational mode to employ.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setOperationalMode(const uint8_t mode);

            /// Determine the current operational mode of the system.
            /// Refer to the module datasheet for definition of the rail operational mode capabilities.
            /// \param mode The current operational mode setting.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getOperationalMode(uint8_t* mode);

            /// Determine the current operational state of the system.
            /// Refer to the module datasheet for definition of the rail operational states.
            /// \param state The current operational state.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getOperationalState(uint8_t* state);

        };


        // MARK:  - Relay Class
        /////////////////////////////////////////////////////////////////////
        /// RelayClass. Interface to relay entities on BrainStem modules.
        /// Relay entities can be set, and the voltage read.  Other capabilities
        /// may be available, please see the product datasheet.
        class aLIBEXPORT RelayClass : public EntityClass
        {
        public:
            /// Constructor.
            RelayClass(void);

            /// Destructor.
            virtual ~RelayClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the digital entity being initialized.
            void init(Module* pModule,
                    const uint8_t index);

            /// Set the enable/disable state.
            /// \param bEnable False or 0 = Disabled, True or 1 = Enabled
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setEnable(const uint8_t bEnable);

            /// Get the state.
            /// \param bEnabled False or 0 = Disabled, True or 1 = Enabled
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getEnable(uint8_t* bEnabled);

            /// Get the scaled micro volt value with refrence to ground.
            /// \param microVolts 32 bit signed integer (in micro Volts) based on the boards
            /// ground and refrence voltages.
            /// \note Not all modules provide 32 bits of accuracy; Refer to the module's
            /// datasheet to determine the analog bit depth and reference voltage.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getVoltage(int32_t* microVolts);

        };

        // MARK:  - Temperature Class
        /////////////////////////////////////////////////////////////////////
        /// TemperatureClass.
        /// This entitiy is only available on certain modules, and provides a
        /// temperature reading in microcelsius.
        class aLIBEXPORT TemperatureClass : public EntityClass
        {
        public:
            /// Constructor.
            TemperatureClass(void);

            /// Destructor.
            ~TemperatureClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the entity.
            void init(Module* pModule,
                      const uint8_t index);

            /// Get the temperature.
            /// \param microcelsius The temperature in micro-Celsius (1 == 1e-6C).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getTemperature(int32_t* microcelsius);

        };
        
        // MARK: - Signal Class
        /////////////////////////////////////////////////////////////////////
        /// SignalClass. Interface to digital pins configured to produce square wave signals.
        /// This class is designed to allow for square waves at various frequencies and duty cycles. Control
        /// is defined by specifying the wave period as (T3Time) and the active portion of the cycle as (T2Time).
        /// See the entity overview section of the reference for more detail regarding the timing.
        class aLIBEXPORT SignalClass : public EntityClass
        {
        public:
            /// Constructor.
            SignalClass(void);
            
            /// Destructor
            ~SignalClass(void);
            
            /// Initialize the class.
            /// \param pModule The module.
            /// \param index The index.
            void init(Module* pModule, const uint8_t index);
            
            ///  Enable/Disable the signal output.
            /// \param enable True to enable, false to disable
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setEnable(const uint8_t enable);
            
            /// Get the Enable/Disable of the signal.
            /// \param enable True to enable, false to disable
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getEnable(uint8_t* enable);
            
            /// Invert the signal output.
            ///
            /// Normal mode is High on t0 then low at t2.
            /// Inverted mode is Low at t0 on period start and high at t2.
            ///
            /// \param invert to invert, false for normal mode.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setInvert(const uint8_t invert);
            
            /// Get the invert status the signal output.
            ///
            /// Normal mode is High on t0 then low at t2.
            /// Inverted mode is Low at t0 on period start and high at t2.
            ///
            /// \param invert to invert, false for normal mode.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getInvert(uint8_t* invert);
            
            /// Set the signal period or T3 in nanoseconds.
            ///
            /// \param t3_nsec Integer not larger than unsigned 32 bit max value representing
            ///        the wave period in nanoseconds.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setT3Time(const uint32_t t3_nsec);
            
            /// Get the signal period or T3 in nanoseconds.
            ///
            /// \param t3_nsec Integer not larger than unsigned 32 bit max value representing
            ///        the wave period in nanoseconds.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getT3Time(uint32_t* t3_nsec);
        
            /// Set the signal active period or T2 in nanoseconds.
            ///
            /// \param t2_nsec Integer not larger than unsigned 32 bit max value representing
            ///        the wave active period in nanoseconds.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setT2Time(const uint32_t t2_nsec);
            
            /// Get the signal active period or T2 in nanoseconds.
            ///
            /// \param t2_nsec Integer not larger than unsigned 32 bit max value representing
            ///        the wave active period in nanoseconds.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getT2Time(uint32_t* t2_nsec);
            
        };
        
        
        
        // MARK: - Equalizer Class
        /////////////////////////////////////////////////////////////////////
        /// EqualizerClass. Provides receiver and transmitter gain/boost/emphasis
        /// settings for some of Acroname's products.  Please see product
        /// documentation for further details.
        class aLIBEXPORT EqualizerClass : public EntityClass
        {
        public:
            /// Constructor.
            EqualizerClass(void);
            
            /// Destructor.
            ~EqualizerClass(void);
            
            /// Initialize the class.
            /// \param pModule The module.
            /// \param index The index.
            void init(Module* pModule, const uint8_t index);
            
            /// Sets the receiver configuration for a given channel.
            /// \param channel The equalizer receiver channel.
            /// \param config Configuration to be applied to the receiver.
            /// \return Returns \ref EntityReturnValues "common entity" return values.
            aErr setReceiverConfig(const uint8_t channel, const uint8_t config);
            
            /// Gets the receiver configuration for a given channel.
            /// \param channel The equalizer receiver channel.
            /// \param config Configuration of the receiver.
            /// \return Returns \ref EntityReturnValues "common entity" return values.
            aErr getReceiverConfig(const uint8_t channel, uint8_t* config);
            
            /// Sets the transmitter configuration
            /// \param config Configuration to be applied to the transmitter.
            /// \return Returns \ref EntityReturnValues "common entity" return values.
            aErr setTransmitterConfig(const uint8_t config);
            
            /// Gets the transmitter configuration
            /// \param config Configuration of the Transmitter.
            /// \return Returns \ref EntityReturnValues "common entity" return values.
            aErr getTransmitterConfig(uint8_t* config);
            
        };
        

        // MARK:  - Store Class
        /////////////////////////////////////////////////////////////////////
        /// StoreClass. The store provides a flat file system on modules that
        /// have storage capacity. Files are referred to as slots and they have
        /// simple zero-based numbers for access.
        /// Store slots can be used for generalized storage and commonly contain
        /// compiled reflex code (files ending in .map) or templates used by the
        /// system. Slots simply contain bytes with no expected organization but
        /// the code or use of the slot may impose a structure.
        /// Stores have fixed indices based on type. Not every module contains a
        /// store of each type. Consult the module datasheet for details on which
        /// specific stores are implemented, if any, and the capacities of implemented stores.
        class aLIBEXPORT StoreClass : public EntityClass
        {
        public:
            /// Constructor.
            StoreClass(void);

            /// Destructor.
            ~StoreClass(void);

            /// Initialize the class.
            /// \param pModule The module.
            /// \param index The index.
            void init(Module* pModule,
                      const uint8_t index);

            /// Get slot state.
            /// \param slot The slot number.
            /// \param state true: enabled, false: disabled.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getSlotState(const uint8_t slot,
                              uint8_t* state);

            /// Load the slot.
            /// \param slot The slot number.
            /// \param pData The data.
            /// \param length The data length.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr loadSlot(const uint8_t slot,
                          const uint8_t* pData,
                          const uint16_t length);

            /// Unload the slot data.
            /// \param pData Byte array that the unloaded data will be placed into.
            /// \param dataLength - The length of pData buffer in bytes. This is the maximum
            ///                     number of bytes that should be unloaded.
            /// \param unloadedLength Length of data that was unloaded. Unloaded length
            ///                       will never be larger than dataLength.
            /// \param slot The slot number.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr unloadSlot(const uint8_t slot,
                            const size_t dataLength,
                            uint8_t* pData,
                            size_t* unloadedLength);

            /// Enable slot.
            /// \param slot The slot number.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr slotEnable(const uint8_t slot);

            /// Disable slot.
            /// \param slot The slot number.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr slotDisable(const uint8_t slot);

            /// Get the slot capacity.
            /// \param slot The slot number.
            /// \param capacity The slot capacity.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getSlotCapacity(const uint8_t slot,
                                 size_t* capacity);

            /// Get the slot size
            /// \param slot The slot number.
            /// \param size The slot size.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getSlotSize(const uint8_t slot,
                             size_t* size);
        };


        // MARK:  - System Class
        /////////////////////////////////////////////////////////////////////
        /// SystemClass. The System class provides access to the core settings,
        /// configuration and system information of the BrainStem module. The class
        /// provides access to the model type, serial number and other static
        /// information as well as the ability to set boot reflexes, toggle the
        /// user LED, as well as affect module and router addresses etc.
        /// The most common brainstem example uses the system entity to blink
        /// the User LED.
        class aLIBEXPORT SystemClass : public EntityClass
        {
        public:
            /// Constructor.
            SystemClass(void);

            /// Destructor.
            virtual ~SystemClass(void);

            /// Initialize the aSystem class.
            /// \param pModule The module to which this entity belongs.
            /// \param index Not used; always 0.
            void init(Module* pModule,
                      const uint8_t index);

            /// Get the current address the module uses on the BrainStem network.
            /// \param address The address the module is using on the BrainStem network.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getModule(uint8_t* address);

            /// Get the base address of the module. Software offsets and hardware offsets are
            /// added to this base address to produce the effective module address.
            /// \param address The address the module is using on the BrainStem network.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getModuleBaseAddress(uint8_t* address);

            /// Set the router address the module uses to communicate with the host and heartbeat to
            /// in order to establish the BrainStem network.
            /// This setting must be saved and the board reset before the setting
            /// becomes active.
            /// Warning: changing the router address may cause the module to "drop off" the
            ///          BrainStem network if the
            /// new router address is not in use by a BrainStem module.
            /// Please review the BrainStem network fundamentals before modifying the router
            /// address.
            /// \param address The router address to be used.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setRouter(const uint8_t address);

            /// Get the router address the module uses to communicate with the host and heartbeat to
            /// in order to establish the BrainStem network.
            /// \param address The address.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getRouter(uint8_t* address);

            /// Set the delay between heartbeat packets which are sent from the module.
            /// For link modules, these these heartbeat are sent to the host.
            /// For non-link modules, these heartbeats are sent to the router address.
            /// Interval values are in 25.6 millisecond increments
            /// Valid values are 1-255; default is 10 (256 milliseconds).
            /// \param interval The desired heartbeat delay.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setHBInterval(const uint8_t interval);

            /// Get the delay between heartbeat packets which are sent from the module.
            /// For link modules, these these heartbeat are sent to the host.
            /// For non-link modules, these heartbeats are sent to the router address.
            /// Interval values are in 25.6 millisecond increments.
            /// \param interval The current heartbeat delay.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getHBInterval(uint8_t* interval);

            /// Set the system LED state. Most modules have a blue system LED. Refer to the module
            /// datasheet for details on the system LED location and color.
            /// \param bOn true: turn the LED on, false: turn LED off.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setLED(const uint8_t bOn);

            /// Get the system LED state. Most modules have a blue system LED. Refer to the module
            /// datasheet for details on the system LED location and color.
            /// \param bOn true: LED on, false: LED off.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getLED(uint8_t* bOn);

            /// Set a store slot to be mapped when the module boots. The boot slot will be
            /// mapped after the module boots from powers up,
            /// receives a reset signal on its reset input, or
            /// is issued a software reset command.
            /// Set the slot to 255 to disable mapping on boot.
            /// \param slot The slot number in aSTORE_INTERNAL to be marked as a boot slot.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setBootSlot(const uint8_t slot);

            /// Get the store slot which is mapped when the module boots.
            /// \param slot The slot number in aSTORE_INTERNAL that is mapped after the module
            /// boots.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getBootSlot(uint8_t* slot);

            /// Get the modules firmware version number.
            /// The version number is packed into the return value. Utility functions
            /// in the aVersion module can unpack the major, minor and patch numbers from
            /// the version number which looks like M.m.p.
            /// \param build The build version date code.
            aErr getVersion(uint32_t* build);

            /// Get the module's model enumeration. A subset of the possible model enumerations
            /// is defined in BrainStem.h under "BrainStem model codes". Other codes are be used
            /// by Acroname for proprietary module types.
            /// \param model The module's model enumeration.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getModel(uint8_t* model);

            /// Get the module's serial number. The serial number is a unique 32bit integer
            /// which is usually communicated in hexadecimal format.
            /// \param serialNumber The module's serial number.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getSerialNumber(uint32_t* serialNumber);

            /// Save the system operating parameters to the persistent module flash memory.
            /// Operating parameters stored in the system flash will be loaded after the module
            /// reboots. Operating parameters include: heartbeat interval, module address,
            /// module router address
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr save(void);

            /// Reset the system.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr reset(void);

            /// Saves system log events to a slot defined by the module (usually ram slot 0).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr logEvents(void);

            /// Get the module's accumulated uptime in minutes
            /// \param uptimeCounter The module's accumulated uptime in minutes.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getUptime(uint32_t* uptimeCounter);

            /// Get the module's maximum temperature in micro-C
            /// \param maxTemperature The module's maximum system temperature in micro-C
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getMaximumTemperature(int32_t* maxTemperature);

            /// Get the module's input voltage.
            /// \param inputVoltage The module's input voltage reported in microvolts.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getInputVoltage(uint32_t* inputVoltage);

            /// Get the module's input current.
            /// \param inputCurrent The module's input current reported in microamps.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getInputCurrent(uint32_t* inputCurrent);

            /// Get the module hardware address offset. This is added to the base address to allow the
            /// module address to be configured in hardware. Not all modules support the
            /// hardware module address offset. Refer to the module datasheet.
            /// \param offset The module address offset.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getModuleHardwareOffset(uint8_t* offset);

            /// Set the software address offset.
            /// This software offset is added to the module base address, and potentially a
            /// module hardware address to produce the final module address. You
            /// must save the system settings and restart for this to take effect.
            /// Please review the BrainStem network fundamentals before modifying the module address.
            /// \param address The address for the module. Value must be even from 0-254.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setModuleSoftwareOffset(const uint8_t address);

            /// Get the software address offset.
            /// This software offset is added to the module base address, and potentially a
            /// module hardware address to produce the final module address. You
            /// must save the system settings and restart for this to take effect.
            /// Please review the BrainStem network fundamentals before modifying the module address.
            /// \param address The address for the module. Value must be even from 0-254.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getModuleSoftwareOffset(uint8_t* address);

            /// Get the router address system setting.
            /// This setting may not be the same as the current router address if
            /// the router setting was set and saved but no reset has occured.
            /// Please review the BrainStem network fundamentals before modifying the module address.
            /// \param address The address for the module. Value must be even from 0-254.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getRouterAddressSetting(uint8_t* address);

            /// Enables/Disables the route to me function.
            /// This function allows for easy networking of BrainStem modules.
            /// Enabling (1) this function will send an I2C General Call to all devices
            /// on the network and request that they change their router address
            /// to the of the calling device. Disabling (0) will cause all devices
            /// on the BrainStem network to revert to their default address.
            /// \param bOn Enable or disable of the route to me function 1 = enable.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr routeToMe(const uint8_t bOn);

        };


        // MARK:  - Timer Class
        /////////////////////////////////////////////////////////////////////
        /// TimerClass. The Timer Class provides access to a simple scheduler.
        /// Reflex routines can be written which will be executed upon expiration
        /// of the timer entity. The timer can be set to fire only once, or to
        /// repeat at a certain interval.
        class aLIBEXPORT TimerClass : public EntityClass
        {
        public:
            /// Constructor
            TimerClass(void);

            /// Destructor
            ~TimerClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the timer entity.
            void init(Module* pModule,
                      const uint8_t index);

            /// Get the currently set expiration time in microseconds. This is not a "live" timer.
            /// That is, it shows the expiration time originally set with setExpiration; it does
            /// not "tick down" to show the time remaining before expiration.
            /// \param usecDuration The timer expiration duration in microseconds.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getExpiration(uint32_t* usecDuration);

            /// Set the expiration time for the timer entity. When the timer expires, it will
            /// fire the associated timer[index]() reflex.
            /// \param usecDuration The duration before timer expiration in microseconds.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setExpiration(const uint32_t usecDuration);

            /// Get the mode of the timer which is either single or repeat mode.
            /// \param mode The mode of the time. aTIMER_MODE_REPEAT or aTIMER_MODE_SINGLE.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getMode(uint8_t* mode);

            /// Set the mode of the timer which is either single or repeat mode.
            /// \param mode The mode of the timer. aTIMER_MODE_REPEAT or aTIMER_MODE_SINGLE.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            /// \return ::aErrNone - Action completed successfully.
            aErr setMode(const uint8_t mode);

        };

        // MARK:  - USB Class
        /////////////////////////////////////////////////////////////////////
        /// USBClass.
        /// The USB class provides methods to interact with a USB hub and USB
        /// switches. Different USB hub products have varying support; check the
        /// datasheet to understand the capabilities of each product.
        class aLIBEXPORT USBClass : public EntityClass
        {
        public:
            /// Constructor.
            USBClass(void);

            /// Destructor.
            ~USBClass(void);

            /// Initialize the class.
            /// \param pModule The module to which this entity belongs.
            /// \param index The index of the entity, i.e. the port
            void init(Module* pModule,
                      const uint8_t index);

            /// Enable both power and data lines for a port.
            /// \param channel The USB sub channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setPortEnable(const uint8_t channel);

            /// Disable both power and data lines for a port.
            /// \param channel The USB sub channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setPortDisable(const uint8_t channel);

            /// Enable the only the data lines for a port without changing the state
            /// of the power line.
            /// \param channel The USB sub channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setDataEnable(const uint8_t channel);

            /// Disable only the data lines for a port without changing the state of
            /// the power line.
            /// \param channel The USB sub channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setDataDisable(const uint8_t channel);

            /// Enable the only the data lines for a port without changing the state
            /// of the power line, Hi-Speed (2.0) only.
            /// \param channel The USB sub channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setHiSpeedDataEnable(const uint8_t channel);

            /// Disable only the data lines for a port without changing the state of
            /// the power line, Hi-Speed (2.0) only.
            /// \param channel The USB sub channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setHiSpeedDataDisable(const uint8_t channel);

            /// Enable the only the data lines for a port without changing the state
            /// of the power line, SuperSpeed (3.0) only.
            /// \param channel The USB sub channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setSuperSpeedDataEnable(const uint8_t channel);

            /// Disable only the data lines for a port without changing the state of
            /// the power line, SuperSpeed (3.0) only.
            /// \param channel The USB sub channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setSuperSpeedDataDisable(const uint8_t channel);

            /// Enable only the power line for a port without changing the state of the
            /// data lines.
            /// \param channel The USB sub channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setPowerEnable(const uint8_t channel);

            /// Disable only the power line for a port without changing the state of the
            /// data lines.
            /// \param channel The USB sub channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setPowerDisable(const uint8_t channel);

            /// Get the current through the power line for a port.
            /// \param channel The USB sub channel.
            /// \param microamps The USB channel current in micro-amps (1 == 1e-6A).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getPortCurrent(const uint8_t channel, int32_t* microamps);

            /// Get the voltage on the power line for a port.
            /// \param channel The USB sub channel.
            /// \param microvolts The USB channel voltage in microvolts (1 == 1e-6V).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getPortVoltage(const uint8_t channel, int32_t* microvolts);

            /// Get a bit mapped representation of the hub mode; see the product
            /// datasheet for state mapping. Usually represents the hub's downstream
            /// ports data and power line enable/disable state.
            /// \param state The USB hub state.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getHubMode(uint32_t* state);

            /// Set a bit mapped hub state; see the product datasheet for state mapping.
            /// Usually represents the hub's downstream ports data and power line
            /// enable/disable state.
            /// \param state The USB hub state.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setHubMode(const uint32_t state);

            /// Clear the error status for the given channel.
            /// \param channel the port to clear error status for.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr clearPortErrorStatus(const uint8_t channel);

            /// Get the upstream switch mode for the USB upstream ports. Returns
            /// auto, port 0 or port 1.
            ///
            /// \param mode The Upstream port mode.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getUpstreamMode(uint8_t* mode);

            /// Set the upstream switch mode for the USB upstream ports. Values are
            /// usbUpstreamModeAuto, usbUpstreamModePort0 and usbUpstreamModePort1
            ///
            /// \param mode The Upstream port mode.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setUpstreamMode(const uint8_t mode);

            /// Get the upstream switch state for the USB upstream ports. Returns 2
            /// if no ports plugged in, 0 if the mode is set correctly and a cable
            /// is plugged into port 0, and 1 if the mode is set correctly and a cable
            /// is plugged into port 1.
            ///
            /// \param state The Upstream port state.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getUpstreamState(uint8_t* state);

            /// Set the interport enumeration delay in milliseconds.
            /// This setting should be saved with a stem.system.save() call.
            ///
            /// \param ms_delay 100ms delay increment.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setEnumerationDelay(const uint32_t ms_delay);

            /// Get the interport enumeration delay.
            ///
            /// \param ms_delay 100ms delay increment.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getEnumerationDelay(uint32_t* ms_delay);


            /// Set the current limit for the port. If the set limit is not achievable,
            /// devices will round down to the nearest available current limit setting.
            /// This setting can be saved with a stem.system.save() call.
            ///
            /// \param channel USB downstream channel to limit.
            /// \param microAmps The current limit setting.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setPortCurrentLimit(const uint8_t channel, const uint32_t microAmps);

            /// Get the current limit for the port. This reflects the limit setting currently.
            /// in effect.
            ///
            /// \param channel USB downstream channel to limit.
            /// \param microAmps The current limit setting.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getPortCurrentLimit(const uint8_t channel, uint32_t* microAmps);

            /// Set the mode for the Port.
            /// The mode is a bitmapped representation of the capabilities of the
            /// usb port. These capabilities change for each of the BrainStem devices
            /// which implement the usb entity. See your device datasheet for a complete
            /// list of capabilities. There is a unified bit mapping for port
            /// mode at \ref usbPortMode
            ///
            /// \param channel USB downstream channel to set the mode on.
            /// \param mode The port mode setting as packet bit mask.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setPortMode(const uint8_t channel, const uint32_t mode);

            /// Get the current mode for the Port.
            /// The mode is a bitmapped representation of the capabilities of the
            /// usb port. These capabilities change for each of the BrainStem devices
            /// which implement the usb entity. See your device datasheet for a complete
            /// list of capabilities. There is a unified bit mapping for port
            /// mode at \ref usbPortMode
            ///
            /// \param channel USB downstream channel.
            /// \param mode The port mode setting. Mode will be filled with the current setting.
            ///  Mode bits that are not used will be marked as don't care
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getPortMode(const uint8_t channel, uint32_t* mode);

            /// Get the current State for the Port.
            ///
            /// \param channel USB downstream channel.
            /// \param state The port mode setting. Mode will be filled with the current setting.
            ///  Mode bits that are not used will be marked as don't care
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getPortState(const uint8_t channel, uint32_t* state);

            /// Get the current error for the Port.
            ///
            /// \param channel USB downstream channel.
            /// \param error The port mode setting. Mode will be filled with the current setting.
            ///  Mode bits that are not used will be marked as don't care
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getPortError(const uint8_t channel, uint32_t* error);

            /// Set the upstream boost mode.
            /// Boost mode increases the drive strength of the USB data signals (power signals
            /// are not changed). Boosting the data signal strength may help to overcome
            /// connectivity issues when using long cables or connecting through "pogo" pins.
            /// Possible modes are 0 - no boost, 1 - 4% boost,
            /// 2 - 8% boost, 3 - 12% boost. This setting is not applied until a stem.system.save() call
            /// and power cycle of the hub. Setting is then persistent until changed or the hub
            /// is reset. After reset, default value of 0% boost is restored.
            ///
            /// \param setting Upstream boost setting 0, 1, 2, or 3.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setUpstreamBoostMode(const uint8_t setting);

            /// Set the downstream boost mode.
            /// Boost mode increases the drive strength of the USB data signals (power signals
            /// are not changed). Boosting the data signal strength may help to overcome
            /// connectivity issues when using long cables or connecting through "pogo" pins.
            /// Possible modes are 0 - no boost, 1 - 4% boost,
            /// 2 - 8% boost, 3 - 12% boost. This setting is not applied until a stem.system.save() call
            /// and power cycle of the hub. Setting is then persistent until changed or the hub
            /// is reset. After reset, default value of 0% boost is restored.
            ///
            /// \param setting Downstream boost setting 0, 1, 2, or 3.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setDownstreamBoostMode(const uint8_t setting);

            /// Get the upstream boost mode. Possible modes are 0 - no boost, 1 - 4% boost,
            /// 2 - 8% boost, 3 - 12% boost.
            ///
            /// \param setting The current Upstream boost setting 0, 1, 2, or 3.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getUpstreamBoostMode(uint8_t* setting);

            /// Get the downstream boost mode. Possible modes are 0 - no boost, 1 - 4% boost,
            /// 2 - 8% boost, 3 - 12% boost.
            ///
            /// \param setting The current Downstream boost setting 0, 1, 2, or 3.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getDownstreamBoostMode(uint8_t* setting);

            /// Get the current data transfer speed for the downstream port.
            /// The data speed can be Hi-Speed (2.0) or SuperSpeed (3.0) depending on what the
            /// downstream device attached is using
            ///
            /// \param channel USB downstream channel to check.
            /// \param speed Filled with the current port data speed
            /// - N/A: usbDownstreamDataSpeed_na = 0
            /// - Hi Speed: usbDownstreamDataSpeed_hs = 1
            /// - SuperSpeed: usbDownstreamDataSpeed_ss = 2
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getDownstreamDataSpeed(const uint8_t channel, uint8_t* speed);

            /// Sets the connect mode of the switch.
            /// \param channel The USB sub channel.
            /// \param mode The connect mode
            /// - usbManualConnect = 0
            /// - usbAutoConnect = 1
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setConnectMode(const uint8_t channel, const uint8_t mode);

            /// Gets the connect mode of the switch.
            /// \param channel The USB sub channel.
            /// \param mode The current connect mode
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getConnectMode(const uint8_t channel, uint8_t* mode);

            /// Set Enable/Disable on the CC1 line.
            /// \param channel - USB channel.
            /// \param bEnable
            /// - Disabled: 0
            /// - Enabled: 1
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setCC1Enable(const uint8_t channel, const uint8_t bEnable);

            /// Get Enable/Disable on the CC1 line.
            /// \param channel - USB channel.
            /// \param pEnable
            /// - Disabled: 0
            /// - Enabled: 1
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getCC1Enable(const uint8_t channel, uint8_t* pEnable);

            /// Set Enable/Disable on the CC2 line.
            /// \param channel - USB channel.
            /// \param bEnable
            /// - Disabled: 0
            /// - Enabled: 1
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setCC2Enable(const uint8_t channel, const uint8_t bEnable);

            /// Get Enable/Disable on the CC1 line.
            /// \param channel - USB channel.
            /// \param pEnable
            /// - Disabled: 0
            /// - Enabled: 1
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getCC2Enable(const uint8_t channel, uint8_t* pEnable);

            /// Get the current through the CC1 for a port.
            /// \param channel The USB sub channel.
            /// \param microamps The USB channel current in micro-amps (1 == 1e-6A).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getCC1Current(const uint8_t channel, int32_t* microamps);

            /// Get the current through the CC2 for a port.
            /// \param channel The USB sub channel.
            /// \param microamps The USB channel current in micro-amps (1 == 1e-6A).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getCC2Current(const uint8_t channel, int32_t* microamps);

            /// Get the voltage of CC1 for a port.
            /// \param channel The USB sub channel.
            /// \param microvolts The USB channel voltage in micro-volts (1 == 1e-6V).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getCC1Voltage(const uint8_t channel, int32_t* microvolts);

            /// Get the voltage of CC2 for a port.
            /// \param channel The USB sub channel.
            /// \param microvolts The USB channel voltage in micro-volts (1 == 1e-6V).
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getCC2Voltage(const uint8_t channel, int32_t* microvolts);

            /// Enable/Disable the only the SBU1/2 based on the configuration of
            /// the usbPortMode settings.
            /// \param channel The USB sub channel.
            /// \param bEnable
            /// - Disabled: 0
            /// - Enabled: 1
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setSBUEnable(const uint8_t channel, const uint8_t bEnable);

            /// Get the Enable/Disable status of the SBU
            /// \param channel The USB sub channel.
            /// \param pEnable The enable/disable status of the SBU
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getSBUEnable(const uint8_t channel, uint8_t* pEnable);

            /// Set Cable flip. This will flip SBU, CC and SS data lines.
            /// \param channel The USB sub channel.
            /// \param bEnable
            /// - Disabled: 0
            /// - Enabled: 1
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setCableFlip(const uint8_t channel, const uint8_t bEnable);

            /// Get Cable setting.
            /// \param channel The USB sub channel.
            /// \param pEnable The enable/disable status of cable flip.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getCableFlip(const uint8_t channel, uint8_t* pEnable);
            
            /// Set USB Alt Mode Configuration.
            /// \param channel The USB sub channel
            /// \param configuration The USB configuration to be set for the given channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr setAltModeConfig(const uint8_t channel, const uint32_t configuration);
            
            /// Set USB Alt Mode Configuration.
            /// \param channel The USB sub channel
            /// \param configuration The USB configuration for the given channel.
            /// \return Returns \ref EntityReturnValues "common entity" return values
            aErr getAltModeConfig(const uint8_t channel, uint32_t* configuration);

        };

    } // mamespace BrainStem
} // namespace Acroname

#endif //__BrainStem_entity_H__
