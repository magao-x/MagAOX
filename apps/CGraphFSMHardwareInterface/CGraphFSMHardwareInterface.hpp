/// \file
/// $Revision: $

#pragma once
#include <stdint.h>

union AdcAccumulator
{
	uint64_t all;
	struct 
	{
		int64_t Samples;
		uint32_t NumAccums;
		
	} __attribute__((__packed__));	
	static const int32_t AdcFullScale = 0x7FFFFFFFL; //2^32 - 1; must divide accumulator by numaccums first obviously
	
	AdcAccumulator() { all = 0; }
		
	void printf() const { ::printf("AdcAccumulator: Sample: %+10.0lf ", (double)Sample); ::printf("(0x%.8lX", (uint32_t)(all >> 32));  ::printf("%.8lX)", (uint32_t)(all)); ::printf(", NumAccums: %lu ", (uint32_t)NumAccums); ::printf("(0x%lX)", (uint32_t)NumAccums); }
	
} __attribute__((__packed__));

union AdcTimestamp
{
	uint32_t all;
	struct 
	{
		uint32_t SubsecondTicks : 27; //Max FPGA clock rate for 27b is 134,217,728Hz (134MHz) works well for our target speed of 103MHz.
		uint32_t UnixTimeLsbs: 5; //We'd really like a little more granularity, but ti's only ambiguious every 32 seconds I think we can deal with that...
		
	} __attribute__((__packed__));	
	
	AdcTimestamp() { all = 0; }
		
	void printf() const { ::printf("AdcTimestamp: Sample: %+10.0lf ", (double)Sample); ::printf("(0x%.8lX", (uint32_t)(all >> 32));  ::printf("%.8lX)", (uint32_t)(all)); ::printf(", NumAccums: %lu ", (uint32_t)NumAccums); ::printf("(0x%lX)", (uint32_t)NumAccums); }
	
} __attribute__((__packed__));

union AdcFifo
{
	uint64_t all;
	struct 
	{
		int32_t Sample;
		AdcTimestamp Timestamp;
		uint16_t NumSamplesInFifo;
		
	} __attribute__((__packed__));	
	
	static const int32_t AdcFullScale = 0x7FFFFFFFL; //2^32 - 1;
	static const uint16_t FifoMaxDepth = 0x0FFFUL; //The FPGA doesn't actually have very much ram for fifos.
	
	AdcFifo() { all = 0; }
		
	void printf() const { ::printf("AdcFifo: Sample: %+10.0lf ", (double)Sample); ::printf("(0x%.8lX", (uint32_t)(all >> 32));  ::printf("%.8lX)", (uint32_t)(all)); ::printf(", NumAccums: %lu ", (uint32_t)NumAccums); ::printf("(0x%lX)", (uint32_t)NumAccums); }
	
} __attribute__((__packed__));

union CGraphFSMHardwareControlRegister
{
	uint32_t all;
	struct 
	{
		uint16_t HighVoltageEnable : 1; //b0; Turn on the +150V supply
		uint16_t P1p2VEnable : 1; //b1; Turn on the +1.2V supply
		uint16_t P3p3VDEnable : 1; //b2; Turn on the +3.3V digital supply
		uint16_t PAnalogVEnable : 1; //b3; Turn on the analog supplies (+2.5V, +3.3VA, +5V, -5V)
		uint16_t SyncAdcs : 1; //b4; Syncronize first A/D sample to next incoming PPS signal
		uint16_t ForceChopper : 1; //b5; Force vs. freerun chopper; advanced feature not initially implemented
		uint16_t ChopperNonInverted : 1; //b6; Chopper polarity
		uint16_t AdcDownsampleRatio; //b16-31;  65535 = 16Hz; 1 = 1.04MHz
		
	} __attribute__((__packed__));	
	
	CGraphFSMHardwareControlRegister() { all = 0; }
		
	void printf() const { ::printf("CGraphFSMHardwareControlRegister: Sample: %+10.0lf ", (double)Sample); ::printf("(0x%.8lX", (uint32_t)(all >> 32));  ::printf("%.8lX)", (uint32_t)(all)); ::printf(", NumAccums: %lu ", (uint32_t)NumAccums); ::printf("(0x%lX)", (uint32_t)NumAccums); }
	
} __attribute__((__packed__));

union CGraphFSMHardwareStatusRegister
{
	uint32_t all;
	struct 
	{
		uint32_t PPSDetected : 1; //b0; Is there toggling on the PPS input?
		uint32_t HVFaultA : 1; //b1; Overtemperature or overcurrent on first high voltage output driver
		uint32_t HVFaultB : 1; //b2; Overtemperature or overcurrent on second high voltage output driver
		uint32_t HVFaultC : 1; //b3; Overtemperature or overcurrent on third high voltage output driver
		
	} __attribute__((__packed__));	
	
	CGraphFSMHardwareStatusRegister() { all = 0; }
		
	void printf() const { ::printf("CGraphFSMHardwareStatusRegister: Sample: %+10.0lf ", (double)Sample); ::printf("(0x%.8lX", (uint32_t)(all >> 32));  ::printf("%.8lX)", (uint32_t)(all)); ::printf(", NumAccums: %lu ", (uint32_t)NumAccums); ::printf("(0x%lX)", (uint32_t)NumAccums); }
	
} __attribute__((__packed__));

static const size_t* CGraphFSMHardwareInterfaceBaseAddress = 0x0ULL;

struct CGraphFSMHardwareInterface
{
	uint32_t DeviceSerialNumber; //ro; FPGA manufacturer hardcoded device UUID
	uint32_t FpgaFirmwareBuildNumber; //ro; Auto-incremented firmware UUID
	uint32_t UnixSeconds; //rw; equivalent to time_t for 32b systems; low order bits of time_t on 64b systems; write to set/initialize FPGA clock
	uint32_t IdealTicksPerSecond; //ro; Target clock speed of FPGA device, approx 100M; likely 14.7456M * 7 = 103,219,200.
	uint32_t ActualTicksLastSecond; //ro; Count of clock ticks for entire last second; equal to IdealTicksPerSecond unless clock was set or GPS PPS signal is present
	uint32_t ClockTicksThisSecond; //ro; Elapsed time since last second rollover in terms of IdealTicksPerSecond; notice in lieu of GPS correction commodity clocks vary ~50ppm over -40/+85oC temperature range, plus jitter.
	uint32_t DacASetpoint; //rw; First D/A; Zero = zero travel, DacFullScale = full scale travel
	uint32_t DacBSetpoint; //rw; Second D/A; Zero = zero travel, DacFullScale = full scale travel
	uint32_t DacCSetpoint; //rw; Third D/A; Zero = zero travel, DacFullScale = full scale travel
	AdcAccumulator AdcAAccumulator; //rw; First A/D; read or write any value to clear & reset accumulator	
	AdcAccumulator AdcBAccumulator; //rw; Second A/D; read or write any value to clear & reset accumulator	
	AdcAccumulator AdcCAccumulator; //rw; Third A/D; read or write any value to clear & reset accumulator	
	AdcFifo AdcAFifo; //rw; First A/D; read of first byte of this structure latches entire strcuture on bus and removes one sample from fifo; write of any value clears fifo
	AdcFifo AdcBFifo; //rw; Second A/D; read of first byte of this structure latches entire strcuture on bus and removes one sample from fifo; write of any value clears fifo
	AdcFifo AdcCFifo; //rw; Third A/D; read of first byte of this structure latches entire strcuture on bus and removes one sample from fifo; write of any value clears fifo
	CGraphFSMHardwareControlRegister ControlRegister; //rw; see definition above
	CGraphFSMHardwareStatusRegister StatusRegister; //ro; see definition above

	static const uint32_t DacFullScale = 0x007FFFFFUL; //2^20 - 1 (do these need to be initialized in constructor even though it's static can't remember?)
	static const double DacDriverFullScaleOutputVoltage = 150.0; //Volts, don't get your fingers near this thing!
	static const double PZTDriverFullScaleOutputTravel = 0.00001; //Meters; note our granularity is this / DacFullScale which is approx 10pm
	
	CGraphFSMHardwareInterface() { }
		
	//~ void printf() const { ::printf("CGraphFSMHardwareInterface: Sample: %+10.0lf ", (double)Sample); ::printf("(0x%.8lX", (uint32_t)(all >> 32));  ::printf("%.8lX)", (uint32_t)(all)); ::printf(", NumAccums: %lu ", (uint32_t)NumAccums); ::printf("(0x%lX)", (uint32_t)NumAccums); }
	
} __attribute__((__packed__));

//EOF
