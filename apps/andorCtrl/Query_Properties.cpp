/* To compile:
 * g++ -c -o Query_Properties.o Query_Properties.cpp
 * g++ -o Query_Properties Query_Properties.o -landor
 * 
 * To Execute:
 * sudo ./Query_Properties 0
 */

#include <stdio.h>
#include <stdlib.h>

#ifdef __GNUC__
#  if(__GNUC__ > 3 || __GNUC__ ==3)
#	define _GNUC3_
#  endif
#endif

#ifdef _GNUC3_
#  include <iostream>
#  include <fstream>
   using namespace std;
#else
#  include <iostream.h>
#  include <fstream.h>
#endif

#include <unistd.h>

#include "atmcdLXd.h"

   
// Function declarations
int CameraSelect (int iNumArgs, char* szArgList[]);
void CheckCooler();
void CheckEMGain();
void CheckAmplifier();
void CheckPreAmplifier();




// Default Parameters
int ReadMode=4;
/* 0 - Full Vertical Binning
 * 1 - Multi-Track; Need to call SetMultiTrack(int NumTracks, int height, int offset, int* bottom, int *gap)
 * 2 - Random-Track; Need to call SetRandomTracks
 * 3 - Single-Track; Need to call SetSingleTrack(int center, int height)
 * 4 - Image; See SetImage, need shutter during readout
 */ 

int AcqMode=5;
/* 1 - Single Scan
 * 2 - Accumulate
 * 3 - Kinetic Series
 * 5 - Run Till Abort
 * 
 * See Page 53 of SDK User's Guide for Frame Transfer Info
 */


// SetShutter(int typ, int mode, int closingtime, int openingtime)
/* typ:
 *      0 - Output TTL low signal to open shutter
 *      1 - Output TTL high signal to open shutter
 * mode:
 *      0 - Fully Auto (Andor controls when it opens and closes
 *      1 - Permanently Open
 *      2 - Permanently Closed
 *      4 - Open for FVB Series
 *      5 - Open for any series
 * closingtime:
 *      Time shutter takes to close in milliseconds
 * openingtime:
 *      Time shutter takes to open in milliseconds
 * Return:
 *      unsigned int
 */

int OutAmp=0;
/* 0 - Standard EMCCD gain register
 * 1 - Conventional CCD register
 */

float expoTime=0.1;




int main(int argc, char* argv[])
{
    // Find connected Camera
    if (CameraSelect (argc, argv) < 0) {
        cout << "*** CAMERA SELECTION ERROR" << endl;
        return -1;
    }
    
    //Initialize CCD
    unsigned long error;
	error = Initialize((char *)"/usr/local/etc/andor");
    cout << "The Error code is " << error << endl;
	if(error!=DRV_SUCCESS){
		cout << "Initialisation error...exiting" << endl;
		return(1);
	}

	sleep(2); //sleep to allow initialization to complete
    
    
    
    // Initialize Variables:
    int width, height;
    int nuWidth=100, nuHeight=100;
    int InternalShutter;
    int NumVSSpeeds, RecommendedVSSpeedIndex;
    float VSSpeed;
    int iTemp;
    bool quit;
	char choice;
	float fChoice;
    int iChoice;

    // Set initial default parameters:
    
    //Get Detector dimensions
	GetDetector(&width, &height);
    
    //Set Read Mode to --Image--
	SetReadMode(ReadMode);
	
    //Set Acquisition mode to --Single scan--
	SetAcquisitionMode(AcqMode);
	
    //Set initial exposure time
	SetExposureTime(expoTime);
	
    //Initialize Shutter
	SetShutter(1,0,50,50);
    
    // Set CameraLink
    SetCameraLinkMode(1);
    
    // Set Output Amplifier
    SetOutputAmplifier(OutAmp);
    
    //Setup Image dimensions
    SetImage(1,1,1,width,1,height);
    /* SetImage(int hbin, int vbin, int hstart, int hend, int vstart, int vend)
     * hbin: number of pixels to bin horizontally
     * vbin: number of pixels to bin vertically
     * hstart: Starting Column (inclusive)
     * hend: End column (inclusive)
     * vstart: Start row (inclusive)
     * vend: End row (inclusive)
     */
    
   // SetImage(1,1,206,305,206,305);
   // width = nuWidth;
   // height = nuHeight;
    
    // Print Detector Frame Size
    cout << "Detector Frame is: " << width << "x" << height << endl;

    
    // Query if Shutter is installed
    error = IsInternalMechanicalShutter(&InternalShutter);
    if(error==DRV_SUCCESS){
        if(InternalShutter==0){
            cout << "There is no installed Shutter" << endl;}
        else if(InternalShutter==1){
            cout << "There is an installed Shutter" << endl;}
    }
    
    
    
    GetNumberVSSpeeds(&NumVSSpeeds);
    cout << "Number of VS Speeds: " << NumVSSpeeds << endl;
    
    GetFastestRecommendedVSSpeed(&RecommendedVSSpeedIndex, &VSSpeed);
    cout << "Recommended VS Speed Index: " << RecommendedVSSpeedIndex << endl;
    SetVSSpeed(RecommendedVSSpeedIndex);
    GetFastestRecommendedVSSpeed(&RecommendedVSSpeedIndex, &VSSpeed);
    cout << "VS Speed is " << VSSpeed << endl;
    

        
    quit = false;
	do{
		//Show menu options
		cout << "        Menu" << endl;
		cout << "====================" << endl;
		cout << "a. Start Acquisition" << endl;
		cout << "s. Stop  Acquisition" << endl;
		cout << "t. Query Number of Aquired Images" << endl;
		cout << "b. Set Exposure Time" << endl;
        cout << "c. Set Temperature" << endl;
        cout << "d. Turn on Cooler" << endl;
        cout << "e. Turn off Cooler" << endl;
        cout << "f. Check Cooler Status" << endl;
        cout << "g. Check Pre Amp" << endl;
        cout << "h. Check Amp" << endl;
        cout << "j. Check EMGain" << endl;
        cout << "k. Set EMGain" << endl;
        cout << "l. Toggle Advanced EM Gain" << endl;
		cout << "z.     Exit" << endl;
		cout << "====================" << endl;
		cout << "Choice?::";
		//Get menu choice
		choice = getchar();

		switch(choice){
/*
		case 'a': //Acquire
			{
			StartAcquisition();

			int status;
			at_32* imageData = new at_32[width*height];

			//Loop until acquisition finished
			GetStatus(&status);
			while(status==DRV_ACQUIRING) GetStatus(&status);

			GetAcquiredData(imageData, width*height);

			//for(int i=0;i<width*height;i++) fout << imageData[i] << endl;

			//SaveAsBmp("./image.bmp", "./GREY.PAL", 0, 0);
            SaveAsFITS((char *)"image_crop.FITS", 4);
            //SaveAsRaw("image_Raw",3);
            
            delete[] imageData;
			}

			break;
*/

		case 'a': //Acquire
			{
			SetKineticCycleTime(0);
			StartAcquisition();
			
			cout << endl << "Starting Continuous Acquisition" << endl;
	
			break;
			}

		case 's':
			error = AbortAcquisition();
			if(error==DRV_SUCCESS){
				cout << "Stopping Acquisition" << endl;
			}
			break;
			
		case 't':
			int ind;
			error = GetTotalNumberImagesAcquired(&ind);
			if(error==DRV_SUCCESS){
				cout << "Number of Acquired Images: " << ind << endl;
			}
			break;

		case 'b': //Set new exposure time
			
			cout << endl << "Enter new Exposure Time(s)::";
			cin >> fChoice;

			SetExposureTime(fChoice);

			break;
        
        case 'c': //Set Target Temperature

			cout << endl << "Enter new Target Temperature(C)::";
			cin >> iTemp;
	
			SetTemperature(iTemp);

			break;

        case 'd': //Turn on cooler
 		
			error = CoolerON();
            if(error==DRV_SUCCESS){
                cout << "Cooler is ON" << endl;
            }

			break;

		case 'e': //Turn off cooler

			error = CoolerOFF();
            if(error==DRV_SUCCESS){
                cout << "Cooler is OFF" << endl;
            }
			break;

		case 'f': //Cooling Status
//			{
// 			unsigned int status;
// 			status = GetTemperature(&iTemp);
// 			
// 			cout << "Temperature is " << iTemp << "C" << endl;
// 			if(status==DRV_TEMPERATURE_OFF) cout << "Cooler is OFF" << endl;
// 			else if(status==DRV_TEMPERATURE_STABILIZED) cout << "Cooler Stabilised at target Temperature" << endl;
// 			else cout << "Cooler is ON" << endl; 
// 			}
            CheckCooler();
			
			break;
          
        case 'g': // Check PreAmplifier
            
            CheckPreAmplifier();
            break;
            
        case 'h': // Check the Output Amplifier
            
            CheckAmplifier();
            break;
            
        case 'j': // Report EM gain values
            
            CheckEMGain();
            break;
            
        case 'k': //Set new EMGain
			
			cout << endl << "Enter new EMGain::";
			cin >> iChoice;

			SetEMCCDGain(iChoice);

			break;
            
        case 'l': // Turn on Access to higher EM gains
            
            cout << endl << "Toggle Advanced EMGain (0 or 1)::";
            cin >> iChoice;
            SetEMAdvanced(iChoice);
            break;
            
		case 'z': //Exit

			quit = true;

			break;
		
		default:

			cout << "!Invalid Option!" << endl;

		} 
		getchar();

	}while(!quit);	
    
    // Shutdown the CCD
    ShutDown();
    cout << "Camera Shutdown Complete" << endl;
    return 0;
}





//for getting the current state of the cooler
void CheckCooler()
{
    int temp, temp_low, temp_high;
    unsigned long error=GetTemperatureRange(&temp_low, &temp_high);
    unsigned long status=GetTemperature(&temp);
    cout << "Current Temperature: " << temp << " C" << endl;
    cout << "Temp Range: {" << temp_low << "," << temp_high << "}" << endl;
    cout << "Status             : ";
    switch(status){
        case DRV_TEMPERATURE_OFF: cout << "Cooler OFF" << endl; break;
        case DRV_TEMPERATURE_STABILIZED: cout << "Stabilised" << endl; break;
        case DRV_TEMPERATURE_NOT_REACHED: cout << "Cooling" << endl; break;
        case DRV_TEMPERATURE_NOT_STABILIZED: cout << "Temp reached but not stablized" << endl; break;
        case DRV_TEMPERATURE_DRIFT: cout << "Temp had stabilized but has since drifted" << endl; break;
        default:cout << "Unknown" << endl;
    }
}


void CheckEMGain()
{
    int state;
    int gain;
    int low, high;
    unsigned long out = GetEMAdvanced(&state);
    if(out==DRV_SUCCESS){
        cout << "The Current Advanced EM gain setting is: " << state << endl;
    }
    out = GetEMCCDGain(&gain);
    if(out==DRV_SUCCESS){
        cout << "Current EMCCD Gain: " << gain << endl;
    }
    out = GetEMGainRange(&low, &high);
    if(out==DRV_SUCCESS){
        cout << " The range of EMGain is: {" << low << "," << high << "}" << endl;
    }
}

void CheckAmplifier()
{
    int amp;
    unsigned long out = GetNumberAmp(&amp);
    if(out==DRV_SUCCESS){
        cout << "Number of output Amplifiers: " << amp << endl;
    }
}
    
void CheckPreAmplifier()
{
    int NumberGains;
    float PreAmpGain;
    unsigned long error = GetNumberPreAmpGains(&NumberGains);
    if(error==DRV_SUCCESS){
        cout << "Number of PreAmp Gains is " << NumberGains << endl;
    }
    for(int i=0;i<NumberGains-1;i++)
    {
        unsigned long error = GetPreAmpGain(i, &PreAmpGain);
        if(error==DRV_SUCCESS){
            cout << "Gain for index: " << i << " is " << PreAmpGain << endl;
        }
    }
}
        


int CameraSelect (int iNumArgs, char* szArgList[])
{
  if (iNumArgs == 2) {
 
    at_32 lNumCameras;
    GetAvailableCameras(&lNumCameras);
    int iSelectedCamera = atoi(szArgList[1]);
 
    if (iSelectedCamera < lNumCameras && iSelectedCamera >= 0) {
      at_32 lCameraHandle;
      GetCameraHandle(iSelectedCamera, &lCameraHandle);
      SetCurrentCamera(lCameraHandle);
      return iSelectedCamera;
    }
    else
      return -1;
  }
  return 0;
}
