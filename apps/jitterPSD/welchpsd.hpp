// Jacob Trzaska
// welchpsd.h
//
// Module containing the code base for realtime welch method estimation 
// of power spectral densities. Uses ImageStreamIO and ImageStruct for 
// interprocess communication.

#pragma once // WELCH_PSD_H


#include <iostream>
#include <cstdlib>
#include <string>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <semaphore.h>

#include <fftw3.h> // should be compiled with AVX/AVX2

#include "buffer.hpp"


/** Configuration class for psd calcuations.
  *
  */ 
struct psd_config {
   size_t m_num_modes;     // number of psds to calculate
   size_t m_signal_length; // number of points for each psd
   double m_sample_time;   // sampling time

   double*       m_in;
   fftw_complex* m_out; 
   fftw_plan     m_plan;
   double*       m_res;
         
   double* m_window;
   double m_norm_squared;

   void psd_init( size_t num_modes, 
                  size_t signal_len,
                  double dt, 
                  double (*win) (size_t, size_t)
                );

   psd_config() {}
   void psd_free();
   ~psd_config();
};



class welchmethod {

   public:

      welchmethod();
      ~welchmethod();

      void welchFetch(double * data);
      
      void* welchCalculate();

      void welch_init( size_t mode_count,
                       size_t signal_length,
                       size_t total_pts,
                       double dt,
                       double (*win) (size_t, size_t),
                       sem_t* semOut
                     );

   protected:
      psd_config m_psd;
      bool m_psd0;
      bool m_welchRunning;

      size_t m_totalDuration;
      size_t m_numPsds;

      sem_t* m_fetch;
      sem_t* m_wfgSem;

      buffer m_inbuf;
      buffer m_circbuf;
      double* m_resultArray;

      // configuration variables for welchCalculate thread 
      bool m_welchThreadInit;

      pid_t m_welchThreadID;
 
      pcf::IndiProperty m_welchThreadProp;

      int m_welchThreadPrio;

      std::thread m_welchThread;

      bool m_welchThreadRestart;

   private:
      void loadInputPSD(double* inbuf);

      void get_psd_rt();

      void sum_psds();

      void update_psd(size_t index);
};
