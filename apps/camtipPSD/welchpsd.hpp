// welchpsd.h
//
// Module containing the code base for realtime 
// welch method estimation of power spectral
// densities.
// Use ImageStreamIO and ImageStruct for interprocess
// communication.

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
#include <ImageStruct.h>
#include <ImageStreamIO.h>

#include "buffer.hpp"



class welchmethod {

   public:

      void welchFetch();
      
      void welchCalculate();

      void welch_init( size_t mode_count,
                       size_t signal_length,
                       size_t total_pts,
                       double dt,
                       double (*win) (size_t, size_t),
                       IMAGE* image
                     );

   protected:
      psd_config m_psd;

      IMAGE* image_out;

      size_t m_totalDuration;
      size_t m_numPsds;

      sem_t* m_fetch;

      // configuration variables for welchCalculate thread 
      bool m_welchThreadInit;
      pid_t m_welchThreadID; 
      pcf::IndiProperty m_welchThreadProp;
      int m_welchThreadPrio;
      std::thread m_welchThread;

   private:
      void loadInputPSD(psd_config* psd, double* inbuf);

      void get_psd_rt();

      void sum_psds()

      void update_psd()
};


/** Configuration class for psd calcuations.
  *
  */ 
struct psd_config {
   size_t num_modes;
   size_t signal_length;
   double sample_time;

   double*       in;
   fftw_complex* out; 
   fftw_plan     plan;
   double*       res;
         
   double* window;
   double norm_squared;

   void psd_init( size_t num_modes, 
                  size_t signal_len,
                  double dt, 
                  double (*win) (size_t, size_t)
                );

   psd_config::psd_config() {}
   psd_config::~psd_config();
};
