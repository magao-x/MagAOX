/**
 * welchpsd.c - calculates power spectral densities in "realtime"
 * Copyright (C) 2021  Jacob Trzaska
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include "welchpsd.hpp"


/* ----------------------------- *
 *            STATIC             *
 *            METHODS            *
 * ----------------------------- */

welchmethod::welchmethod()
: m_welchThreadInit { true },
  m_welchThreadID { 0 }
  
{

}



void welchmethod::loadInputPSD(double* inbuf)
{
   double mean = 0;
   for (size_t i = 0; i < m_psd.m_num_modes; ++i) {
                
      for (size_t j = 0; j < m_psd.m_signal_length; ++j) {
         mean += inbuf[i + j * m_psd.m_num_modes];
      }

      mean /= m_psd.m_signal_length;
                
      for (size_t j = 0; j < m_psd.m_signal_length; ++j) {
         m_psd.m_in[i + (j * m_psd.m_num_modes)] = (inbuf[i + (j * m_psd.m_num_modes)] - mean) 
                                             * m_psd.m_window[j];
      }

      mean = 0;
   }
}



void welchmethod::get_psd_rt()
{
   size_t max_index = m_psd.m_num_modes * (m_psd.m_signal_length / 2 + 1);
   fftw_execute(m_psd.m_plan);

   for (size_t i = 0; i < max_index; ++i) {
      m_psd.m_res[i] = (m_psd.m_out[i])[0] * (m_psd.m_out[i])[0]
                   + (m_psd.m_out[i])[1] * (m_psd.m_out[i])[1];
   }

}



void welchmethod::sum_psds(double* res)
{
   size_t signal_len = (m_psd.m_signal_length / 2) + 1;
   size_t numpsds = m_numPsds;
   size_t signals = m_psd.m_num_modes;
   double norm2 = m_psd.m_norm_squared;
   double dt = m_psd.m_sample_time;        

   for (size_t mode = 0; mode < signals; ++mode) {

      double sum;
      for(size_t j = 0; j < signal_len; ++j) {

         sum = 0;
         for(size_t k = 0; k < numpsds; ++k) {
            sum += m_circbuf.buffer_start[k + numpsds * (j + signal_len * mode)];
         }

         res[mode + (j * signals)] = (2 * dt * sum) / (numpsds * norm2);

      }

   }

}



void welchmethod::update_psd(size_t index) 
{
   size_t sig_len = (m_psd.m_signal_length / 2) + 1;
   size_t num_modes = m_psd.m_num_modes;

   for (size_t i = 0; i < num_modes; ++i) {
	   for (size_t k = 0; k < sig_len; ++k) {
         m_circbuf.buffer_start[index + m_numPsds * (k + sig_len * i)] 
         = m_psd.m_res[i + (k * num_modes)]; 
      }
   }
}




void* welchmethod::welchCalculate()
{
   m_welchThreadRestart = false;

   m_welchThreadID = syscall(SYS_gettid);

   bool   firstPSD = true;
   size_t numPSDs  = 0;
   size_t cycle    = 0;
   size_t ready    = 0;

   struct timespec ts;
   struct timespec ts_out;

   while(1) {
      clock_gettime(CLOCK_REALTIME, &ts);
      ts.tv_sec += 2; // assuming a 1 second psd
      sem_timedwait(m_fetch, &ts);

      if (m_welchThreadRestart == true) {
         ready = 0;
         cycle = 0;
         firstPSD = true;
         numPSDs = 0;
         m_welchThreadRestart = false;
      } 

      if (firstPSD == true) {
         loadInputPSD(m_inbuf.buffer_start);
         get_psd_rt();
         update_psd(numPSDs); 
         m_inbuf.copy_buf_block(m_inbuf.blocks[0], m_inbuf.blocks[1]);
         firstPSD = false;
      } else 
         switch(cycle) {	
         case 0:
            m_inbuf.copy_buf_block(m_inbuf.blocks[1], m_inbuf.blocks[2]);
            loadInputPSD(m_inbuf.buffer_start);
            get_psd_rt();
                       
            update_psd(numPSDs);
            m_inbuf.copy_buf_block(m_inbuf.blocks[0], m_inbuf.blocks[1]);
            cycle = 1;
            break;
         case 1:
            m_inbuf.copy_buf_block(m_inbuf.blocks[1], m_inbuf.blocks[3]);
            loadInputPSD(m_inbuf.buffer_start);
            get_psd_rt();	
                        
            update_psd(numPSDs);
            m_inbuf.copy_buf_block(m_inbuf.blocks[0], m_inbuf.blocks[1]);
            cycle = 0;
            break;
      }          

      if (ready || (numPSDs == (m_numPsds - 1))) {
      
         ready = 1;
         
         if (numPSDs == (m_numPsds-1))
            numPSDs = 0;
         else ++numPSDs;

         clock_gettime(CLOCK_REALTIME, &ts_out);
         m_imageOut->md[0].write = 1;
         sum_psds(m_imageOut->array.D);
         m_imageOut->md[0].cnt0++;
         m_imageOut->md[0].atime = ts_out;
         m_imageOut->md[0].write = 0;
         ImageStreamIO_sempost(m_imageOut, -1); 
      } else { 
         ++numPSDs;
      }

   }
        
}



void welchmethod::welchFetch()
{
   switch (m_psd0) {
   
      case true:

         m_inbuf.add_buf_line(m_imageIn->array.D);

         if (m_inbuf.dataptr == m_inbuf.blocks[2]) {
            m_psd0 = false;
            sem_post(m_fetch);
         } 

         break; 

      case false:

            m_inbuf.add_buf_line(m_imageIn->array.D);

            if (m_inbuf.dataptr == m_inbuf.blocks[3])
               sem_post(m_fetch);
            else if (m_inbuf.dataptr == m_inbuf.blocks[4]) 
               {
                  m_inbuf.dataptr = m_inbuf.blocks[2];
                  sem_post(m_fetch);
               }
         
            break;
            
   }


}


/*========================================*
 *               PUBLIC                   *
 *              INTERFACE                 *
 *              FUNCTIONS                 *
 *========================================*/

void psd_config::psd_init(size_t mode_count, size_t signal_len,
                           double dt, double (*win)(size_t, size_t))
{
   size_t in_size = mode_count * signal_len * sizeof(double);
   size_t out_size = mode_count * (signal_len/2 + 1) * sizeof(fftw_complex);
   size_t res_size = mode_count * (signal_len/2 + 1) * sizeof(double);

        int rank = 1; // 1D transform
        const int m[] = {(int) signal_len};
        int howmany = mode_count; 	
        int istride = mode_count, ostride = mode_count;
        int idist = 1, odist = 1;
        const int *inembed = m, *onembed = m;

        m_num_modes = mode_count;
        m_signal_length = signal_len;
        m_sample_time = dt;

        m_in = (double *)fftw_malloc(in_size);
        m_out = (fftw_complex *)fftw_malloc(out_size);
        m_res = (double *)fftw_malloc(res_size);

        m_window = (double *)fftw_malloc(signal_len * sizeof(double));
        
        m_plan = fftw_plan_many_dft_r2c(rank, m, howmany, m_in, inembed, 
                                       istride, idist, m_out, onembed, 
                                       ostride, odist, FFTW_MEASURE);
        
        m_norm_squared = 0;
        for (size_t i = 0; i < signal_len; ++i) {
                m_window[i] = win(i, signal_len);
                m_norm_squared += m_window[i] * win(i, signal_len);
        }

}



void welchmethod::welch_init( size_t mode_count, size_t signal_len,
                              size_t total_pts,  double dt, 
                              double (*win)(size_t, size_t),
                              IMAGE* image1,
                              IMAGE* image2
                            )
{
   if (!m_welchRunning)
   {

      m_psd.psd_init(mode_count, signal_len, dt, win);
      m_totalDuration = total_pts;
      m_numPsds = total_pts / (signal_len/2) - 1;
      m_imageIn = image1;
      m_imageOut = image2;
      m_fetch = (sem_t *)fftw_malloc(sizeof(sem_t *));

      sem_init(m_fetch, 0, 0);

      m_inbuf.buf_init(mode_count * (signal_len / 2), mode_count, 4);
      m_circbuf.buf_init(m_numPsds * (signal_len / 2 + 1),  m_numPsds, mode_count);
   }
   else 
   {
      //free current memory 
      m_inbuf.buf_free();
      m_circbuf.buf_free();
      m_psd.psd_free();
      
      // reallocate new memory
      m_psd.psd_init(mode_count, signal_len, dt, win);
      m_totalDuration = total_pts;
      m_numPsds = total_pts / (signal_len/2) - 1;
      m_imageIn = image1;
      m_imageOut = image2;
      m_fetch = (sem_t *)fftw_malloc(sizeof(sem_t *));
   
      m_inbuf.buf_init(mode_count * (signal_len / 2), mode_count, 4);
      m_circbuf.buf_init(m_numPsds * (signal_len / 2 + 1),  m_numPsds, mode_count);
   }
}



welchmethod::~welchmethod()
{
   m_inbuf.buf_free();
   m_circbuf.buf_free();
}



void psd_config::psd_free()
{
   fftw_destroy_plan(m_plan);
   fftw_free(m_in);
   fftw_free(m_out);
   fftw_free(m_res);
   fftw_free(m_window);
}



psd_config::~psd_config()
{
   fftw_destroy_plan(m_plan);
   fftw_free(m_in);
   fftw_free(m_out);
   fftw_free(m_res);
   fftw_free(m_window);
}
