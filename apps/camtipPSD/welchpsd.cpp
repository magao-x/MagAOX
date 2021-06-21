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



void welchmethod::load_inputpsd(double* inbuf)
{
   double mean = 0;
   for (size_t i = 0; i < m_psd.num_modes; ++i) {
                
      for (size_t j = 0; j < m_psd.signal_length; ++j) {
         mean += inbuf[i + j * m_psd.num_modes];
      }

      mean /= m_psd.signal_length;
                
      for (size_t j = 0; j < m_psd.signal_length; ++j) {
         m_psd.in[i + (j * m_psd.num_modes)] = (inbuf[i + (j * m_psd.num_modes)] - mean) 
                                             * m_psd.window[j];
      }

      mean = 0;
   }
}



void welchmethod::get_psd_rt()
{
   size_t max_index = m_psd.num_modes * (m_psd.signal_length / 2 + 1);
   fftw_execute(m_psd.plan);

   for (size_t i = 0; i < max_index; ++i) {
      m_psd.res[i] = (m_psd.out[i])[0] * (m_psd.out[i])[0]
                   + (m_psd.out[i])[1] * (m_psd.out[i])[1];
   }

}



void welchmethod::sum_psds(welch_config* welch, buffer* circbuf, 
                     double* res)
{
   size_t signal_len = (welch->psd.signal_length / 2) + 1;
   size_t numpsds = welch->num_psds;
   size_t signals = welch->psd.num_modes;
   double norm2 = welch->psd.norm_squared;
   double dt = welch->psd.sample_time;        

   for (size_t mode = 0; mode < signals; ++mode) {

      double sum;
      for(size_t j = 0; j < signal_len; ++j) {

         sum = 0;
         for(size_t k = 0; k < numpsds; ++k) {
            sum += circbuf->buffer_start[k + numpsds * (j + signal_len * mode)];
         }

         res[mode + (j * signals)] = (2 * dt * sum) / (numpsds * norm2);

      }

   }

}



void welchmethod::update_psd(buffer* out, 
                size_t num_psds, size_t index) 
{
   size_t sig_len = (psd.m_signal_length / 2) + 1;
   size_t num_modes = psd.m_num_modes;

   for (size_t i = 0; i < num_modes; ++i) {
	   for (size_t k = 0; k < sig_len; ++k) {
         out->buffer_start[index + num_psds * (k + sig_len * i)] 
         = psd.res[i + (k * num_modes)]; 
      }
   }
}




void* welchmethod::welch_calculate()
{
   buffer* buf = wc_input->in_buf;
   buffer* circbuf = wc_input->circ_buf;
   IMAGE* image_out = wc_input->image;
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
         reset = false;
         ready = 0;
         cycle = 0;
         firstPSD = true;
         numPSDs = 0;
      } 

      if (firstPSD == true) {
         load_inputpsd(buf->buffer_start);
         get_psd_rt();
         update_psd(circbuf, m_num_psds, numPSDs); 
         buf->copy_buf_block(buf->blocks[0], buf->blocks[1]);
         firstPSD = false;
      } else 
         switch(cycle) {	
         case 0:
            buf->copy_buf_block(buf->blocks[1], buf->blocks[2]);
            load_inputpsd(buf->buffer_start);
            get_psd_rt();
                       
            update_psd(circbuf, m_num_psds, numPSDs);
            buf->copy_buf_block(buf->blocks[0], buf->blocks[1]);
            cycle = 1;
            break;
         case 1:
            buf->copy_buf_block(buf->blocks[1], buf->blocks[3]);
            load_inputpsd(buf->buffer_start);
            get_psd_rt();	
                        
            update_psd(circbuf, m_num_psds, numPSDs);
            buf->copy_buf_block(buf->blocks[0], buf->blocks[1]);
            cycle = 0;
            break;
      }          

      if (ready || (numPSDs == (m_num_psds - 1))) {
      
         ready = 1;
         
         if (numPSDs == (m_num_psds-1))
            numPSDs = 0;
         else ++numPSDs;

         clock_gettime(CLOCK_REALTIME, &ts_out);
         image_out->md[0].write = 1;
         sum_psds(welch, circbuf, image_out->array.D);
         image_out->md[0].cnt0++;
         image_out->md[0].atime = ts_out;
         image_out->md[0].write = 0;
         ImageStreamIO_sempost(image_out, -1); 
      } else { 
         ++numPSDs;
      }

   }
        
}



welchmethod::welchFetch()
{


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



void welchmethod::welch_init(size_t mode_count, size_t signal_len,
                               size_t total_pts,  double dt, 
                               double (*win)(size_t, size_t),
                               IMAGE* image)
{
   m_psd = psd_init(mode_count, signal_len, dt, win);
   m_total_duration = total_pts;
   m_num_psds = total_pts / (signal_len/2) - 1;
   m_image = image;
   m_fetch = (sem_t *)fftw_malloc(sizeof(sem_t *));

   sem_init(m_fetch, 0, 0);


   buffer
}


welchmethod::~welchmethod()
{
}


psd_config::~psd_config()
{
        fftw_destroy_plan(m_plan);
        fftw_free(m_in);
        fftw_free(m_out);
        fftw_free(m_res);
        fftw_free(m_window);
}
