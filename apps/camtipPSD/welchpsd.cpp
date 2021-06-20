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

static
void load_inputpsd(psd_config* psd, double* inbuf)
{
        double mean = 0;
        for (size_t i = 0; i < psd->num_modes; ++i) {
                
                for (size_t j = 0; j < psd->signal_length; ++j) {
                        mean += inbuf[i + j * psd->num_modes];
                }

                mean /= psd->signal_length;
                
                for (size_t j = 0; j < psd->signal_length; ++j) {
                        psd->in[i + (j * psd->num_modes)] = 
                              (inbuf[i + (j * psd->num_modes)] - mean) * psd->window[j];
                }

                mean = 0;
        }
}



static void get_psd_rt(psd_config* psd)
{
        size_t max_index = psd->num_modes * (psd->signal_length / 2 + 1);
        fftw_execute(psd->plan);

        for (size_t i = 0; i < max_index; ++i) {
                psd->res[i] = (psd->out[i])[0] * (psd->out[i])[0]
                            + (psd->out[i])[1] * (psd->out[i])[1];
        }

}



static void sum_psds(welch_config* welch, buffer* circbuf, 
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
                                sum += circbuf->buffer
                                       [k + numpsds * (j + signal_len * mode)];
                        }
                        res[mode + (j * signals)] = (2 * dt * sum) / (numpsds * norm2);
                }
        }
}



static
void update_psd(psd_config* psd, buffer* out, 
                size_t num_psds, size_t index) 
{
        size_t sig_len = (psd->signal_length / 2) + 1;
        size_t num_modes = psd->num_modes;

        for (size_t i = 0; i < num_modes; ++i) {
	        for (size_t k = 0; k < sig_len; ++k) {
                        out->buffer[index + num_psds * (k + sig_len * i)]
                                        = psd->res[i + (k * num_modes)]; 
                }
        }
}




void* welch_calculate(void* inputs)
{
        wc_inputs* wc_input = (struct wc_inputs *)inputs;
        welch_config* welch = wc_input->wconfig;
        buffer* buf = wc_input->in_buf;
        buffer* circbuf = wc_input->circ_buf;
        IMAGE* image_out = wc_input->image;
        bool reset = wc_input->welchThreadRestart;

        bool   firstPSD = true;
        size_t numPSDs  = 0;
        size_t cycle    = 0;
        size_t ready    = 0;

        struct timespec ts;
        struct timespec ts_out;

        while(1) {
                clock_gettime(CLOCK_REALTIME, &ts);
                ts.tv_sec += 2; // assuming a 1 second psd
                sem_timedwait(welch->fetch_complete, &ts);

                if (reset == true) {
                   reset = false;
                   ready = 0;
                   cycle = 0;
                   firstPSD = true;
                   numPSDs = 0;
                } 

                if (firstPSD == true) {
                        load_inputpsd(&welch->psd, buf->buffer);
                        get_psd_rt(&welch->psd);
                        update_psd(&welch->psd, circbuf, welch->num_psds, numPSDs); 
                        copy_buf_block(buf, buf->blocks[0], buf->blocks[1]);
                        firstPSD = false;
                } else 
                switch(cycle) {	
                case 0:
                        copy_buf_block(buf, buf->blocks[1], buf->blocks[2]);
                        load_inputpsd(&welch->psd, buf->buffer);
                        get_psd_rt(&welch->psd);
                       
                        update_psd(&welch->psd, circbuf, welch->num_psds, numPSDs);
                        copy_buf_block(buf, buf->blocks[0], buf->blocks[1]);
                        cycle = 1;
                        break;
                case 1:
                        copy_buf_block(buf, buf->blocks[1], buf->blocks[3]);
                        load_inputpsd(&welch->psd, buf->buffer);
                        get_psd_rt(&welch->psd);	
                        
                        update_psd(&welch->psd, circbuf, welch->num_psds, numPSDs);
                        copy_buf_block(buf, buf->blocks[0], buf->blocks[1]);
                        cycle = 0;
                        break;
                }          

               if (ready || (numPSDs == (welch->num_psds - 1))) {
                        ready = 1;
                        if (numPSDs == (welch->num_psds-1)) {
                            numPSDs = 0;
                        } else {
                            ++numPSDs;
                        }

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



/*========================================*
 *               PUBLIC                   *
 *              INTERFACE                 *
 *              FUNCTIONS                 *
 *========================================*/

psd_config psd_init(size_t mode_count, size_t signal_len,
                           double dt, double (*win)(size_t, size_t))
{
        size_t in_size = 
                mode_count * signal_len * sizeof(double);
        size_t out_size = 
                mode_count * (signal_len/2 + 1) * sizeof(fftw_complex);
        size_t res_size = 
                mode_count * (signal_len/2 + 1) * sizeof(double);

        int rank = 1; // 1D transform
        const int m[] = {signal_len};
        int howmany = mode_count; 	
        int istride = mode_count, ostride = mode_count;
        int idist = 1, odist = 1;
        const int *inembed = m, *onembed = m;

        psd_config psd;
                psd.num_modes = mode_count;
                psd.signal_length = signal_len;
                psd.sample_time = dt;

                psd.in = (double *)fftw_malloc(in_size);
                psd.out = (fftw_complex *)fftw_malloc(out_size);
                psd.res = (double *)fftw_malloc(res_size);

                psd.window = (double *)fftw_malloc(signal_len * sizeof(double));
        
        psd.plan = fftw_plan_many_dft_r2c(rank, m, howmany, psd.in, inembed, 
                                       istride, idist, psd.out, onembed, 
                                       ostride, odist, FFTW_MEASURE);
        
        psd.norm_squared = 0;
        for (size_t i = 0; i < signal_len; ++i) {
                psd.window[i] = win(i, signal_len);
                psd.norm_squared += psd.window[i] * win(i, signal_len);
        }

        return psd;
}



welch_config welch_init(size_t mode_count, size_t signal_len,
                               size_t total_pts,  double dt, 
                               double (*win)(size_t, size_t),
                               IMAGE* image)
{
        welch_config welch;
                welch.psd = psd_init(mode_count, signal_len, dt, win);
                welch.total_duration = total_pts;
                welch.num_psds = total_pts / (signal_len/2) - 1;
                welch.image = image;
                welch.fetch_complete = (sem_t *)malloc(sizeof(sem_t *));

        sem_init(welch.fetch_complete, 0, 0);
        return welch;
}



void get_psd(psd_config* psd)
{
        size_t max_index = psd->num_modes * (psd->signal_length / 2 + 1);
        fftw_execute(psd->plan);
        for (size_t i = 0; i < max_index; ++i) {
            psd->res[i] = (psd->out[i])[0] * (psd->out[i])[0]
                        + (psd->out[i])[1] * (psd->out[i])[1];
        }
}


void free_welch_rt(struct welch_config* welch)
{
        free_psd(&welch->psd);
}


void free_psd(struct psd_config* psd)
{
        fftw_destroy_plan(psd->plan);
        fftw_free(psd->in);
        fftw_free(psd->out);
        fftw_free(psd->res);
        fftw_free(psd->window);
}
