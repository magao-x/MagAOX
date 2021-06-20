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


#include "welchpsd.h"


/* ----------------------------- *
 *            STATIC             *
 *            METHODS            *
 * ----------------------------- */

static
void load_inputpsd(struct psd_config* psd, double* inbuf)
{
        double mean = 0;
        for (int i = 0; i < psd->num_modes; ++i) {
                
                for (int j = 0; j < psd->signal_length; ++j) {
                        mean += inbuf[i + j * psd->num_modes];
                }

                mean /= psd->signal_length;
                
                for (int j = 0; j < psd->signal_length; ++j) {
                        psd->in[i + (j * psd->num_modes)] = 
                              (inbuf[i + (j * psd->num_modes)] - mean) * psd->window[j];
                }

                mean = 0;
        }
}



static void get_psd_rt(struct psd_config* psd)
{
        size_t max_index = psd->num_modes * (psd->signal_length / 2 + 1);
        fftw_execute(psd->plan);

        for (size_t i = 0; i < max_index; ++i) {
                psd->res[i] = (psd->out[i])[0] * (psd->out[i])[0]
                            + (psd->out[i])[1] * (psd->out[i])[1];
        }

}



static void sum_psds(struct welch_config* welch, struct buffer* circbuf, 
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
void update_psd(struct psd_config* psd, struct buffer* out, 
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




static void* welch_calculate(void* inputs)
{
        struct wc_inputs* wc_input = (struct wc_inputs *)inputs;
        struct welch_config* welch = wc_input->wconfig;
        struct buffer* buf = wc_input->in_buf;
        struct buffer* circbuf = wc_input->circ_buf;
        IMAGE* image_out = wc_input->image;

        bool   firstPSD = true;
        size_t numPSDs  = 0;
        size_t cycle    = 0;
        size_t ready    = 0;

        struct timespec ts;
        struct timespec ts_out;
        FILE* fp = fopen("psd_out.dat","w");


        while(1) {
                clock_gettime(CLOCK_REALTIME, &ts);
                ts.tv_sec += 2; // assuming a 1 second psd
                sem_timedwait(welch->fetch_complete, &ts); 

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
        
        fclose(fp);
}



/*========================================*
 *               PUBLIC                   *
 *              INTERFACE                 *
 *              FUNCTIONS                 *
 *========================================*/

struct psd_config psd_init(size_t mode_count, size_t signal_len,
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

        struct psd_config psd = {
                .num_modes = mode_count,
                .signal_length = signal_len,
                .sample_time = dt,

                .in = (double *)fftw_malloc(in_size),
                .out = (fftw_complex *)fftw_malloc(out_size),
                .res = (double *)fftw_malloc(res_size),

                .window = (double *)fftw_malloc(signal_len * sizeof(double))
        };
        psd.plan = fftw_plan_many_dft_r2c(rank, m, howmany, psd.in, inembed, 
                                       istride, idist, psd.out, onembed, 
                                       ostride, odist, FFTW_MEASURE),
        
        psd.norm_squared = 0;
        for (size_t i = 0; i < signal_len; ++i) {
                psd.window[i] = win(i, signal_len);
                psd.norm_squared += psd.window[i] * win(i, signal_len);
        }

        return psd;
}



struct welch_config welch_init(size_t mode_count, size_t signal_len,
                               size_t total_pts,  double dt, 
                               double (*win)(size_t, size_t),
                               IMAGE* image)
{
        struct welch_config welch = {
                .psd = psd_init(mode_count, signal_len, dt, win),
                .total_duration = total_pts,
                .num_psds = total_pts / (signal_len/2) - 1,
                .image = image,
                .fetch_complete = (sem_t *)malloc(sizeof(sem_t *))
        };

        sem_init(welch.fetch_complete, 0, 0);
        return welch;
}



void get_psd(struct psd_config* psd)
{
        size_t max_index = psd->num_modes * (psd->signal_length / 2 + 1);
        fftw_execute(psd->plan);
        for (size_t i = 0; i < max_index; ++i) {
            psd->res[i] = (psd->out[i])[0] * (psd->out[i])[0]
                        + (psd->out[i])[1] * (psd->out[i])[1];
        }
}



void run_welch_rt(struct welch_config welch, IMAGE* welch_psd)
{
        struct buffer inbuf 
                = buf_init(welch.psd.num_modes * (welch.psd.signal_length / 2),
                                          welch.psd.num_modes, 4);
        struct buffer circbuf 
                = buf_init(welch.num_psds * (welch.psd.signal_length/2 + 1),
                                          welch.num_psds, welch.psd.num_modes);
        struct timespec ts;
        sem_t* sem = (welch.image)->semptr[0];

        struct wc_inputs thrd_in = {
                .wconfig = &welch,
                .in_buf = &inbuf,
                .circ_buf = &circbuf,
                .image = welch_psd
        };

        pthread_t estimation;
        pthread_create(&estimation, NULL, welch_calculate, (void *)&thrd_in);
       
        bool psd0 = true; 
        while(1) { 
                clock_gettime(CLOCK_REALTIME, &ts); 
                ts.tv_sec += 1;
       
                if(sem_timedwait(sem, &ts) == 0) {
                        if (psd0 == true) {
                                add_buf_line(&inbuf, welch.image->array.D);
                                if (inbuf.dataptr == inbuf.blocks[2]) {
                                        psd0 = false;
                                        sem_post(welch.fetch_complete);
                                }		
                        } else {
                                add_buf_line(&inbuf, welch.image->array.D);
                                if (inbuf.dataptr == inbuf.blocks[3]) {
                                        sem_post(welch.fetch_complete);
                                }
                                else if (inbuf.dataptr == inbuf.blocks[4]) {
                                        inbuf.dataptr = inbuf.blocks[2];
                                        sem_post(welch.fetch_complete);
                                }
                        }
	        } else {
                        puts("Timed out.");
                        if((welch.image)->md[0].sem <= 0) break; // Server cleaned up.
           
                        if(errno == EINTR) break; 
           
                        if(errno != ETIMEDOUT) {
                                fprintf(stderr, "Error in PSD esimation \
                                               (Welch): Unknown error.\n");
                                break;
                        }
                }
        }

        pthread_join(estimation, NULL);
        
        free_buf(inbuf);
        free_buf(circbuf);
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
