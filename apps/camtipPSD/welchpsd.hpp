// welchpsd.h
//
// Module containing the code base for realtime 
// welch method estimation of power spectral
// densities.
// Use ImageStreamIO and ImageStruct for interprocess
// communication.

#pragma once // WELCH_PSD_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <semaphore.h>

#include <fftw3.h> // should be compiled with AVX/AVX2
#include <ImageStruct.h>
#include <ImageStreamIO.h>

#include <buffer.h>



#ifdef __cplusplus
extern "C" {
#endif

/* Contains configuration for psd calcuations
 *
 * This struct may be used standalone or in 
 * conjuction with the realtime welch method
 * calculator.
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
};



/* Configuration for realtime welch method 
 * calculations. 
 * 
 * The image structure is for inter-process
 * communication.
 */
struct welch_config {
        IMAGE* image;
        
        struct psd_config psd;
        size_t            total_duration;
        size_t            num_psds;

        sem_t* fetch_complete;        
};


/* struct wc_inputs
 * 
 * Holds the important structures for
 * the realtime welch calculation. This
 * struct is used to pass the structures by
 * by reference to a pthread.
 */
struct wc_inputs {
        struct welch_config* wconfig; 
        struct buffer* in_buf;
        struct buffer* circ_buf;
        IMAGE* image;
};



/********************
*     Functions     *
*********************/
struct psd_config psd_init(size_t mode_count, size_t signal_len,
                           double dt, double (*win)(size_t, size_t));

struct welch_config welch_init(size_t mode_count, size_t signal_len,
                               size_t total_pts, double dt,
                               double (*win)(size_t, size_t),
                               IMAGE* image);


void get_psd(struct psd_config* psd_setup);

void run_welch_rt(struct welch_config welch, IMAGE* welch_psd);

void free_welch_rt(struct welch_config* welch);

void free_psd(struct psd_config* psd);

#ifdef __cplusplus
}
#endif // extern "C"
