/** -- FIRSTLIGHT --
  \file    cred2_sdk.h

  \brief   The purpose of the libcred2sdk library is to provide an easy way to achieve C-RED 2 USB acquisition

  \author FIRSTLIGHT

  \note    You will find in this header C-RED 2 camera characteristic constants and functions protoypes.
*/
 
#ifndef CRED2_SDK_H
#define CRED2_SDK_H
 
#ifdef __cplusplus
extern "C" {
#endif

#ifdef WIN32
#define DLL_EXPORT 	__declspec( dllexport )
#else
#define DLL_EXPORT
#endif

#include <stdint.h>


/** Error criticity mask */
#define LIBCRED2_ERROR_MASK            0xF000

/** Error that prevents proper operation of the C-RED 2 SDK */
#define LIBCRED2_ERROR_LEVEL_ERROR     0x8000

/** Error that impacts performance but do not prevents proper operation of the C-RED 2 SDK */
#define LIBCRED2_ERROR_LEVEL_WARNING   0x4000


/** Critical OS error (semaphore, locking... ) that prevents proper operation of the library */
#define LIBCRED2_ERROR_CRITICAL_OS             (LIBCRED2_ERROR_LEVEL_ERROR   | 0x0001)

/** Critical out of memory condition */
#define LIBCRED2_ERROR_CRITICAL_MEMORY         (LIBCRED2_ERROR_LEVEL_ERROR   | 0x0002)

/** Non critical OS error (memory lock ...) that may impact library performance */
#define LIBCRED2_ERROR_WARNING_OS              (LIBCRED2_ERROR_LEVEL_WARNING | 0x0003)

/** Critical acquisition error that requires restart of acquisition */
#define LIBCRED2_ERROR_ERROR_ACQUISITION       (LIBCRED2_ERROR_LEVEL_ERROR   | 0x0004)

/** Non critical acquisition error  */
#define LIBCRED2_ERROR_WARNING_ACQUISITION     (LIBCRED2_ERROR_LEVEL_WARNING | 0x0005)


/** USB Statistics */
struct cred2_stats
{
	int nb_rx_frames;			/**< Number of frames received */
	int nb_rx_errors;			/**< Number of errors detected during reception of frames */
	int rx_min_delay;			/**< Minimum delay between reception of two frames */
	int rx_avg_delay;			/**< Average delay between reception of two frames */
	int rx_max_delay;			/**< Maximum delay between reception of two frames */
	
	int nb_tx_frames;			/**< Number of frames sent to user application */
	int tx_min_delay;			/**< Minimum delay between transmission of two frames to user application */	
	int tx_avg_delay;			/**< Average delay between transmission of two frames to user application */
	int tx_max_delay;			/**< Maximum delay between transmission of two frames to user application */
	
	int frame_nb_prod_event;		/**< Number of GPIF prod event - FOR USB DEBUGGING */	
	int frame_nb_get_buffer_ok;		/**< Number of buffer successfully received from GPIF - FOR USB DEBUGGING*/
	int frame_nb_get_buffer_ko;		/**< Number of buffer unsuccessfully received from GPIF  - FOR USB DEBUGGING*/	
	int frame_nb_commit_buffer_ok;	/**< Number of buffer successfully committed - FOR USB DEBUGGING */
	int frame_nb_commit_buffer_ko;	/**< Number of buffer unsuccessfully committed - FOR USB DEBUGGING */	
	int frame_nb_cons_event;		/**< Number of USB cons event - FOR USB DEBUGGING */	
	int frame_nb_error;				/**< Number of errors seen - FOR USB DEBUGGING */
	
	int uart_rx_nb_prod_event;		/**< Number of UART->USB prod event - FOR USB DEBUGGING */		
	int uart_rx_nb_commit_buffer_ok;	/**< Number of UART->USB buffer successfully committed - FOR USB DEBUGGING */
	int uart_rx_nb_commit_buffer_ko;	/**< Number of UART->USB buffer unsuccessfully committed - FOR USB DEBUGGING */	
	int uart_rx_nb_wrapup_ok;			/**< Number of UART->USB partial buffer successfully committed - FOR USB DEBUGGING */
	int uart_rx_nb_wrapup_ko;			/**< Number of UART->USB partial buffer successfully committed - FOR USB DEBUGGING */
	int uart_rx_nb_cons_event;		/**< Number of UART->USB cons event - FOR USB DEBUGGING */	
	int uart_rx_nb_error;				/**< Number of errors seen - FOR USB DEBUGGING */
};


/**
 * @brief Detects connected C-RED 2 camera
 *
 * @return The number of detected camera, or a negative value on failure
 */
DLL_EXPORT int CRED2_detect();

/**
 * @brief Establishes a connection with the specified C-RED 2 camera.
 *
 *
 * @param id index of the camera. >Currently, only one camera is handled, should be 0
 * @param callback callback function called when an error is detected during camera operation
 * @param userctx user context for callback function
 *
 * @return the camera context, to be used in other API functions on success, or NULL on failure
 */

DLL_EXPORT void * CRED2_open(int id, void (*callback) (void * userctx, int error, const char * diag), void * userctx);

/**
 * @brief Performs basic checks on the connection with a C-RED 2 camera
 *
 * @param ctx camera context, returned by CRED2_open()
 * @param buffer diagnostic string
 * @param size size of the diagnostic string
 * 
 * @return 1 if no problem has been detected.
 */

DLL_EXPORT int CRED2_check(void * ctx, char * buffer, int size);


/**
 * @brief Starts C-RED 2 image acquisition
 *
 * @param ctx camera context, returned by CRED2_open()
 * @param width  image width
 * @param height image height
 * @param callback callback function called when an image has been received
 * @param userctx user context for frame callback
 *
 * @return 1 on success.
 */

DLL_EXPORT int CRED2_startAcquisition(void * ctx, int width, int height, void (* callback) (void * userctx, int16_t * frame), void * userctx);

/**
 * @brief Stops C-RED 2 image acquisition
 *
 * @param ctx camera context, returned by CRED2_open()
 * @return 1 on success.
 */
 
DLL_EXPORT int CRED2_stopAcquisition(void * ctx);

/**
 * @brief Terminates the connection with the specified C-RED 2 camera.
 *
 * @param ctx camera context, returned by CRED2_open()
 *
 * @return 1 on success.
 */

DLL_EXPORT int CRED2_close(void * ctx);

/**
 * @brief Control tags checking in the acquired frames
 *
 * When tag checking is enabled, extra check is performed
 * on frames acquired and error callback is called when a tag error is detected
 *
 * @param ctx camera context, returned by CRED2_open()
 * @param enable : enable (1) or disable (0) tag checks
 *
 * @return 1 on success.
 */

int CRED2_checkTagEnable(void * ctx, int enable);


/**
 * @brief Get USB statistics
 *
 * @param ctx camera context, returned by CRED2_open()
 
 * @param stats address of struct cred2_stats to be populated
 
 * @return 1 on success.
 */
 
DLL_EXPORT int CRED2_getStats(void * ctx, struct cred2_stats * stats);


/**
 * @brief Starts C-RED 2 debug logging
 *
 * @param ctx camera context, returned by CRED2_open() 
 * @param callback callback function called when a log is generated
 * @param userctx user context for frame callback
 *
 * @return 1 on success.
 */

DLL_EXPORT int CRED2_startLogger(void * ctx, void (* callback) (void * userctx, const char * log), void * userctx);

/**
 * @brief Stops C-RED 2 image debug logging
 *
 * @param ctx camera context, returned by CRED2_open()
 * @return 1 on success.
 */
 
 DLL_EXPORT int CRED2_stopLogger(void * ctx);
 
#ifdef __cplusplus
}
#endif

#endif
