/** \file ar_controller.hpp
 * \brief DDSPC predictive controller interface.
 *
 * \ingroup loPredCtrl_files
 */

#ifndef PCARC_HPP
#define PCARC_HPP

#include <Eigen/Dense>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
using namespace mx::improc;

#include "utils.hpp"
#include "recursive_least_squares.hpp"

namespace DDSPC
{

class PredictiveController
{

  private:
    RecursiveLeastSquares *rls;

    uint buffer_size;
    uint measurement_head{ 0 };
    uint command_head{ 0 };
    Matrix measurement_buffer;
    Matrix command_buffer;
    Matrix accel_buffer;
    uint   accel_head{ 0 };

    Matrix *regularization_matrix;
    bool    use_regularization_matrix_01{ true };
    bool    do_switch_regularization_matrix{ false };
    Matrix  regularization_matrix_01;
    Matrix  regularization_matrix_02;

    Matrix controller;
    Matrix integrator;

    int   _num_modes;
    int   _num_future;
    int   _num_history;
    int   _num_accel_channels;
    int   _num_accel_history;
    realT _gain;
    realT _delta_max;
    realT _regularization;

    int num_predictors;
    int num_features;
    int num_state_features;
    int num_accel_features;
    int num_correlations;

  public:
    /// Construct the predictive controller with optional accelerometer regressors.
    PredictiveController( int   num_actuators,          /**< [in] number of controlled WFS/command modes */
                          int   num_history,            /**< [in] history length for WFS and command regressors */
                          int   num_future,             /**< [in] prediction horizon in frames */
                          realT gain,                   /**< [in] integrator gain */
                          realT gamma,                  /**< [in] RLS forgetting factor */
                          realT initial_regularization, /**< [in] initial controller regularization value */
                          realT initial_covariance,     /**< [in] initial inverse covariance diagonal value */
                          int   num_accel_channels = 0, /**< [in] number of accelerometer channels */
                          int   accel_history = 0       /**< [in] accelerometer history length */
    );

    ~PredictiveController();

    /// Update the regularization strength used in the controller solve.
    void set_regularization( realT new_regularization /**< [in] new diagonal regularization value */ );

    /// Return a copy of the learned RLS prediction matrix.
    inline Matrix get_prediction_matrix()
    {
        return rls->prediction_matrix;
    };

    /// Reset the controller and learner state.
    void reset();

    /// Return stacked future WFS measurements.
    Matrix get_measurement_future();

    /// Return stacked past WFS measurements.
    Matrix get_measurement_past();

    /// Return stacked future commands with optional leading skip.
    Matrix get_command_future( int skip_cmds /**< [in] number of leading future command steps to skip */ );

    /// Return stacked past commands.
    Matrix get_command_past();

    /// Return recent command history relative to the current command head.
    Matrix get_current_command_past( int num_steps /**< [in] number of historical command steps */ );

    /// Return recent measurement history relative to the current measurement head.
    Matrix get_current_measurement_past( int num_steps /**< [in] number of historical measurement steps */ );

    /// Return stacked past accelerometer telemetry.
    Matrix get_accelerometer_past();

    /// Push one accelerometer sample into the synchronized telemetry ring.
    void push_accelerometer_sample( const Matrix &new_acceleration /**< [in] column vector of current accelerometer channels */
    );

    /// Compute the next command increment from the latest measurement and exploration noise.
    Matrix calculate_command( Matrix new_measurement,  /**< [in] current WFS residual vector */
                              Matrix exploration_noise /**< [in] exploratory control perturbation */
    );

    /// Run one RLS update using the current regressor and target vectors.
    void update_system();

    /// Recompute the optimal predictive controller from the current RLS matrix.
    void update_controller();
};

}

#endif
