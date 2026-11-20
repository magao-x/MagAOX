/** \file multirate_ddspc.hpp
 * \brief Phase-conditioned lifted DDSPC: one shared QR-RLS updated per WFS frame and a prediction per DM slot.
 *
 * Derivation and validation: agents/plans/2026-11/ddspc-multirate-8khz-assisted.md.
 */
#ifndef PC_MULTIRATE_DDSPC_HPP
#define PC_MULTIRATE_DDSPC_HPP

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace DDSPC
{

/// Double-precision column-major matrix used by the multirate learner and its slot gains.
using MatrixD = Eigen::MatrixXd;

/// Double-precision column vector.
using VectorD = Eigen::VectorXd;

/// Fixed dimensions and RLS/control parameters of the multirate assisted model.
/** With m modes, q DM slots and r ACC samples per WFS frame, N WFS frames of history, F frames of horizon and an ACC
 * base window of L = r*accelHistory samples, the phase-k regressor (k = 0..r-1) is
 * [u_f; y_p; u_p; a_c..a_(c-L+1); a_(c+1)..a_(c+k)], newest first within each block except the trailing extra samples,
 * which are oldest first so that every phase regressor is a leading part of the phase r-1 regressor.
 */
struct MultirateConfig
{
    int    m_modes{ 2 };            ///< Controlled modes, m.
    int    m_history{ 15 };         ///< WFS history N in frames; command history is N-1 lifts.
    int    m_future{ 3 };           ///< Horizon F in WFS frames.
    int    m_slots{ 4 };            ///< DM slots per WFS frame, q (scheduler steps per cycle).
    int    m_ratio{ 4 };            ///< ACC samples per WFS frame, r.
    int    m_accelHistory{ 5 };     ///< ACC base window in WFS frames; L = r*accelHistory samples.
    double m_gamma{ 1 };            ///< Forgetting factor per WFS training example.
    double m_prior{ 1e-5 };         ///< Prior information rho; R starts at sqrt(rho)*I.
    double m_lambda{ 100 };         ///< Command penalty of the DDSPC cost.
    bool   m_scaledPenalty{ true }; ///< Scale lambda by max(C^T C), as the operational standard law does.
    double m_ripple{ 0 };           ///< Weight of command variation within each future frame (scaled like lambda).

    /// Command lift size q*m.
    int lift() const;

    /// Future command entries F*q*m.
    int futureInputs() const;

    /// ACC base window L in samples.
    int window() const;

    /// Known-state size of phase k: N*m + (N-1)*q*m + 2*(L+k).
    int state( int phase /**< [in] Extra ACC samples k. */ ) const;

    /// Regressor size of phase k.
    int features( int phase /**< [in] Extra ACC samples k. */ ) const;

    /// Target size F*m.
    int outputs() const;

    /// Throw std::invalid_argument for unsupported dimensions or parameters.
    void validate() const;
};

/// Augmented QR information factor [R | Z] of exponentially weighted least squares with prior rho*I.
/** R^T R = gamma^t rho I + sum_j gamma^(t-j) phi_j phi_j^T and R Theta^T = Z. Each Givens rotation j involves only
 * columns at and after j and the incoming row's earlier eliminated entries, so the leading block of R and the leading
 * rows of Z are exactly the factor of the same problem restricted to the leading regressor entries.
 */
class InformationQR
{
    /// Row-major [R | Z], features by features+outputs; R upper triangular.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m_factor;

    /// Incoming [phi^T | y^T] row being eliminated.
    Eigen::RowVectorXd m_row;

    int    m_features{ 0 }, m_outputs{ 0 };    ///< Regressor and target sizes.
    double m_rootGamma{ 1 }, m_rootPrior{ 1 }; ///< Square roots of the forgetting factor and prior.

  public:
    /// Allocate the prior factor.
    InformationQR( int    features /**< [in] Regressor size. */,
                   int    outputs /**< [in] Target size. */,
                   double gamma /**< [in] Forgetting factor in (0,1]. */,
                   double prior /**< [in] Positive prior information rho. */ );

    /// Restore the prior factor and zero coefficients.
    void reset();

    /// Add one weighted example by Givens rotations; no allocation.
    void update( const VectorD &regressor /**< [in] Finite phi, features long. */,
                 const VectorD &target /**< [in] Finite y, outputs long. */ );

    /// Solve the least-squares coefficients of the leading regressor entries: Theta^T = R_SS^-1 Z_S.
    void solve( int      leading /**< [in] Leading regressor entries, 1..features. */,
                MatrixD &coefficients /**< [out] Pre-sized leading by outputs Theta^T. */ ) const;

    /// Regressor size.
    int features() const;
};

/// Frozen known state of the newest WFS frame for the per-slot calculation.
struct MultirateOrigin
{
    std::uint64_t m_index{ 0 };     ///< WFS counter of the newest frame.
    std::uint64_t m_anchor{ 0 };    ///< Newest ACC counter at that WFS decision, c_i.
    std::uint64_t m_segment{ 0 };   ///< ACC source segment of the base window.
    bool          m_valid{ false }; ///< Complete WFS, command and ACC histories.
    VectorD       m_state;          ///< [y_p; u_p; a_c..a_(c-L+1)], state(0) entries.
};

/// Coherent set of slot gains from one coefficient version; owned by one thread at a time.
struct MultirateGains
{
    std::uint64_t        m_version{ 0 };   ///< Increments with every refreshed set.
    bool                 m_ready{ false }; ///< Trained and finite.
    std::vector<MatrixD> m_slot;           ///< K(k,p) at k*q+p: m rows by state(k)+p*m; the next slot's command rows.
};

/// Learning and per-phase prediction diagnostics.
struct MultirateStatus
{
    bool                       m_ready{ false };    ///< Enough updates for the full regressor.
    bool                       m_training{ false }; ///< Updates enabled.
    std::uint64_t              m_updates{ 0 };      ///< QR updates since reset, one per genuine WFS target.
    std::uint64_t              m_skipped{ 0 };      ///< Matured examples without complete ACC data.
    std::uint64_t              m_breaks{ 0 };       ///< History breaks (WFS gaps, unknown copied commands).
    std::uint64_t              m_refreshes{ 0 };    ///< Coefficient extractions and gain sets.
    std::vector<double>        m_rmse;              ///< A priori future-WFS RMSE of each phase predictor.
    std::vector<std::uint64_t> m_scored;            ///< Scalars scored for each phase.
};

/// Phase-conditioned lifted DDSPC with one shared information factor.
/** The caller serializes append(), learn() and refresh(). command() is static and reads only its arguments, so the
 * per-slot thread can use gains and an origin copied from this object while learning continues.
 */
class MultirateDDSPC
{
    MultirateConfig m_config; ///< Fixed dimensions.
    InformationQR   m_qr;     ///< Shared factor over the phase r-1 regressor.

    /// Frame ring, indexed by WFS counter modulo its size.
    struct Frame
    {
        VectorD       m_y;              ///< WFS measurement.
        VectorD       m_lift;           ///< Copied commands over the following interval, newest slot first.
        std::uint64_t m_anchor{ 0 };    ///< ACC counter c of this decision.
        std::uint64_t m_segment{ 0 };   ///< ACC segment of the window.
        bool          m_accel{ false }; ///< ACC window valid at this decision.
    };
    std::vector<Frame> m_frames;

    /// ACC sample ring keyed by counter; each entry stores its counter and segment for validation.
    struct Sample
    {
        std::uint64_t m_counter{ ~std::uint64_t( 0 ) }, m_segment{ 0 };
        double        m_value[2]{};
    };
    std::vector<Sample> m_samples;

    std::uint64_t m_last{ 0 };  ///< Newest appended WFS counter.
    std::uint64_t m_count{ 0 }; ///< Contiguous frames since the last history break.

    VectorD             m_phi, m_target;    ///< Matured phase r-1 example.
    VectorD             m_error;            ///< A priori prediction workspace.
    MatrixD             m_ctc;              ///< C^T C of the phase being refreshed.
    std::vector<double> m_buffer;           ///< ACC window plus extras of one example, oldest first.
    bool                m_example{ false }; ///< m_phi/m_target hold an unlearned example.

    std::vector<MatrixD> m_theta; ///< Theta_k^T of each phase from the latest refresh.

    bool            m_training{ false };  ///< Updates enabled.
    bool            m_dirty{ true };      ///< Factor or penalty changed since the last refresh.
    bool            m_extracted{ false }; ///< m_theta holds valid coefficients.
    std::uint64_t   m_version{ 0 };       ///< Gain-set version counter.
    double          m_lambda;             ///< Current command penalty.
    MultirateStatus m_status;             ///< Diagnostics.

  public:
    /// Allocate all histories, the factor and workspaces.
    explicit MultirateDDSPC( const MultirateConfig &config /**< [in] Validated dimensions. */ );

    /// Return the fixed configuration.
    const MultirateConfig &config() const;

    /// Reset coefficients, histories and diagnostics; training intent is kept.
    void reset();

    /// Break WFS/command/ACC histories after a gap, keeping coefficients.
    void break_history();

    /// Enable or pause updates.
    void training( bool enabled /**< [in] Requested state. */ );

    /// Change the command penalty; gains are recomputed at the next refresh.
    void regularization( double lambda /**< [in] Positive penalty. */ );

    /// Append WFS frame i and the commands copied since frame i-1; publish the frame's origin.
    /** applied holds the q slots copied over [i-1,i), oldest first (modes by q), or is empty when unknown, which breaks
     * the histories. window is the ACC base window ending at anchor, newest first (2L values). Returns origin.m_valid.
     */
    bool append( std::uint64_t    index /**< [in] WFS counter. */,
                 const VectorD   &y /**< [in] Measured modes. */,
                 const MatrixD   &applied /**< [in] Copied slot commands, chronological, or empty. */,
                 const VectorD   &window /**< [in] ACC base window, newest first, or empty when invalid. */,
                 std::uint64_t    anchor /**< [in] Newest ACC counter at this decision. */,
                 std::uint64_t    segment /**< [in] ACC source segment of the window. */,
                 MultirateOrigin &origin /**< [out] Pre-allocated origin for the per-slot calculation. */ );

    /// Score the matured example with every phase predictor and, when training, update the shared factor once.
    bool learn();

    /// Extract every phase's coefficients and compute all slot gains after a change; returns true when out changed.
    bool refresh( MultirateGains &out /**< [in,out] Gain set allocated by allocate(). */ );

    /// Size a gain set for this configuration.
    void allocate( MultirateGains &gains /**< [out] Gain set. */ ) const;

    /// Size an origin for this configuration.
    void allocate( MultirateOrigin &origin /**< [out] Origin. */ ) const;

    /// Return diagnostics.
    const MultirateStatus &status() const;

    /// Return the latest matured phase r-1 regressor (for equivalence tests).
    const VectorD &regressor() const;

    /// Return the latest matured future-WFS target, newest frame first.
    const VectorD &target() const;

    /// Return Theta_k^T of the latest refresh.
    const MatrixD &coefficients( int phase /**< [in] Phase k. */ ) const;

    /// Return the shared factor (for equivalence tests).
    const InformationQR &factor() const;

    /// Compute the m-row slot gain K(k,p) from Theta_k^T: the next-slot rows of the re-solved DDSPC law.
    /** With C the future-command columns and M the state columns of Theta_k, the remaining v = F*q*m-p*m commands
     * minimize ||M x + C u||^2 + lambda ||v||^2 + mu ||P u||^2 with the p oldest current-frame slots fixed, where P
     * removes each future frame's mean command (per mode) and mu = ripple (both scaled by max(C^T C) when
     * scaledPenalty). Returns the last m rows of -H_vv^-1 [C_v^T M, C_v^T C_pre + mu P_v,pre], which apply to
     * [x; u_pre], with H_vv = C_v^T C_v + lambda I + mu P_vv. With ripple = 0 this is the original DDSPC law.
     */
    static void slot_gain( const MultirateConfig &config /**< [in] Dimensions. */,
                           const MatrixD         &thetaT /**< [in] Theta_k^T, features(k) by outputs. */,
                           int                    prefix /**< [in] Fixed slots p in [0,q). */,
                           double                 lambda /**< [in] Penalty before optional scaling. */,
                           MatrixD               &gain /**< [out] m by state(k)+p*m. */ );

    /// Calculate one slot command from coherent gains, the frame origin, k new ACC samples and p executed slots.
    static void command( const MultirateConfig &config /**< [in] Dimensions. */,
                         const MultirateGains  &gains /**< [in] Ready gain set. */,
                         const MultirateOrigin &origin /**< [in] Valid origin. */,
                         int                    phase /**< [in] ACC samples k since the anchor, < r. */,
                         int                    prefix /**< [in] Executed slots p of this frame, < q. */,
                         const double          *extras /**< [in] 2k values a_(c+1)..a_(c+k), X/Y pairs. */,
                         const double          *executed /**< [in] p*m executed commands, newest slot first. */,
                         double                *out /**< [out] m modal commands of slot p. */ );

  private:
    /// Compute K(k,p) from Theta_k^T, its precomputed C^T C and the already scaled penalty.
    static void slot_gain( const MultirateConfig &config /**< [in] Dimensions. */,
                           const MatrixD         &thetaT /**< [in] Theta_k^T. */,
                           const MatrixD         &ctc /**< [in] C^T C of Theta_k. */,
                           int                    prefix /**< [in] Fixed slots p. */,
                           double                 penalty /**< [in] Final penalty. */,
                           MatrixD               &gain /**< [out] m by state(k)+p*m. */ );

    /// Store one ACC sample in the counter-keyed ring.
    void sample( std::uint64_t counter /**< [in] ACC counter. */,
                 std::uint64_t segment /**< [in] Source segment. */,
                 const double *value /**< [in] X/Y acceleration. */ );

    /// Copy ACC samples [first, first+count) oldest first; false unless all are present in one segment.
    bool samples( std::uint64_t first /**< [in] First counter. */,
                  int           count /**< [in] Sample count. */,
                  std::uint64_t segment /**< [in] Required segment. */,
                  double       *out /**< [out] 2*count values. */ ) const;
};

} // namespace DDSPC

#endif // PC_MULTIRATE_DDSPC_HPP
