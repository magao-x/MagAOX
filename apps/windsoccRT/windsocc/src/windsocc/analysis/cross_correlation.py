'''
TODO create a new function to remove the bias from the cross-correlation maps.
The bias is the cross-correlation of the two circle apertures of the same size.
'''

import os
import sys
import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve

# Import your circular template extraction function.
# from analysis.templates import make_circular_template

def compute_aperture_bias(median_image, fft_pad_shape=None):
    '''
    Compute the bias from the cross-correlation of the two circle apertures of the same size.
    - Compute the cross-correlation of the median image with itself.
    - Return the resulting CC map as the bias.
    
    Parameters:
        median_image: np.ndarray
            2D array representing the median image
        fft_pad_shape: tuple, optional
            (height, width) tuple for output shape. If None, uses mode="full" (default).
    Returns:
        bias: np.ndarray
            Cross-correlation map with shape matching fft_pad_shape or (2*h-1, 2*w-1) if None
    '''
    aperture_function = median_image
    h, w = median_image.shape
    
    if fft_pad_shape is not None:
        output_h, output_w = fft_pad_shape
        # Compute cross-correlation using FFT with specified output shape
        # This matches the shape used in compute_all_delays_welch_optimized
        fft_a = np.fft.rfft2(aperture_function, s=(output_h, output_w))
        # fft_a_reversed = np.fft.rfft2(aperture_function[::-1, ::-1], s=(output_h, output_w))
        # bias_fft = fft_a * fft_a_reversed
        bias_fft = fft_a * np.conj(fft_a)
        bias = np.fft.irfft2(bias_fft, s=(output_h, output_w))
        # Apply fftshift to center the zero-lag (consistent with CC maps)
        bias = np.fft.fftshift(bias)
    else:
        # cross-correlate (reverse the second image) with mode="full"
        bias = fftconvolve(aperture_function, aperture_function[::-1, ::-1], mode="full")
    
    return bias

def load_reduced_series(data_dir):
    """
    Load all reduced FITS cubes (each of shape (512, 120, 120)) from a directory,
    sort them, and concatenate along the time axis.
    """
    file_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if (f.endswith('.fits') and f.startswith('camwfs_'))])
    series = []
    file_skips = 0
    expect_cube_length = None
    for file in file_list:
        with fits.open(file) as hdul:
            cube = hdul[0].data  # Expecting shape (512, 120, 120)
            if expect_cube_length is None:
                expect_cube_length = cube.shape[0]
            else:
                if not expect_cube_length == cube.shape[0]:
                    file_skips += 1
                    continue
            series.append(cube)
    if series:
        return np.concatenate(series, axis=0), expect_cube_length, file_skips
    else:
        raise ValueError(f"No FITS files found in {data_dir}.")


def process_segment_fft_from_precomputed(fft_time_series, fft_time_series_conj,
                                         segment_start, segment_length, delay,
                                         output_h, output_w):
    """
    Compute cross-correlation for one segment using pre-computed FFTs and their conjugates.
    
    Parameters
    ----------
    fft_time_series : ndarray
        Pre-computed FFT array of shape (total_frames, h_rfft, w_rfft) from rfft2.
    fft_time_series_conj : ndarray
        Pre-computed conjugate of FFT array (same shape as fft_time_series).
    segment_start : int
        Starting frame index for this segment.
    segment_length : int
        Length of the segment in frames.
    delay : int
        Delay (in frames) for cross-correlation.
    output_h : int
        Output height for IFFT (from fft_pad_shape).
    output_w : int
        Output width for IFFT (from fft_pad_shape).
          
    Returns
    -------
    segment_cc : ndarray or None
        The averaged net cross-correlation map for this segment, or None if no valid pairs.
    """
    # Extract FFT slice for this segment
    segment_end = segment_start + segment_length
    fft_segment = fft_time_series[segment_start:segment_end]
    fft_segment_conj = fft_time_series_conj[segment_start:segment_end]
    
    n_frames_in_segment = len(fft_segment)
    
    # Check if we have enough frames for this delay
    if n_frames_in_segment <= delay:
        return None
    
    # Accumulate cross-correlation products in frequency domain
    cc_sum_fft = np.zeros_like(fft_segment[0], dtype=complex)
    pair_count = 0
    
    for i in range(n_frames_in_segment - delay):
        # Cross-correlation: FFT(frame[i]) * conj(FFT(frame[i+delay]))
        # Using pre-computed conjugate to avoid repeated np.conj() calls
        cc_sum_fft += fft_segment[i] * fft_segment_conj[i + delay]
        pair_count += 1
    
    if pair_count == 0:
        return None
    
    # Average in frequency domain
    cc_avg_fft = cc_sum_fft / pair_count
    
    # Single IFFT to get final cross-correlation map (using irfft2 since we used rfft2)
    cc_map = np.fft.irfft2(cc_avg_fft, s=(output_h, output_w))
    
    # Apply fftshift to center the zero-lag at the center of the array
    # (pixel [0,0] moves from top-left to center)
    cc_map = np.fft.fftshift(cc_map)
    
    return cc_map


def process_segment_fft_optimized(segment, delay, fft_pad_shape=None):
    """
    Process one segment using optimized FFT-based cross-correlation.
    Pre-computes FFTs for all frames, accumulates products in frequency domain,
    and performs a single IFFT at the end.
    
    Parameters
    ----------
    segment : ndarray
        3D array of shape (n_frames, height, width) representing the segment.
    delay : int
        Delay (in frames) for cross-correlation.
          
    Returns
    -------
    segment_cc : ndarray
        The averaged net cross-correlation map for this segment.
    """
    n_frames, h, w = segment.shape
    
    # Remove static pattern
    static_pattern = np.median(segment, axis=0)
    segment = segment - static_pattern
    
    
    if fft_pad_shape is not None:
        output_h = fft_pad_shape[0]
        output_w = fft_pad_shape[1]
    else:
        # Default to (h + h - 1, w + w - 1) = (2*h - 1, 2*w - 1)
        output_h = 2 * h - 1
        output_w = 2 * w - 1
    # Pre-compute 2D FFTs for all frames (full complex FFT for proper cross-correlation handling)
    # Using fft2 instead of rfft2 to handle the reversed frame correctly
    # Note: This could be optimized further with rfft2, but requires careful handling
    # of the reversal in frequency domain
    fft_frames = np.fft.rfft2(segment, s=(output_h, output_w), axes=(1, 2))
    # Shape: (n_frames, output_h, output_w)
    
    # Also pre-compute FFTs of reversed frames for cross-correlation
    # frame_td[::-1,::-1] reversed in space
    # fft_frames_reversed = np.fft.fft2(segment[:, ::-1, ::-1], s=(output_h, output_w), axes=(1, 2))
    # Shape: (n_frames, output_h, output_w)
    
    # Accumulate products in frequency domain
    cc_sum_fft = np.zeros_like(fft_frames[0], dtype=complex)
    pair_count = 0
    
    for i in range(n_frames - delay):
        # frame_t_fft = fft_frames[i]
        # Use the reversed FFT for the delayed frame
        # frame_td_reversed_fft = fft_frames_reversed[i + delay]
        
        # Cross-correlation: multiply in frequency domain
        # cc_sum_fft += frame_t_fft * frame_td_reversed_fft
        prod = fft_frames[:-i] * np.conj(fft_frames[i:])
        pair_count += 1
    
    if pair_count == 0:
        return None
    
    # Average in frequency domain
    cc_avg_fft = cc_sum_fft / pair_count
    
    # Single IFFT to get final cross-correlation map (take real part since input was real)
    cc_map = np.fft.ifft2(cc_avg_fft, s=(output_h, output_w)).real
    
    return cc_map

def compute_all_delays_welch_optimized(time_series, frames_per_cube, delays, 
                                       segment_cube_count=22, overlap_fraction=0.5, 
                                       fft_pad_shape=None):
    """
    Compute cross-correlation maps for multiple delays using optimized FFT pre-computation.
    
    Pre-computes FFTs for the entire time_series once, then processes all delays
    and segments using the pre-computed FFTs. This eliminates redundant FFT computations.
    
    Parameters
    ----------
    time_series : ndarray
        Full time series array of shape (n_frames, height, width).
    frames_per_cube : int
        Number of frames per cube.
    delays : list or array
        List of delay values to process.
    segment_cube_count : int, optional
        Number of cubes per segment (default: 22).
    overlap_fraction : float, optional
        Fractional overlap between segments (default: 0.5).
    fft_pad_shape : tuple, optional
        Optional (height, width) tuple for FFT padding. If None, uses (2*h-1, 2*w-1).
    
    Returns
    -------
    cc_maps : list
        List of cross-correlation maps, one per delay value (in order of delays input).
    
    Ex. parameters
     - N_frames per cube 64
     - 4 ms exptime per frame
     - ~10 seconds of cube data per subdirectory
    """
    import gc

    total_frames, h, w = time_series.shape
    
    # Remove static pattern from entire time_series
    static_pattern = np.median(time_series, axis=0)
    time_series = time_series - static_pattern
    
    # Determine FFT output shape
    if fft_pad_shape is not None:
        output_h, output_w = fft_pad_shape
    else:
        # Default to (h + h - 1, w + w - 1) = (2*h - 1, 2*w - 1) for full mode
        output_h = 2 * h - 1
        output_w = 2 * w - 1
    
    # Pre-compute FFTs for entire time_series once using rfft2 (efficient for real inputs)
    fft_time_series = np.fft.rfft2(time_series, s=(output_h, output_w), axes=(1, 2))

    del time_series
    gc.collect()
    # Shape: (total_frames, output_h, output_w_rfft) where output_w_rfft = output_w//2 + 1
    
    # Pre-compute conjugate once to avoid repeated np.conj() calls in loops
    fft_time_series_conj = np.conj(fft_time_series)
    
    # Define segment parameters
    segment_length = segment_cube_count * frames_per_cube
    step = int(segment_length * (1 - overlap_fraction))
    max_delay = max(delays)
    
    # Define valid segment start indices (account for max delay)
    valid_starts = []
    for start in range(0, total_frames - segment_length - max_delay + 1, step):
        valid_starts.append(start)
    
    if len(valid_starts) == 0:
        valid = (total_frames - segment_length - max_delay) / step
        raise ValueError(f"No valid segments found. Check segment and delay parameters: \
                          \nTotal frames: {total_frames}, Segment length: {segment_length}, \
                          \nMax delay: {max_delay}, Welch step: {step}, \
                          \n(Total frames - Segment length - Max delay) / Step = {valid}")
    
    # Process each delay
    cc_maps = []
    for delay in delays:
        segment_cc_maps = []
        
        # Process all segments for this delay
        for start in valid_starts:
            result = process_segment_fft_from_precomputed(
                fft_time_series, fft_time_series_conj, start, segment_length, delay, output_h, output_w
            )
            if result is not None:
                segment_cc_maps.append(result)
        
        if len(segment_cc_maps) == 0:
            raise ValueError(f"No segments were processed for delay {delay}. Check delay and segment parameters.")
        
        # Average across segments for this delay
        master_cc_map = np.mean(segment_cc_maps, axis=0)
        cc_maps.append(master_cc_map)
    
    return cc_maps


def compute_master_cc_map_welch_shared_fft(time_series, frames_per_cube, delay, 
                                segment_cube_count=22, overlap_fraction=0.5, fft_pad_shape=None):
    """
    Compute the master cross-correlation map using Welch's method with optimized FFT computation.
    Processes segments sequentially with pre-computed FFTs for efficiency.
    
    NOTE: This function is kept for backward compatibility. For processing multiple delays,
    use compute_all_delays_welch_optimized() instead, which is more efficient.
    
    Parameters
    ----------
    time_series : ndarray
        Concatenated array of shape (N_total, height, width).
    delay : int
        Delay (in frames) between frames for cross-correlation.
    segment_cube_count : int, optional
        Number of cubes per segment (default: 22, where each cube is 512 frames).
    overlap_fraction : float, optional
        Fractional overlap between segments (default: 0.5 for 50% overlap).
    
    Returns
    -------
    master_cc_map : ndarray
        The final averaged cross-correlation map (2D array).

    Ex. parameters
     - N_frames per cube 64
     - 4 ms exptime per frame
     - ~10 seconds of cube data per subdirectory
    """
    segment_length = segment_cube_count * frames_per_cube # Ex. segment_length 1408
    step = int(segment_length * (1 - overlap_fraction)) # Ex. step 704
    total_frames = time_series.shape[0] # Ex. total_frames 2304 with 36 cubes
    # Process segments sequentially
    segment_results = []
    # Ex. range(0, 2304 - 1408 - 1) -> range(0, 895) for delay 1 (normally the min)
    # Ex. range(0, 2304 - 1408 - 251) -> range(0, 645) for delay 251 (normally the max)
    for start in range(0, total_frames - segment_length - delay + 1, step):
        segment = time_series[start : start + segment_length]
        result = process_segment_fft_optimized(segment, delay, fft_pad_shape)
        if result is not None:
            segment_results.append(result)
    
    if len(segment_results) == 0:
        raise ValueError("No segments were processed. Check delay and segment parameters.")
    
    # Average across segments
    master_cc_map = np.mean(segment_results, axis=0)
    
    return master_cc_map
