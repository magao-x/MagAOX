import os
import logging
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from windsocc.analysis.wedge_photometry import wedge_photometry
from windsocc.analysis.wind_stats import filter_sources_by_angle
from windsocc.analysis.wind_stats import tracked_df_to_frame_sources
from windsocc.analysis.wind_stats import wedge_find_peak
from windsocc.analysis.wind_stats import prep_polar_coords

def make_wedge_photometry_plots(
    mf_response_cube_paths: list,
    mf_response_cube_fnames: list,
    sources_all: list,
    output_dir: str,
    inner_radius: float,
    outer_radius: float,
    num_wedges: int,
) -> bool:
    """Collapse each MF response cube and save wedge+SEP two-panel plots."""
    os.makedirs(output_dir, exist_ok=True)
    for cube_idx, (cube_path, cube_fname) in enumerate(zip(mf_response_cube_paths, mf_response_cube_fnames)):
        cube_data = fits.getdata(cube_path)
        if cube_data.ndim != 3:
            logging.warning(f"Skipping {cube_fname}: expected 3D cube, got shape {cube_data.shape}")
            continue
        mean_stack = np.mean(cube_data, axis=0)
        wedge_counts, wedge_stds, wedge_centers, _ = wedge_photometry(
            mean_stack,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            num_wedges=num_wedges,
        )
        ny, nx = mean_stack.shape
        center_x = nx / 2.0 - 0.5
        center_y = ny / 2.0 - 0.5
        x_coords = []
        y_coords = []
        cube_sources = sources_all[cube_idx]
        cube_sources_by_frame = tracked_df_to_frame_sources(
            cube_sources=cube_sources,
            n_frames=cube_data.shape[0],
        )
        for sources in cube_sources_by_frame:
            filtered = filter_sources_by_angle(
                sources,
                center_x=center_x,
                center_y=center_y,
            )
            for source in filtered:
                x_coords.append(source["x"])
                y_coords.append(source["y"])

        nwedges = len(wedge_counts)
        theta = np.linspace(0, 2 * np.pi, nwedges, endpoint=False)
        width = 2 * np.pi / nwedges

        fig = plt.figure(figsize=(12, 6))
        ax_left = fig.add_subplot(1, 2, 1, projection="polar")
        ax_left.set_theta_zero_location("N")
        bars = ax_left.bar(
            theta,
            wedge_counts,
            width=width,
            color="orange",
            alpha=1.0,
            edgecolor="black",
            align="edge",
        )
        for bar, std in zip(bars, wedge_stds):
            bar_center = bar.get_x() + bar.get_width() / 2
            bar_height = bar.get_height()
            ax_left.errorbar(
                bar_center,
                bar_height,
                yerr=std * 3,
                fmt="--",
                color="black",
                ecolor="black",
                capsize=3,
            )
        ax_left.set_title("Wedge photometry")

        ax_right = fig.add_subplot(1, 2, 2)
        if x_coords:
            ax_right.scatter(
                x_coords,
                y_coords,
                s=12,
                alpha=0.15,
                color="k",
                marker="o",
            )
        ax_right.grid(True, linestyle="--", alpha=0.33)
        ax_right.set_xlabel("X-coords")
        ax_right.set_ylabel("Y-coords")
        ax_right.set_title("Filtered SEP detections")
        ax_right.set_aspect("equal", adjustable="box")

        fig.tight_layout()
        save_to = os.path.join(output_dir, f"{os.path.splitext(cube_fname)[0]}.png")
        fig.savefig(save_to)
        plt.close(fig)
    return True


def make_azimuthal_line_sweep_plots(
    response_map_paths: list,
    response_map_fnames: list,
    output_dir: str,
    inner_radius: float,
    outer_radius: float,
    num_angles: int = 360,
) -> bool:
    """Sweep a radial line model over 2pi and save mean-subtracted sums."""
    os.makedirs(output_dir, exist_ok=True)
    for map_path, map_fname in zip(response_map_paths, response_map_fnames):
        response_data = fits.getdata(map_path)
        if response_data.ndim == 3:
            mean_stack = np.mean(response_data, axis=0)
        elif response_data.ndim == 2:
            mean_stack = response_data
        else:
            logging.warning(
                f"Skipping line sweep for {map_fname}: expected 2D/3D data, got shape {response_data.shape}"
            )
            continue
        ny, nx = mean_stack.shape
        center_x = nx / 2.0 - 0.5
        center_y = ny / 2.0 - 0.5
        radii = np.arange(int(inner_radius), int(outer_radius) + 1)
        angles = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
        
        line_sums = np.zeros_like(angles)

        for idx, angle in enumerate(angles):
            xs = np.rint(center_x + radii * np.cos(angle)).astype(int)
            ys = np.rint(center_y + radii * np.sin(angle)).astype(int)
            valid = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
            if np.any(valid):
                line_sums[idx] = np.sum(mean_stack[ys[valid], xs[valid]])

        line_sums = line_sums - np.mean(line_sums)
        line_sums = np.clip(line_sums, 0.0, None)
        width = 2.0 * np.pi / num_angles

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"polar": True})
        ax.set_theta_zero_location("N")
        ax.tick_params(labelsize=14, labelcolor="b", pad=10)
        ax.bar(
            angles - np.pi/2,
            line_sums,
            width=width,
            color="orange",
            alpha=1.0,
            edgecolor="black",
            align="edge",
        )
        fig.tight_layout()
        save_to = os.path.join(output_dir, f"{os.path.splitext(map_fname)[0]}_line_sweep.png")
        fig.savefig(save_to)
        plt.close(fig)
    return True

def wind_stats(pa_wind, cube_data, med_stack,
                    delays_list,
                    inner_rad, outer_rad, wedge_width,
                    ap_width, sn_threshold, diam_primary,
                    diam_pupils):
    pix_scale = diam_primary / diam_pupils
    wind_at_pa_speeds = None
    subtracted_cube = cube_data - med_stack
    # print(f"print(pa_wind) -> {pa_wind}")
    wind_at_pa_speeds = []
    cc_peak_sums = []
    cc_peak_sns = []
    #Prepare the coordinate system before the looping
    ny, nx = cube_data.shape[1:]
    center = (nx / 2 - 0.5, ny / 2 - 0.5) #true image center
    # cx, cy = center
    # print(angle)

    ## Create coordinate grids.
    # y_grid, x_grid = np.indices(image.shape)
    # r = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    # theta = (np.degrees(np.arctan2(y_grid - cy, x_grid - cx)) % 360)  # 0° at right, increasing CCW
    r, theta = prep_polar_coords((ny, nx))
    theta += np.pi
    theta = np.rad2deg(theta)
    for ea, d in enumerate(delays_list):
        # print(d)
        # rprof_here = radial_profile(cube_data[ea])
        # frame_rprofsub = subtracted_cube[ea] - rprof_here
        frame_rprofsub = subtracted_cube[ea]
        # plt.imshow(frame_rprofsub)
        # plt.colorbar()
        # plt.show()
        # print(f"print(pa_wind) -> {pa_wind}")
        cc_peak_rdist, cc_peak_sum, cc_peak_sn = wedge_find_peak(
                                            image=frame_rprofsub,
                                            angle=pa_wind,
                                            r_start=inner_rad,
                                            #  OUTER_RADIUS=stimage_h // 2,
                                            r_end=outer_rad,
                                            wedge_width=wedge_width,
                                            ap_width=ap_width,
                                            sn_threshold=sn_threshold,
                                            r=r, theta=theta
                                            # debug=True,
                                            # debug_prefix=args.debug_out,
                                            )
        wind_speed = pix_scale * cc_peak_rdist / d #[meters/pixel] * [pixels] * [1/seconds]
        wind_at_pa_speeds.append(wind_speed*2)
        cc_peak_sums.append(cc_peak_sum)
        cc_peak_sns.append(cc_peak_sn)
        # peak_vals.append(cc_peak_val)
        # if cc_peak_radial_dist == cc_peak_radial_dist:
        # print(d, cc_peak_radial_dist, wind_speed)
    # where_strongest_peak = np.argmax(peak_vals)
    # mean_fast_wind_speed = np.nanmean(wind_at_pa_speeds)

    # return mean_fast_wind_speed
    wind_speeds = np.asarray(wind_at_pa_speeds)
    cc_sums = np.asarray(cc_peak_sums)
    cc_sns = np.asarray(cc_peak_sns)
    rad_dists = range(int(inner_rad), int(outer_rad - ap_width) + 1)
    return wind_speeds, cc_sums, cc_sns, rad_dists

def plot_diagnostics(speeds_list, sums_list, sns_list, delays, save_to):
    """
    Plots aperture sums and wind speeds over sample index, with S/N ratios as color.

    Parameters:
    - speeds_list (list of np.ndarray): 1–2 arrays of wind speeds.
    - sums_list (list of np.ndarray): 1–2 arrays of aperture sums.
    - sns_list (list of np.ndarray): 1–2 arrays of S/N values.
    - delays (ignored): Kept for compatibility but unused.
    - save_to (str): Path prefix for the saved plot (e.g., '/path/to/output/myfile').
    """
    plt.figure(figsize=(10, 10))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    x_vals = np.arange(len(sums_list[0]))
    apsum_markers = ["o", "s", "+", "p"]
    # Plot aperture sums with S/N as color
    for idx, (sums, snr) in enumerate(zip(sums_list, sns_list)):
        scatter = ax1.scatter(x_vals, sums, c=snr, cmap='viridis', marker=apsum_markers[idx],
                              label=f'Aperture Sum {idx+1}', edgecolor='k', s=30)

    # Plot wind speeds
    for idx, speeds in enumerate(speeds_list):
        ax2.plot(x_vals, speeds, linestyle='--', label=f'Wind Speed {idx+1}', alpha=0.8)

    # Labels and title
    ax1.set_xlabel("Time Index")
    ax1.set_ylabel("Aperture Sum")
    ax2.set_ylabel("Wind Speed [m/s]")
    plt.title(f"Resolved wind layers: {len(sums_list)}")

    # Legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=[ax1, ax2])
    cbar.set_label("S/N Aperture Sum")

    plt.grid(True)
    # plt.tight_layout()

    save_path = f"{save_to}_diag.png"
    plt.savefig(save_path)
    plt.close()

def find_top_valid_peaks(wedge_counts, sigma=5.0, min_distance=5):
    """
    Find the indices of the two tallest peaks in 1D array of counts
    from wedge photometry.

    Parameters:
    -----------
    wedge_counts : np.ndarray
        Input 1D array of float values (e.g., wind speed over direction).
    sigma : float
        Sigma threshold for peak height criterion.
    min_distance : int
        Minimum number of indices between two peaks (inter-peak distance).

    Returns:
    --------
    peak_indices : np.ndarray
        Array of two values: indices of the tallest peaks.
        If only one peak is found, returns [index, np.nan].
        If no peaks are found, returns [np.nan, np.nan].
    """
    stddev = np.std(wedge_counts)

    # Find peaks in the original data, but only at unmasked locations
    valid_peaks, _ = find_peaks(wedge_counts, height=sigma*stddev, distance=min_distance)

    if len(valid_peaks) == 0:
        print("No wind detected.")
        return None, np.nan
    else:
        # Sort peaks by height
        peak_heights = wedge_counts[valid_peaks]
        # print(peak_heights)
        sorted_indices = np.argsort(peak_heights)[::-1]  # Descending
        top_valid = [valid_peaks[i] for i in sorted_indices[:3]]
        # print(top_two)
        #  print(wedge_counts[top_two])
        print(f"Detected {len(valid_peaks)} winds.")
        # sys.exit()
        return np.asarray(top_valid), None

def check_and_process_wind_pas(fastest_wind_pas, wedge_centers,
                               pa_offset, do_pa_flip):
    pas_fastest_wind = []
    uncorrected_pas = []
    # print(np.rad2deg(wedge_centers))
    # sys.exit()
    # nans = [] #null measurements
    # There better be only 2 directions...
    # Not true anymore! 07/10/2025
    if do_pa_flip:
        pa_offset_camsci = pa_offset
    else:
        pa_offset_camsci = pa_offset
    # pa_offset_coord = -90 #to get back to north up, east left


    for pa in fastest_wind_pas:
        pa_fast_wind = wedge_centers[pa]
        # pa_fast_wind = np.rad2deg(pa_fast_wind) + pa_offset_coord + pa_offset_camsci
        # print(f"check_and_process_wind_pas: print(pa) -> {np.rad2deg(pa_fast_wind)} (uncorrected)")
        pa_fast_wind = np.rad2deg(pa_fast_wind) #+ pa_offset_coord
        pa_fast_wind_offset = round(pa_fast_wind + pa_offset_camsci)
        pa_fast_wind_corrected = np.mod(pa_fast_wind_offset, 360.)
        uncorrected_pas.append(pa_fast_wind)
        pas_fastest_wind.append(round(pa_fast_wind_corrected))

    
    return np.asarray(pas_fastest_wind), np.asarray(uncorrected_pas)