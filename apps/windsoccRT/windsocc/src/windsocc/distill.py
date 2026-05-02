"""Distill stage for the WindsoCC pipeline.

TODO refactoring: 
- move all but main function and logic to core/distill.py
- rename this script to ws_distill.py
"""

#!/usr/bin/env python3
import os
import re
import glob
import argparse
from windsocc.io.config_handling import parse_config_file
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from skimage.feature import match_template

DEFAULT_TEMPLATE_SIZE = 65

# Batch UTC time embedded in ``group_suffix`` / distill ``suffix`` (see ``realtime.format_batch_timestamp``).
# Digit-delimited (not ``\\b``) so a trailing underscore after microseconds still matches.
_BATCH_UTC_TOKEN_RE = re.compile(r"(?<!\d)(\d{8}T\d{6}\d{6})(?!\d)")
# Compact variant without ``T`` (and optional ``_00000`` trailer), e.g. ``camwfs_20230313071857408144000``.
_BATCH_UTC_COMPACT_RE = re.compile(r"(?<!\d)(\d{8})(\d{6})(\d*)(?!\d)")


def resolve_parangs_lookup_path(config_params, directory):
    """
    Resolve ``PARANGS_LOOKUP`` from config to an absolute path if the file exists.

    Returns ``None`` when no lookup file is configured or the configured path is
    missing. Callers should skip derotation in that case.
    """
    raw = config_params.get("PARANGS_LOOKUP")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        logging.warning(
            "PARANGS_LOOKUP is not set; skipping distill derotation."
        )
        return None
    if not isinstance(raw, str):
        logging.warning(
            "PARANGS_LOOKUP must be a string path, got %s; skipping distill derotation.",
            type(raw).__name__,
        )
        return None
    path = raw.strip()
    full = path if os.path.isabs(path) else os.path.join(directory, path)
    if not os.path.isfile(full):
        logging.warning(
            "PARANGS_LOOKUP file not found at %r; skipping distill derotation.",
            full,
        )
        return None
    return full


def _parse_iso_timestamp_to_utc_seconds(ts_str):
    """
    Parse an ISO-like timestamp string to Unix seconds (UTC).

    Supports nanosecond (or arbitrary-length) fractional seconds; ``datetime.fromisoformat``
    only accepts up to 6 fractional digits, so sub-microsecond tails are parsed manually.
    Naive times are treated as UTC. Optional trailing ``Z`` or ``±HH:MM`` offsets are honored.
    """
    text = str(ts_str).strip()
    if not text:
        raise ValueError("empty timestamp")
    text = text.replace(" ", "T", 1)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    m_tz = re.search(r"([+-])(\d{2}):(\d{2})$", text)
    if m_tz:
        sign = 1 if m_tz.group(1) == "+" else -1
        offset_sec = sign * (int(m_tz.group(2)) * 3600 + int(m_tz.group(3)) * 60)
        core = text[: m_tz.start()]
        tzinfo = timezone(timedelta(seconds=offset_sec))
    else:
        core = text
        tzinfo = timezone.utc

    if "." in core:
        main, rest = core.split(".", 1)
        digits = "".join(c for c in rest if c.isdigit())
        if digits:
            frac = Decimal(digits) / (Decimal(10) ** len(digits))
        else:
            frac = Decimal(0)
    else:
        main = core
        frac = Decimal(0)

    dt = datetime.strptime(main, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tzinfo)
    return dt.timestamp() + float(frac)


def load_parangs_lookup(path):
    """
    Load PARANG vs time from a TSV (tab-separated) with optional header row.

    Expects at least two columns: PARANG (degrees) and timestamp (ISO-like).
    Rows are sorted by time; duplicate timestamps keep the last PARANG value.

    Returns:
        tuple: ``(xp, fp)`` as ``numpy.ndarray`` of increasing times and PARANG values for ``np.interp``.
    """
    times = []
    parangs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"PARANGS_LOOKUP line must have at least two tab-separated columns: {line!r}"
                )
            if parts[0].strip().upper() == "PARANG" and parts[1].strip().lower() == "timestamp":
                continue
            try:
                parang = float(parts[0].strip())
                t_sec = _parse_iso_timestamp_to_utc_seconds(parts[1])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Could not parse PARANG/timestamp from PARANGS_LOOKUP line: {line!r}"
                ) from exc
            times.append(t_sec)
            parangs.append(parang)
    if not times:
        raise ValueError(f"No data rows found in PARANGS_LOOKUP: {path!r}")
    order = np.argsort(times, kind="mergesort")
    xp = np.asarray(times, dtype=np.float64)[order]
    fp = np.asarray(parangs, dtype=np.float64)[order]
    # Collapse duplicate times (np.interp requires strictly increasing xp).
    if np.any(np.diff(xp) == 0):
        uniq_xp = []
        uniq_fp = []
        for x, p in zip(xp, fp):
            if uniq_xp and uniq_xp[-1] == x:
                uniq_fp[-1] = p
            else:
                uniq_xp.append(x)
                uniq_fp.append(p)
        xp = np.asarray(uniq_xp, dtype=np.float64)
        fp = np.asarray(uniq_fp, dtype=np.float64)
    return xp, fp


def parse_batch_utc_seconds_from_suffix(suffix):
    """
    Parse the UTC batch instant embedded in ``suffix``.

    Supports:

    - ``YYYYMMDDTHHMMSSffffff`` as produced by ``realtime.format_batch_timestamp``.
    - Compact ``YYYYMMDDHHMMSS`` + optional fractional digits (no ``T``), e.g.
      ``camwfs_20230313071857408144000`` (fractional tail interpreted as
      ``int(frac) / 10**len(frac)`` seconds).
    """
    m = _BATCH_UTC_TOKEN_RE.search(suffix)
    if m:
        token = m.group(1)
        dt = datetime.strptime(token, "%Y%m%dT%H%M%S%f").replace(tzinfo=timezone.utc)
        return dt.timestamp()

    # Use the rightmost compact match when multiple digit runs appear in ``suffix``.
    matches = list(_BATCH_UTC_COMPACT_RE.finditer(suffix))
    if not matches:
        raise ValueError(
            "Could not find batch UTC time in distill suffix (expected "
            "YYYYMMDDTHHMMSSffffff or YYYYMMDDHHMMSS plus optional fractional digits): "
            f"{suffix!r}"
        )
    m2 = matches[-1]
    ymd, hms, frac = m2.group(1), m2.group(2), m2.group(3)
    if len(ymd) != 8 or len(hms) != 6:
        raise ValueError(f"Invalid compact batch date/time in suffix: {suffix!r}")
    base = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    if not frac:
        return base.timestamp()
    subsec = int(frac, 10) / (10.0 ** len(frac))
    return base.timestamp() + subsec


def derotate_cc_cube(cube, angle_deg, order=3):
    """
    Rotate each 2D frame of a CC cube by ``angle_deg`` using ``scipy.ndimage.rotate``.

    Camwfs parity uses negated PARANG at the call site (pass ``-parang`` here).
    """
    if cube.ndim != 3:
        raise ValueError(f"CC cube must be 3D, got shape {cube.shape}.")
    out = np.empty_like(cube, dtype=np.float64)
    for i in range(cube.shape[0]):
        out[i] = rotate(
            np.asarray(cube[i], dtype=np.float64),
            float(angle_deg),
            reshape=False,
            order=order,
            prefilter=False,
        )
    return out


def gaussian_kernel(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def group_files_by_suffix(directory):
    """
    Group all FITS files in the directory by the filename after the initial prefix.
    The prefixes are assumed to be one of: "ul_", "ur_", "ll_", "lr_".
    
    Parameters:
        directory (str): The path to the directory containing the FITS files.
    
    Returns:
        dict: A dictionary where keys are the common filename suffix (without prefix) and values 
              are lists of full file paths for that group.
    """
    cc_map_groups = {}
    bias_groups = {}
    # Define the expected prefixes.
    prefixes = ("ul_", "ur_", "ll_", "lr_")
    bias_suffix = "bias.fits"
    
    # Find all .fits files in the directory, then find the corresponding bias file.
    fits_files = glob.glob(os.path.join(directory, "*.fits"))
    for file_path in fits_files:
        fname = os.path.basename(file_path)
        # Check which prefix the file starts with
        for prefix in prefixes: # ul_, ur_, ll_, lr_
            # we know that each CC map has a corresponding bias file with the same date stamp.
            if fname.startswith(prefix):
                # Remove the prefix to get the common suffix.
                fname_sans_prefix = fname[len(prefix):]
                fname_sans_fits = os.path.splitext(fname)[0]
                fname_camtimesplit = fname_sans_fits.split("_")[1:-1]
                fname_camtime = "_".join(fname_camtimesplit)
                cc_map_path = os.path.join(directory, f"{prefix}{fname_sans_prefix}")
                bias_path = os.path.join(directory, f"biases/{prefix}{fname_camtime}_{bias_suffix}")
                cc_map_groups.setdefault(fname_camtime,[]).append(cc_map_path)
                bias_groups.setdefault(fname_camtime,[]).append(bias_path)
                break  # stop checking prefixes after the first match
    return cc_map_groups, bias_groups

def load_and_average(file_list):
    """
    Load a list of FITS cubes and average them.
    
    Parameters:
      file_list (list): List of FITS file paths. They should all have the same dimensions.
    
    Returns:
      tuple: (averaged_cube, header)
    """
    cubes = []
    header = None
    for file_path in file_list:
        with fits.open(file_path) as hdul:
            data = hdul[0].data  # assume the cube is in the primary HDU
            cubes.append(data)
            # Use header from the first file (adjust if needed)
            if header is None:
                header = hdul[0].header

    # Convert the list of arrays to a single array and average across the new axis (i.e. from the set of 4 cubes)
    cubes_array = np.array(cubes)  # shape should be (4, 251, 120, 120)
    averaged_cube = np.mean(cubes_array, axis=0)  # resulting shape: (251, 120, 120)
    
    return averaged_cube, header

def write_cube(output_path, cube, header):
    """Write out a FITS cube with the provided header."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    fits.writeto(output_path, cube, header=header, overwrite=True)
    logging.info(f"Wrote file to: {output_path}")


def save_png(output_path, image, title=None, cmap="viridis"):
    """Save a 2D image to PNG with robust percentile scaling."""
    outdir = os.path.dirname(output_path)
    os.makedirs(outdir, exist_ok=True)
    finite = np.isfinite(image)
    if np.any(finite):
        vmin, vmax = np.percentile(image[finite], [1, 99])
    else:
        vmin, vmax = None, None
    fig, ax = plt.subplots()
    im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logging.info(f"Wrote file to: {output_path}")

def apply_unsharp_mask_cube(cube, fwhm_pixels=5.0):
    """High-pass filter a cube by subtracting a Gaussian blur per frame."""
    if fwhm_pixels is None or fwhm_pixels <= 0:
        return cube
    sigma = fwhm_pixels / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # debug: view the kernel
    # kernel = gaussian_kernel(cube.shape[1], sigma)
    # plt.imshow(kernel, cmap='viridis')
    # plt.show()
    # exit()
    blurred = gaussian_filter(cube, sigma=(0, sigma, sigma))
    return cube - blurred

def build_template(
    frame,
    # center_coord: tuple[int, int] = (33, 33),
    size: int = 65
    ) -> np.ndarray:
    """Build a normalized matched-filter template from (probably) 
    the first frame of the cube.
    """
    if frame.ndim != 2:
        raise ValueError(f"Frame must be 2D, got shape {frame.shape}.")
    ny, nx = frame.shape
    if size <= 0 or size > min(ny, nx):
        raise ValueError(
            f"Template size {size} invalid for frame shape {frame.shape}."
        )
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    half = size // 2
    start_y = int(round(center_y)) - half
    start_x = int(round(center_x)) - half
    end_y = start_y + size
    end_x = start_x + size
    start_y = max(start_y, 0)
    start_x = max(start_x, 0)
    end_y = min(end_y, ny)
    template = frame[start_y:end_y, start_x:end_x].astype(float)
    template -= np.nanmedian(template)
    scale = np.nanstd(template)
    if scale > 0:
        template /= scale
    return template

def compute_mf_response_cube(cube, template):
    """Compute the matched-filter response cube."""
    if cube.ndim != 3 or template.ndim != 2:
        raise ValueError(f"Cube must be 3D, template must be 2D, got shapes {cube.shape} and {template.shape}.")
    response_cube = np.empty_like(cube, dtype=np.float64)
    for idx in range(cube.shape[0]):
        response_cube[idx] = match_template(cube[idx], template, pad_input=True)
    return response_cube

def compute_snr_cube(cube):
    """Compute SNR plus last-quarter and whole-cube noise maps."""
    if cube.ndim != 3:
        raise ValueError(f"Cube must be 3D, got shape {cube.shape}.")
    n_frames = cube.shape[0]
    last_quarter_start = int(np.floor(0.75 * n_frames))
    last_quarter = cube[last_quarter_start:]
    if last_quarter.size == 0:
        last_quarter = cube

    error_map = np.nanstd(last_quarter, axis=0)
    error_map_wholecube = np.nanstd(cube, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_cube = np.where(error_map > 0, cube / error_map, 0.0)
    return snr_cube, error_map, error_map_wholecube

def convolve_cube_with_kernel(cube, kernel_2d):
    """Convolve each frame of a cube with a 2D kernel."""
    if kernel_2d.ndim != 2:
        raise ValueError(f"Kernel must be 2D, got shape {kernel_2d.shape}.")
    convolved = np.empty_like(cube, dtype=np.float64)
    for idx in range(cube.shape[0]):
        convolved[idx] = fftconvolve(cube[idx], kernel_2d, mode="same")
    return convolved

def normalize_kernel_peak(kernel_2d):
    """Normalize a 2D kernel to have peak value 1."""
    kernel = np.array(kernel_2d, dtype=np.float64, copy=True)
    peak = np.max(kernel)
    if peak > 0:
        kernel /= peak
    return kernel


def ensure_distill_output_dirs(distilled_dir):
    """Create the standard output directories for distill products."""
    os.makedirs(distilled_dir, exist_ok=True)
    os.makedirs(os.path.join(distilled_dir, "biases"), exist_ok=True)
    os.makedirs(os.path.join(distilled_dir, "sn_maps"), exist_ok=True)
    os.makedirs(os.path.join(distilled_dir, "mf_templates"), exist_ok=True)
    os.makedirs(os.path.join(distilled_dir, "mf_response_cubes"), exist_ok=True)


def process_distill_group(
    suffix,
    averaged_cube,
    averaged_bias,
    header,
    distilled_dir,
    template_size,
    save_pngs=True,
    hp_filter_fwhm=None,
    cc_header=None,
):
    """Write one distill group from already-averaged xcorr products."""
    out_hdr = cc_header if cc_header is not None else header
    output_stem = os.path.join(distilled_dir, suffix)
    output_path = output_stem + ".fits"
    output_path_unsharp = output_stem + "_unsharp.fits"

    bias_output_path = os.path.join(distilled_dir, "biases", f"{suffix}_bias.fits")
    write_cube(bias_output_path, averaged_bias, header)

    mf_template = build_template(averaged_cube[0], size=template_size)
    if hp_filter_fwhm is not None:
        logging.info(f"Applying high-pass filter with FWHM {hp_filter_fwhm} pixels.")
        high_pass_cube = apply_unsharp_mask_cube(averaged_cube, fwhm_pixels=hp_filter_fwhm)
    else:
        logging.warning("No high-pass filter FWHM provided; using default of 5.0 pixels.")
        high_pass_cube = apply_unsharp_mask_cube(averaged_cube)
    averaged_cube_output_path = os.path.join(distilled_dir, f"{suffix}.fits")
    high_pass_cube_output_path = os.path.join(distilled_dir, f"{suffix}_unsharp.fits")
    write_cube(averaged_cube_output_path, averaged_cube, out_hdr)
    write_cube(high_pass_cube_output_path, high_pass_cube, out_hdr)
    mf_template_unsharp = build_template(high_pass_cube[0], size=template_size)
    mf_output_path = os.path.join(distilled_dir, "mf_templates", f"{suffix}_mf_template.fits")
    mf_output_path_unsharp = os.path.join(distilled_dir, "mf_templates", f"{suffix}_mf_template_unsharp.fits")
    write_cube(mf_output_path_unsharp, mf_template_unsharp, out_hdr)
    write_cube(mf_output_path, mf_template, out_hdr)

    mf_response_cube = compute_mf_response_cube(averaged_cube, mf_template)
    mf_response_unsharp_cube = compute_mf_response_cube(high_pass_cube, mf_template_unsharp)
    # clamped_mf_response_unsharp_cube = np.clip(mf_response_unsharp_cube, 0, None)
    mf_response_output_path = os.path.join(
        distilled_dir, "mf_response_cubes", f"{suffix}_mf_response.fits"
    )
    mf_response_unsharp_output_path = os.path.join(
        distilled_dir, "mf_response_cubes", f"{suffix}_mf_response_unsharp.fits"
    )
    write_cube(mf_response_output_path, mf_response_cube, out_hdr)
    write_cube(mf_response_unsharp_output_path, mf_response_unsharp_cube, out_hdr)

    # collapsed_mf_response = np.mean(mf_response_cube, axis=0)
    # collapsed_mf_response_unsharp = np.mean(mf_response_unsharp_cube, axis=0)
    # collapsed_mf_response_output_path = os.path.join(
    #     distilled_dir, "mf_response_cubes", f"{suffix}_mf_response_mean_collapsed.fits"
    # )
    # collapsed_mf_response_unsharp_output_path = os.path.join(
    #     distilled_dir,
    #     "mf_response_cubes",
    #     f"{suffix}_mf_response_unsharp_mean_collapsed.fits",
    # )
    # write_cube(collapsed_mf_response_output_path, collapsed_mf_response, header)
    # write_cube(
    #     collapsed_mf_response_unsharp_output_path,
    #     collapsed_mf_response_unsharp,
    #     header,
    # )
    # if save_pngs:
    #     save_png(
    #         collapsed_mf_response_output_path.replace(".fits", ".png"),
    #         collapsed_mf_response,
    #         title=f"{suffix} MF response mean-collapsed",
    #         cmap="viridis",
    #     )
    #     save_png(
    #         collapsed_mf_response_unsharp_output_path.replace(".fits", ".png"),
    #         collapsed_mf_response_unsharp,
    #         title=f"{suffix} MF response unsharp mean-collapsed",
    #         cmap="viridis",
    #     )

    # write_cube(output_path_unsharp, high_pass_cube, header)
    # write_cube(output_path, averaged_cube, header)

    snr_map, _, _ = compute_snr_cube(mf_response_cube)
    (
        snr_map_unsharp,
        _,
        _,
    ) = compute_snr_cube(mf_response_unsharp_cube)
    _, error_map, error_map_wholecube = compute_snr_cube(averaged_cube)
    _, error_map_unsharp, error_map_unsharp_wholecube = compute_snr_cube(high_pass_cube)
    snr_output_path = os.path.join(distilled_dir, "sn_maps", f"{suffix}_snr.fits")
    snr_map_unsharp_output_path = os.path.join(
        distilled_dir, "sn_maps", f"{suffix}_snr_unsharp.fits"
    )
    error_map_output_path = os.path.join(distilled_dir, "noise_maps", f"{suffix}_error_map.fits")
    error_map_unsharp_output_path = os.path.join(distilled_dir, "noise_maps", f"{suffix}_error_map_unsharp.fits")
    error_map_wholecube_output_path = os.path.join(
        distilled_dir,
        "noise_maps",
        f"{suffix}_error_map_wholecube.fits",
    )
    error_map_unsharp_wholecube_output_path = os.path.join(
        distilled_dir,
        "noise_maps",
        f"{suffix}_error_map_unsharp_wholecube.fits",
    )
    write_cube(error_map_output_path, error_map, out_hdr)
    write_cube(error_map_unsharp_output_path, error_map_unsharp, out_hdr)
    write_cube(error_map_wholecube_output_path, error_map_wholecube, out_hdr)
    write_cube(
        error_map_unsharp_wholecube_output_path,
        error_map_unsharp_wholecube,
        out_hdr,
    )
    write_cube(snr_output_path, snr_map, out_hdr)
    write_cube(snr_map_unsharp_output_path, snr_map_unsharp, out_hdr)


def run_distill_stage(directory, config_params=None, save_pngs=True):
    """Run the distill stage for a single batch directory."""
    if config_params is None:
        config_path = os.path.join(directory, "ws_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No `ws_config.yaml` file found at {config_path}. "
                "The pipeline is being run out of order or in the wrong directory."
            )
        config_params = parse_config_file(config_path)

    xcorr_dir_name = config_params.get("XCORR_DIR", "xcorr_results")
    xcorr_dir = xcorr_dir_name if os.path.isabs(xcorr_dir_name) else os.path.join(directory, xcorr_dir_name)
    cc_map_groups, bias_groups = group_files_by_suffix(xcorr_dir)

    distilled_dir_name = config_params.get("DISTILL_DIR", "distill_results")
    distilled_dir = (
        distilled_dir_name
        if os.path.isabs(distilled_dir_name)
        else os.path.join(directory, distilled_dir_name)
    )
    ensure_distill_output_dirs(distilled_dir)
    if "parangs.txt" in os.listdir(directory):
        parangs_path = os.path.join(directory, "parangs.txt")
    else:
        parangs_path = resolve_parangs_lookup_path(config_params, directory)
    parangs_lookup = load_parangs_lookup(parangs_path) if parangs_path is not None else None

    failed_groups = []
    processed_groups = []
    for suffix, file_list in cc_map_groups.items():
        if len(file_list) < 4:
            logging.warning(
                "Group '%s' has less than 4 files (%d). Skipping...",
                suffix,
                len(file_list),
            )
            continue
        if len(file_list) > 4:
            logging.warning(
                "Group '%s' has more than 4 files (%d). Something is wrong with this group.",
                suffix,
                len(file_list),
            )
            failed_groups.append(suffix)
            continue

        averaged_cube, header = load_and_average(file_list)
        cc_hdr = header.copy()
        if parangs_lookup is not None:
            xp, fp = parangs_lookup
            t_batch = parse_batch_utc_seconds_from_suffix(suffix)
            if t_batch < float(xp[0]) or t_batch > float(xp[-1]):
                logging.warning(
                    "Batch time for suffix %s is outside PARANGS_LOOKUP time span; "
                    "np.interp will clamp to endpoints.",
                    suffix,
                )
            parang = float(np.interp(t_batch, xp, fp))
            averaged_cube = derotate_cc_cube(averaged_cube, parang)
            cc_hdr["PARANG_I"] = (parang, "interpolated PA (deg)")
            cc_hdr["DEROT_DEG"] = (-parang, "ndimage.rotate angle on CC cube (deg)")
        else:
            cc_hdr["DEROTATE"] = (False, "PARANGS lookup unavailable; not derotated")

        bias_file_list = bias_groups[suffix]
        averaged_bias, bias_header = load_and_average(bias_file_list)
        hp_filter_fwhm = config_params.get("HIGH_PASS_FWHM", None)
        template_size = int(config_params.get("TEMPLATE_SIZE", DEFAULT_TEMPLATE_SIZE))
        process_distill_group(
            suffix,
            averaged_cube,
            averaged_bias,
            bias_header or header,
            distilled_dir,
            template_size,
            save_pngs=save_pngs,
            hp_filter_fwhm=hp_filter_fwhm,
            cc_header=cc_hdr,
        )
        processed_groups.append(suffix)

    return {
        "distill_dir": distilled_dir,
        "processed_groups": processed_groups,
        "failed_groups": failed_groups,
    }


def run_distill_stage_in_memory(directory, xcorr_result, config_params=None, save_pngs=True):
    """Run distill for one realtime batch from in-memory xcorr products."""
    if config_params is None:
        config_path = os.path.join(directory, "ws_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No `ws_config.yaml` file found at {config_path}. "
                "The pipeline is being run out of order or in the wrong directory."
            )
        config_params = parse_config_file(config_path)

    distilled_dir_name = config_params.get("DISTILL_DIR", "distill_results")
    distilled_dir = (
        distilled_dir_name
        if os.path.isabs(distilled_dir_name)
        else os.path.join(directory, distilled_dir_name)
    )
    ensure_distill_output_dirs(distilled_dir)

    quadrant_results = xcorr_result.get("quadrant_results", {})
    suffix = xcorr_result.get("group_suffix")
    if not suffix or len(quadrant_results) < 4:
        return {
            "distill_dir": distilled_dir,
            "processed_groups": [],
            "failed_groups": [suffix] if suffix else [],
        }

    parangs_path = resolve_parangs_lookup_path(config_params, directory)
    parangs_lookup = load_parangs_lookup(parangs_path) if parangs_path is not None else None

    cc_cubes = [quadrant_results[quadrant]["cc_cube"] for quadrant in ("ul", "ur", "ll", "lr")]
    biases = [quadrant_results[quadrant]["bias"] for quadrant in ("ul", "ur", "ll", "lr")]
    header = quadrant_results["ul"]["header"]
    averaged_cube = np.mean(np.stack(cc_cubes, axis=0), axis=0)
    cc_hdr = header.copy()
    if parangs_lookup is not None:
        xp, fp = parangs_lookup
        t_batch = parse_batch_utc_seconds_from_suffix(suffix)
        if t_batch < float(xp[0]) or t_batch > float(xp[-1]):
            logging.warning(
                "Batch time for suffix %s is outside PARANGS_LOOKUP time span; "
                "np.interp will clamp to endpoints.",
                suffix,
            )
        parang = float(np.interp(t_batch, xp, fp))
        # it's positive parang here
        averaged_cube = derotate_cc_cube(averaged_cube, parang)
        cc_hdr["PARANG_I"] = (parang, "interpolated PA (deg)")
        cc_hdr["DEROT_DEG"] = (-parang, "ndimage.rotate angle on CC cube (deg)")
    else:
        cc_hdr["DEROTATE"] = (False, "PARANGS lookup unavailable; not derotated")

    averaged_bias = np.mean(np.stack(biases, axis=0), axis=0)
    hp_filter_fwhm = config_params.get("HIGH_PASS_FWHM", None)
    template_size = int(config_params.get("TEMPLATE_SIZE", DEFAULT_TEMPLATE_SIZE))
    process_distill_group(
        suffix,
        averaged_cube,
        averaged_bias,
        header,
        distilled_dir,
        template_size,
        save_pngs=save_pngs,
        hp_filter_fwhm=hp_filter_fwhm,
        cc_header=cc_hdr,
    )
    return {
        "distill_dir": distilled_dir,
        "processed_groups": [suffix],
        "failed_groups": [],
    }

def main():
    parser = argparse.ArgumentParser(
        description=("Process a directory of FITS cubes that start with "
                     "'ul_', 'ur_', 'll_', 'lr_' and average each set of 4 cubes into one cube. "
                     "The new filename is the old filename with the prefix removed.")
    )
    parser.add_argument("-d","--directory",
                        type=str,
                        help="Directory containing the camwfs sub-directories", default=".")
    args = parser.parse_args()

    run_distill_stage(args.directory, save_pngs=True)

if __name__ == "__main__":
    main()
