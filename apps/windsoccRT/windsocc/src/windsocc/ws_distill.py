"""Distill stage for the WindsoCC pipeline."""

#!/usr/bin/env python3
import argparse
import logging
import os

import numpy as np

from windsocc.io.config_handling import parse_config_file
from windsocc.io.fits_handling import load_and_average
from windsocc.utils.timestamps import parse_batch_utc_seconds_from_suffix
from windsocc.core.distill import (
    DEFAULT_TEMPLATE_SIZE,
    derotate_cc_cube,
    ensure_distill_output_dirs,
    group_files_by_suffix,
    load_parangs_lookup,
    process_distill_group,
    resolve_parangs_lookup_path,
)

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


if __name__ == '__main__':
    main()
