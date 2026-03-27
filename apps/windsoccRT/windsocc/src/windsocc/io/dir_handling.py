import os

def allocate_measure_dirs(basedir: str, params_yaml: dict) -> dict:
    """Allocate the directories for the measure results."""
    directories = {}
    measure_directory = os.path.join(
        basedir, params_yaml.get("MEASURE_DIR", "measure_results"))
    directories["measure_directory"] = measure_directory
    distill_directory = os.path.join(
        basedir, params_yaml.get("DISTILL_DIR", "distill_results"))
    directories["distill_directory"] = distill_directory
    rejected_directory = os.path.join(
        measure_directory, "sources_rejected")
    os.makedirs(rejected_directory, exist_ok=True)
    directories["rejected_directory"] = rejected_directory
    if not os.path.exists(distill_directory):
        raise FileNotFoundError(
            f"The distill directory {distill_directory} does not exist. \
            Is the pipeline being run out of order? \
            Please run ws_distill first.")
    # Make the standard measure directories
    movies_dir = os.path.join(measure_directory, "movies")
    os.makedirs(movies_dir, exist_ok=True)
    directories["movies_dir"] = movies_dir
    decay_plots_dir = os.path.join(measure_directory, "decay_plots")
    os.makedirs(decay_plots_dir, exist_ok=True)
    directories["decay_plots_dir"] = decay_plots_dir
    wedgephotometry_plots_dir = os.path.join(measure_directory, "wedgephotometry_plots")
    os.makedirs(wedgephotometry_plots_dir, exist_ok=True)
    directories["wedgephotometry_plots_dir"] = wedgephotometry_plots_dir
    line_sweep_plots_dir = os.path.join(measure_directory, "line_sweep_plots")
    os.makedirs(line_sweep_plots_dir, exist_ok=True)
    directories["line_sweep_plots_dir"] = line_sweep_plots_dir
    roi_masks_dir = os.path.join(measure_directory, "roi_masks")
    os.makedirs(roi_masks_dir, exist_ok=True)
    directories["roi_masks_dir"] = roi_masks_dir
    wind_data_dir = os.path.join(measure_directory, "wind_data")
    os.makedirs(wind_data_dir, exist_ok=True)
    directories["wind_data_dir"] = wind_data_dir

    return directories
