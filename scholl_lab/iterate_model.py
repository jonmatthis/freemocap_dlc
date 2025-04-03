import shutil
import logging
from pathlib import Path
from deeplabcut import DEBUG
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from scholl_lab.dlc_utils.create_dlc_project_data import fill_in_labelled_data_folder

logger = logging.getLogger(__name__)


def iterate_model(
    config_path: str | Path,
    clicker_csv_path: str | Path,
    source_data_path: str | Path,
    experimenter: str, 
    overwrite_csv: bool = True,
) -> None:
    config = auxiliaryfunctions.read_config(config_path)
    project_path = Path(config["project_path"])  # load from yaml here

    shuffles_path = project_path / "training-datasets"
    results_path = project_path / "dlc-models"

    # bump the iteration in the config file
    config["iteration"] += 1
    iteration_count = int(config["iteration"])
    logger.info(f"Bumped iteration to: {iteration_count}")

    # Create common subdirectories for training-datasets
    iteration_path = shuffles_path / f"iteration-{iteration_count}"
    iteration_path.mkdir(parents=True, exist_ok=bool(DEBUG))
    logger.info(f"Created training dataset directory: {iteration_path}")

    # Create common subdirectories for dlc-models
    model_iteration_path = results_path / f"iteration-{iteration_count}"
    model_iteration_path.mkdir(parents=True, exist_ok=bool(DEBUG))
    logger.info(f"Created model directory: {model_iteration_path}")

    auxiliaryfunctions.write_config(config_path, config)
    logger.info(f"Saved updated config file: {config_path}")

    if overwrite_csv:
        output_directory = Path(config_path).parent
        stored_csv_path = list(output_directory.glob("*.csv"))[0] # TODO: eventually store this in config
        shutil.copy(clicker_csv_path, stored_csv_path)

    logger.info("Processing labeled frames...")
    labeled_frames = fill_in_labelled_data_folder(
        path_to_recording=Path(source_data_path),
        path_to_dlc_project_folder=project_path,
        path_to_image_labels_csv=Path(clicker_csv_path),
        scorer_name=experimenter,
    )

    # make new training dataset
    logger.info("Creating training dataset...")
    deeplabcut.create_training_dataset(
        config=config_path,
    )

    # rerun model
    logger.info("Training network...")
    deeplabcut.train_network(
        config=config_path,
        epochs=2,
    )


if __name__ == "__main__":
    config_path = "/Users/philipqueen/DLCtest/freemocap_sample_data_test_philip_20250402/config.yaml"
    clicker_csv_path = "/Users/philipqueen/freemocap_data/recording_sessions/freemocap_test_data/skellyclicker_data/2025-04-03_12-15-39_skellyclicker_output.csv"
    source_data_path = (
        Path.home() / "freemocap_data/recording_sessions/freemocap_test_data"
    )
    experimenter = "philip"

    iterate_model(config_path, clicker_csv_path, source_data_path, experimenter)
