import logging
from pathlib import Path
from deeplabcut import DEBUG
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from scholl_lab.dlc_utils.create_dlc_project_data import fill_in_labelled_data_folder

logger = logging.getLogger(__name__)


def iterate_model(config_path: str | Path, clicker_csv_path: str | Path, source_data_path: str | Path, experimenter: str) -> None:
    config = auxiliaryfunctions.read_config(config_path)
    project_path = Path(config["project_path"]) # load from yaml here

    video_path = project_path / "videos"
    data_path = project_path / "labeled-data"
    shuffles_path = project_path / "training-datasets"
    results_path = project_path / "dlc-models"
    
    # Create common subdirectories for training-datasets
    iteration_path = shuffles_path / "iteration-0"
    iteration_path.mkdir(parents=True, exist_ok=bool(DEBUG))
    logger.info(f"Created training dataset directory: {iteration_path}")
    
    # Create common subdirectories for dlc-models
    model_iteration_path = results_path / "iteration-0"
    model_iteration_path.mkdir(parents=True, exist_ok=bool(DEBUG))
    logger.info(f"Created model directory: {model_iteration_path}")

    # bump the iteration in the config file
    config["iteration"] += 1

    auxiliaryfunctions.write_config(config_path, config)

    logger.info("Processing labeled frames...")
    labeled_frames = fill_in_labelled_data_folder(
        path_to_recording=Path(source_data_path),
        path_to_dlc_project_folder=project_path,
        path_to_image_labels_csv=Path(clicker_csv_path),
        scorer_name=experimenter
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
